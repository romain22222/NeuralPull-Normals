# -*- coding: utf-8 -*-

import sys
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from models.dataset import DatasetNP
from models.fields import NPullNetwork
import argparse
from pyhocon import ConfigFactory
import os
from shutil import copyfile
import numpy as np
import trimesh
from models.utils import get_root_logger, print_log
import math
from skimage.measure import marching_cubes as sk_marching_cubes
import warnings
warnings.filterwarnings("ignore")


class Runner:
    def __init__(self, args, conf_path, mode='train'):
        self.device = torch.device('cuda')

        # Configuration
        self.conf_path = conf_path
        f = open(self.conf_path)
        conf_text = f.read()
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.np_data_name'] = self.conf['dataset.np_data_name']
        self.base_exp_dir = self.conf['general.base_exp_dir'] + args.dir
        os.makedirs(self.base_exp_dir, exist_ok=True)


        self.dataset_np = DatasetNP(self.conf['dataset'], args.dataname)
        self.dataname = args.dataname
        self.iter_step = 0

        # Training parameters
        self.maxiter = self.conf.get_int('train.maxiter')
        self.save_freq = self.conf.get_int('train.save_freq')
        self.report_freq = self.conf.get_int('train.report_freq')
        self.val_freq = self.conf.get_int('train.val_freq')
        self.batch_size = self.conf.get_int('train.batch_size')
        self.learning_rate = self.conf.get_float('train.learning_rate')
        self.warm_up_end = self.conf.get_float('train.warm_up_end', default=0.0)
        self.eval_num_points = self.conf.get_int('train.eval_num_points')
        self.normal_weight = self.conf.get_float('train.normal_weight', default=0.0)

        self.mode = mode

        # Networks
        self.sdf_network = NPullNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.optimizer = torch.optim.Adam(self.sdf_network.parameters(), lr=self.learning_rate)

        if self.mode[:5] == 'train':
            self.file_backup()

    def train(self):
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        log_file = os.path.join(os.path.join(self.base_exp_dir), f'{timestamp}.log')
        logger = get_root_logger(log_file=log_file, name='outs')
        self.logger = logger
        batch_size = self.batch_size

        res_step = self.maxiter - self.iter_step

        for iter_i in tqdm(range(res_step), dynamic_ncols=True, file=sys.stderr):
            self.update_learning_rate_np(iter_i)

            points, samples, point_gt = self.dataset_np.np_train_data(batch_size)

            samples.requires_grad = True
            gradients_sample = self.sdf_network.gradient(samples).squeeze() # 5000x3
            sdf_sample = self.sdf_network.sdf(samples)                      # 5000x1
            grad_norm = F.normalize(gradients_sample, dim=1)                # 5000x3
            sample_moved = samples - grad_norm * sdf_sample                 # 5000x3

            loss_sdf = torch.linalg.norm((points - sample_moved), ord=2, dim=-1).mean()

            loss = loss_sdf

            if self.normal_weight > 0.0 and self.dataset_np.normals_gt is not None:
                point_gt_sample = point_gt[torch.randint(0, point_gt.shape[0], (batch_size,))]
                point_gt_sample.requires_grad_(True)
                grad_gt = self.sdf_network.gradient(point_gt_sample).squeeze()
                grad_gt_norm = F.normalize(grad_gt, dim=1)
                normals_batch = self.dataset_np.normals_gt[
                    torch.randint(0, self.dataset_np.normals_gt.shape[0], (batch_size,))
                ]
                normals_batch = F.normalize(normals_batch, dim=1)
                loss_normal = (1.0 - torch.abs((grad_gt_norm * normals_batch).sum(dim=-1))).mean()
                loss = loss + self.normal_weight * loss_normal
            else:
                loss_normal = torch.tensor(0.0)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_step += 1
            if self.iter_step % self.report_freq == 0:
                print_log('iter:{:8>d} cd_l1={:.6f} loss_n={:.6f} lr={}'.format(
                    self.iter_step, loss_sdf, loss_normal, self.optimizer.param_groups[0]['lr']), logger=logger)

            if self.iter_step % self.val_freq == 0 and self.iter_step!=0:
                self.validate_mesh(resolution=256, threshold=args.mcubes_threshold, point_gt=point_gt, iter_step=self.iter_step, logger=logger)

            if self.iter_step % self.save_freq == 0 and self.iter_step!=0:
                self.save_checkpoint()


    def validate_mesh(self, resolution=64, threshold=0.0, point_gt=None, iter_step=0, logger=None):

        bound_min = torch.tensor(self.dataset_np.object_bbox_min, dtype=torch.float32)
        bound_max = torch.tensor(self.dataset_np.object_bbox_max, dtype=torch.float32)
        os.makedirs(os.path.join(self.base_exp_dir, 'outputs'), exist_ok=True)
        mesh = self.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold, query_func=lambda pts: -self.sdf_network.sdf(pts))

        mesh.export(os.path.join(self.base_exp_dir, 'outputs', '{:0>8d}_{}.ply'.format(self.iter_step,str(threshold))))


    def update_learning_rate_np(self, iter_step):
        warn_up = self.warm_up_end
        max_iter = self.maxiter
        init_lr = self.learning_rate
        lr =  (iter_step / warn_up) if iter_step < warn_up else 0.5 * (math.cos((iter_step - warn_up)/(max_iter - warn_up) * math.pi) + 1)
        lr = lr * init_lr
        for g in self.optimizer.param_groups:
            g['lr'] = lr

    def extract_fields(self, bound_min, bound_max, resolution, query_func):
        N = 32
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
                        val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
        return u

    def extract_geometry(self, bound_min, bound_max, resolution, threshold, query_func):
        print('Creating mesh with threshold: {}'.format(threshold))
        u = self.extract_fields(bound_min, bound_max, resolution, query_func)
        verts, faces, _, _ = sk_marching_cubes(u, level=0.0)
        b_min = bound_min.cpu().numpy()
        b_max = bound_max.cpu().numpy()
        verts = verts / (resolution - 1.0) * (b_max - b_min) + b_min
        mesh = trimesh.Trimesh(verts, faces)
        return mesh

    def file_backup(self):
        dir_lis = self.conf['general.recording']
        os.makedirs(os.path.join(self.base_exp_dir, 'recording'), exist_ok=True)
        for dir_name in dir_lis:
            cur_dir = os.path.join(self.base_exp_dir, 'recording', dir_name)
            os.makedirs(cur_dir, exist_ok=True)
            files = os.listdir(dir_name)
            for f_name in files:
                if f_name[-3:] == '.py':
                    copyfile(os.path.join(dir_name, f_name), os.path.join(cur_dir, f_name))

        copyfile(self.conf_path, os.path.join(self.base_exp_dir, 'recording', 'config.conf'))

    def load_checkpoint(self, checkpoint_name):
        checkpoint = torch.load(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name), map_location=self.device)
        print(os.path.join(self.base_exp_dir, 'checkpoints', checkpoint_name))
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])

        self.iter_step = checkpoint['iter_step']

    def save_checkpoint(self):
        checkpoint = {
            'sdf_network_fine': self.sdf_network.state_dict(),
            'iter_step': self.iter_step,
        }
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save(checkpoint, os.path.join(self.base_exp_dir, 'checkpoints', 'ckpt_{:0>6d}.pth'.format(self.iter_step)))


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/np_srb.conf')
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--mcubes_threshold', type=float, default=0.0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--dir', type=str, default='gargoyle')
    parser.add_argument('--dataname', type=str, default='gargoyle')
    args = parser.parse_args()

    torch.cuda.set_device(args.gpu)
    runner = Runner(args, args.conf, args.mode)

    if args.mode == 'train':
        runner.train()
    elif args.mode == 'validate_mesh':
        threshs = [-0.001,-0.0025,-0.005,-0.01,-0.02,0.0,0.001,0.0025,0.005,0.01,0.02]
        for thresh in threshs:
            runner.validate_mesh(resolution=256, threshold=thresh)