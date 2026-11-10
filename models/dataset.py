
import torch
import torch.nn.functional as F
import numpy as np
import os
from scipy.spatial import cKDTree
import trimesh

def search_nearest_point(point_batch, point_gt):
    num_point_batch, num_point_gt = point_batch.shape[0], point_gt.shape[0]
    point_batch = point_batch.unsqueeze(1).repeat(1, num_point_gt, 1)
    point_gt = point_gt.unsqueeze(0).repeat(num_point_batch, 1, 1)

    distances = torch.sqrt(torch.sum((point_batch-point_gt) ** 2, axis=-1) + 1e-12)
    dis_idx = torch.argmin(distances, axis=1).detach().cpu().numpy()

    return dis_idx


def load_visibility_data(path):
    if not os.path.exists(path):
        return None, None
    raw = np.loadtxt(path, dtype=np.float32)
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    if not np.isfinite(raw).all():
        raise ValueError(f'Visibility-normal data contains non-finite values: {path}')
    if raw.shape[1] >= 6:
        return raw[:, :3].copy(), raw[:, 3:6].copy()
    if raw.shape[1] == 3:
        return None, raw.copy()
    raise ValueError(f'Expected visibility normals with 3 or 6 columns, got {raw.shape}: {path}')


def process_data(data_dir, dataname):
    if os.path.exists(os.path.join(data_dir, dataname) + '.ply'):
        pointcloud = trimesh.load(os.path.join(data_dir, dataname) + '.ply').vertices
        pointcloud = np.asarray(pointcloud)
    elif os.path.exists(os.path.join(data_dir, dataname) + '.xyz'):
        pointcloud = np.loadtxt(os.path.join(data_dir, dataname) + '.xyz')
    else:
        raise FileNotFoundError(f'Only .xyz or .ply point clouds are supported: {data_dir}/{dataname}')
    shape_scale = np.max([np.max(pointcloud[:,0])-np.min(pointcloud[:,0]),np.max(pointcloud[:,1])-np.min(pointcloud[:,1]),np.max(pointcloud[:,2])-np.min(pointcloud[:,2])])
    shape_center = [(np.max(pointcloud[:,0])+np.min(pointcloud[:,0]))/2, (np.max(pointcloud[:,1])+np.min(pointcloud[:,1]))/2, (np.max(pointcloud[:,2])+np.min(pointcloud[:,2]))/2]
    if not np.isfinite(shape_scale) or shape_scale <= 0:
        raise ValueError(f'Cannot normalize a degenerate point cloud: {dataname}')

    normal_path = os.path.join(data_dir, dataname + '_visnormals.xyz')
    normal_positions, normals = load_visibility_data(normal_path)
    pointcloud = pointcloud - shape_center
    pointcloud = pointcloud / shape_scale
    if normal_positions is not None:
        normal_positions = (normal_positions - shape_center) / shape_scale

    POINT_NUM = pointcloud.shape[0] // 60
    POINT_NUM_GT = pointcloud.shape[0] // 60 * 60
    QUERY_EACH = 1000000//POINT_NUM_GT

    point_idx = np.random.choice(pointcloud.shape[0], POINT_NUM_GT, replace = False)
    pointcloud = pointcloud[point_idx,:]
    # print(np.max(pointcloud[:,0]),np.max(pointcloud[:,1]),np.max(pointcloud[:,2]),np.min(pointcloud[:,0]),np.min(pointcloud[:,1]),np.min(pointcloud[:,2]))
    ptree = cKDTree(pointcloud)
    sigmas = []
    for p in np.array_split(pointcloud,100,axis=0):
        d = ptree.query(p,51)
        sigmas.append(d[0][:,-1])

    sigmas = np.concatenate(sigmas)
    sample = []
    sample_near = []

    for i in range(QUERY_EACH):
        scale = 0.25 * np.sqrt(POINT_NUM_GT / 20000)
        tt = pointcloud + scale*np.expand_dims(sigmas,-1) * np.random.normal(0.0, 1.0, size=pointcloud.shape)
        sample.append(tt)
        tt = tt.reshape(-1,POINT_NUM,3)

        sample_near_tmp = []
        for j in range(tt.shape[0]):
            nearest_idx = search_nearest_point(torch.tensor(tt[j]).float().cuda(), torch.tensor(pointcloud).float().cuda())
            nearest_points = pointcloud[nearest_idx]
            nearest_points = np.asarray(nearest_points).reshape(-1,3)
            sample_near_tmp.append(nearest_points)
        sample_near_tmp = np.asarray(sample_near_tmp)
        sample_near_tmp = sample_near_tmp.reshape(-1,3)
        sample_near.append(sample_near_tmp)

    sample = np.asarray(sample)
    sample_near = np.asarray(sample_near)

    np.savez(
        os.path.join(data_dir, dataname)+'.npz',
        sample=sample,
        point=pointcloud,
        sample_near=sample_near,
        normalization_center=np.asarray(shape_center, dtype=np.float32),
        normalization_scale=np.asarray(shape_scale, dtype=np.float32),
        normal_points=(
            np.asarray(normal_positions, dtype=np.float32)
            if normal_positions is not None else np.empty((0, 3), dtype=np.float32)
        ),
        normals=(
            np.asarray(normals, dtype=np.float32)
            if normals is not None else np.empty((0, 3), dtype=np.float32)
        ),
    )


class DatasetNP:
    def __init__(self, conf, dataname):
        super(DatasetNP, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.conf = conf

        self.data_dir = conf.get_string('data_dir')
        self.np_data_name = dataname + '.npz'
        npz_path = os.path.join(self.data_dir, self.np_data_name)
        visnormals_path = os.path.join(self.data_dir, dataname + '_visnormals.xyz')

        required_fields = {
            'sample_near',
            'sample',
            'point',
            'normalization_center',
            'normalization_scale',
            'normal_points',
            'normals',
        }
        if os.path.exists(npz_path):
            with np.load(npz_path) as cached:
                cache_valid = required_fields.issubset(cached.files)
                if os.path.exists(visnormals_path):
                    cache_valid = cache_valid and {'normal_points', 'normals'}.issubset(cached.files)
        else:
            cache_valid = False
        if cache_valid:
            print('Data existing. Loading data...')
        else:
            print('Data not found. Processing data...')
            process_data(self.data_dir, dataname)
        with np.load(npz_path) as load_data:
            self.point = np.asarray(load_data['sample_near']).reshape(-1,3)
            self.sample = np.asarray(load_data['sample']).reshape(-1,3)
            self.point_gt = np.asarray(load_data['point']).reshape(-1,3)
            self.sample_points_num = self.sample.shape[0]-1
            self.normalization_center = np.asarray(load_data['normalization_center'], dtype=np.float32)
            self.normalization_scale = float(load_data['normalization_scale'])
            normal_points_np = np.asarray(load_data['normal_points'], dtype=np.float32)
            normals_np = np.asarray(load_data['normals'], dtype=np.float32)

        self.point = torch.from_numpy(self.point).to(self.device).float()
        self.sample = torch.from_numpy(self.sample).to(self.device).float()
        self.point_gt = torch.from_numpy(self.point_gt).to(self.device).float()

        self.normals_gt = None
        self.normal_points = None
        if normals_np.shape[0]:
            self.normals_gt = torch.from_numpy(normals_np).to(self.device).float()
            if normal_points_np.shape[0]:
                self.normal_points = torch.from_numpy(normal_points_np).to(self.device).float()
            print(f'Loaded visibility normals: {self.normals_gt.shape[0]} entries from {visnormals_path}')
        elif not os.path.exists(visnormals_path):
            print(f'No visibility normals found at {visnormals_path} — training without normal loss.')

        bbox_points = self.point
        if self.normal_points is not None:
            bbox_points = torch.cat((bbox_points, self.normal_points), dim=0)
        self.object_bbox_min = bbox_points.min(dim=0).values.cpu().numpy() - 0.05
        self.object_bbox_max = bbox_points.max(dim=0).values.cpu().numpy() + 0.05
        print('Data bounding box:',self.object_bbox_min,self.object_bbox_max)
        print('NP Load data: End')

    def np_train_data(self, batch_size):
        index_coarse = np.random.choice(10, 1)
        index_fine = np.random.choice(self.sample_points_num//10, batch_size, replace = False)
        index = index_fine * 10 + index_coarse
        points = self.point[index]
        sample = self.sample[index]
        return points, sample, self.point_gt
