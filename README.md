# NeuralPull-Normals

This repository contains the PyTorch NeuralPull surface-reconstruction method with optional supervision from
visibility-based surface normals. The baseline follows the [official PyTorch NeuralPull implementation](https://github.com/mabaorui/NeuralPull-Pytorch),
which implements the [Neural-Pull ICML 2021 paper](https://arxiv.org/abs/2011.13495). The visibility-normal training
extension follows the [DGNS NeuralPull-normals implementation](https://github.com/sbreuils/discreteGeometry-NeuralSurface/tree/master/NeuralPull-normals).

When `<shape>_visnormals.xyz` is present beside `<shape>.ply`, training adds an orientation-invariant normal loss to the
original pull loss. Six-column files (`x y z nx ny nz`) keep each normal paired with its position, and the loss is
evaluated at those positions. Three-column normal-only files remain supported and use randomly sampled surface points.
The default loss weight is `0.1` in `confs/npull.conf`; setting it to `0` disables normal supervision. Without a
visibility-normal file, the method runs as ordinary NeuralPull.

Install PyTorch using the CUDA wheel appropriate for the host, then install the remaining dependencies:

```
pip install -r requirements.txt
python run.py --gpu 0 --conf confs/npull.conf --dataname <shape> --dir <shape>
```

Input point clouds and optional visibility normals go in `data/`; checkpoints, logs and reconstructed meshes are
written below `outs/`. The visibility-normal files are generated from a discrete volume by
[DGtal visibilityLattices](https://github.com/sbreuils/discreteGeometry-NeuralSurface).
For six-column files, the XYZ positions must already be expressed in the same coordinate frame as the point cloud.

The PyTorch NeuralPull implementation is distributed under the MIT license; see `LICENSE`.
