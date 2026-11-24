# pyCMH: Intrinsic/extrinsic embedding for functional remeshing of 3D shapes

Melzi, S. and Marin, R. and Musoni, P. and Bardon, F. and Tarini, M. and Castellani, U., C&G 2020

Python implementation of [CMH](https://github.com/PietroMsn/CMH) (Coordinates Manifold Harmonics), the functional remeshing pipeline of the papers below: the geometry of one shape is transferred onto the connectivity of another.

<p align="center">
<img src="https://raw.githubusercontent.com/PietroMsn/CMH/master/teaser.jpg">
</p>

## Contents
* [Quick-start](#quick-start)
* [Requirements](#requirements)
* [Code](#code)
* [Tests](#tests)
* [License](#license)
* [Citation](#citation)

## Quick-start
The example code runs over the couple of meshes inside the directory:
```
shapes/
```

To run the whole pipeline:
```
python runme.py
```
The remeshed shape is saved in `results/remeshed.obj`. Use `--no-plot` to skip the plots and `--cpd` to enable the Coherent Point Drift module on the hands (off by default, as in the MATLAB code). You can run different couples of shapes by setting their paths and landmarks at the top of `runme.py` (landmarks are 0-based). Meshes can be `.mat` files with the same variables as in the MATLAB code, or `.off`, `.obj` and `.ply` files.

## Requirements
Python 3.10 or above, with NumPy and SciPy only:
```
pip install -r requirements.txt
```
or `pip install .` to install the `cmh` package. Optional parts have their own extras:
* `pip install .[plot]`: the plots (matplotlib);
* `pip install .[cpd]`: the Coherent Point Drift module on the hands (pycpd and the exact geodesics of libigl);
* `pip install .[robust]`: the robust Laplacian and the heat geodesics on the tufted cover (robust-laplacian, potpourri3d).

## Code
| MATLAB (CMH repo) | Python |
|---|---|
| `getShape`, `load_shape`, `lb_basis_surface`, `scaling_shapes` | `cmh/geometry.py` |
| `compute_CMH` | `cmh/basis.py` |
| `waveKernelSignature`, `waveKernelMap`, `hks` | `cmh/descriptors.py` |
| `compute_map_CMH`, `icp_refine` (`minConf_PQN` → L-BFGS) | `cmh/fmap.py` |
| `arap_gradient` (gptoolbox) | `cmh/arap.py` |
| `perform_functional_remeshing`, `arap_dato` | `cmh/remeshing.py` |
| `LocalPat` + CPD2 | `cmh/local_patch.py` |
| `plotCMH`, `plot_comparison_mh_cmh`, `plot_results` | `cmh/plotting.py` |
| gptoolbox mesh functions | `cmh/mesh.py` |

Beyond the MATLAB code, the package also contains:
* `cmh/refine.py`: ZoomOut in the CMH basis and its bijective variant, which refines the maps in the two directions together;
* `cmh/remeshing.py`: `arap_local_global`, the remeshing as local-global ARAP iterations with a point-to-plane data term, per-vertex data weights and rejection of incompatible closest points, and `arap_targets`, ARAP towards fixed targets;
* `cmh/geometry.py`: `laplacian(kind='robust' | 'robust_pc')`, the robust Laplacians of Sharp and Crane (2020) for meshes with several connected components;
* `cmh/landmarks.py`: automatic landmarks at the extremities of the shapes (maxima of the average geodesic distance), matched between two shapes by their geodesic distances and chosen by the local rigidity of the resulting map.

The mesh operations that the MATLAB code takes from gptoolbox (cotangent Laplacian, normals, closest point on a mesh, mesh input/output) are in `cmh/mesh.py`, on NumPy and SciPy only. Against the ones of libigl they give the same distances and closest points, but where a closest point lies on an edge or a vertex the two may pick a different triangle, hence a different normal, so `arap_local_global` can differ slightly from a libigl-based run (on the couple of `runme.py`: 0.8 mm mean, 21 mm at most, on a body of radius 1.1 m).

## Tests
```
python -m pytest
```
`tests/test_sanity.py` and `tests/test_mesh.py` check the code on its own. `tests/test_against_matlab.py` compares every step with the MATLAB code and is skipped until its reference exists: run `matlab/export_reference.m` from the root of the [CMH](https://github.com/PietroMsn/CMH) repository (branch `python-port`), then copy the resulting `tests/reference/reference_cmh.mat` into `tests/reference/` here.

## License
Please check the license terms (also of third parts software) before downloading and/or using the code, the models and the data.
All code and results obtained from it not already covered by other licenses has to be intendend only for non-commercial scientific research purposes.
Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes including, for example, 3D models, movies, or video games.

## Citation
If you use this code, please cite the following:

```bibtex
@inproceedings {cmh20193dor,
title = {{CMH: Coordinates Manifold Harmonics for Functional Remeshing}},
author = {Marin, Riccardo and Melzi, Simone and Musoni, Pietro and Bardon, Filippo and Tarini, Marco and Castellani, Umberto},
booktitle = {Eurographics Workshop on 3D Object Retrieval},
year = {2019},
publisher = {The Eurographics Association}
}
```

```bibtex
@article{2020intrinsic,
  title={Intrinsic/extrinsic embedding for functional remeshing of 3D shapes},
  author={Melzi, Simone and Marin, Riccardo and Musoni, Pietro and Bardon, Filippo and Tarini, Marco and Castellani, Umberto},
  journal={Computers \& Graphics},
  volume={88},
  pages={1--12},
  year={2020},
  publisher={Elsevier}
}
```
