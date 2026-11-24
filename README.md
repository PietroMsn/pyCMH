# pyCMH

Python port of [CMH](https://github.com/PietroMsn/CMH): Coordinates Manifold Harmonics for functional remeshing.

Melzi, S. and Marin, R. and Musoni, P. and Bardon, F. and Tarini, M. and Castellani, U., C&G 2020

## Quick-start
The package needs only NumPy and SciPy (Python 3.10+):
```
pip install -r requirements.txt
```
or `pip install .` to install it as the `cmh` package. Optional parts have their own extras: `pip install .[plot]` for the plots (matplotlib), `.[cpd]` for the Coherent Point Drift module on the hands (pycpd and libigl's exact geodesics), `.[robust]` for the robust Laplacian and the heat geodesics on the tufted cover (robust-laplacian, potpourri3d).

Run the whole pipeline on the couple of meshes in `shapes/` with:
```
python runme.py
```
The remeshed shape is saved in `results/remeshed.obj`. Use `--no-plot` to skip the plots (and run without matplotlib) and `--cpd` to enable the Coherent Point Drift module on the hands (off by default, as in the MATLAB code). Other couples of shapes can be set at the top of `runme.py`; landmarks are 0-based.

## Contents
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
| gptoolbox / libigl mesh functions | `cmh/mesh.py` |

Beyond the MATLAB code, the package also contains:
- `cmh/refine.py`: ZoomOut in the CMH basis and its bijective variant, which refines the maps in the two directions together;
- `cmh/remeshing.py`: `arap_local_global`, the remeshing as local-global ARAP iterations with a point-to-plane data term, per-vertex data weights and rejection of incompatible closest points, and `arap_targets`, ARAP towards fixed targets;
- `cmh/geometry.py`: `laplacian(kind='robust' | 'robust_pc')`, the robust Laplacians of Sharp and Crane (2020) for meshes with several connected components;
- `cmh/landmarks.py`: automatic landmarks at the extremities of the shapes (maxima of the average geodesic distance), matched between two shapes by their geodesic distances and chosen by the local rigidity of the resulting map.

The mesh operations the MATLAB code takes from gptoolbox (cotangent Laplacian, normals, closest point on a mesh, mesh input/output) are in `cmh/mesh.py`, on NumPy and SciPy only; the CPD module uses [pycpd](https://github.com/siavashk/pycpd) and the exact geodesics of [libigl](https://github.com/libigl/libigl-python-bindings). Against libigl's versions these give the same distances and closest points, but where a closest point lies on an edge or a vertex the two may pick a different triangle, hence a different normal, so `arap_local_global` can differ slightly from a libigl-based run (on the pair of `runme.py`: 0.8 mm mean, 21 mm at most, on a body of radius 1.1 m).

## Tests
```
python -m pytest
```
`tests/test_sanity.py` checks the port on its own. `tests/test_against_matlab.py` compares every step with the MATLAB code and is skipped until its reference exists: run `matlab/export_reference.m` from the root of the [CMH](https://github.com/PietroMsn/CMH) repository (branch `python-port`), then copy the resulting `tests/reference/reference_cmh.mat` into `tests/reference/` here.

## License
Please check the license terms (also of third parts software) before downloading and/or using the code, the models and the data.
All code and results obtained from it not already covered by other licenses has to be intendend only for non-commercial scientific research purposes.
Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes including, for example, 3D models, movies, or video games.

## Citation
If you use this code, please cite the following:
```
@inproceedings {cmh20193dor,
title = {{CMH: Coordinates Manifold Harmonics for Functional Remeshing}},
author = {Marin, Riccardo and Melzi, Simone and Musoni, Pietro and Bardon, Filippo and Tarini, Marco and Castellani, Umberto},
booktitle = {Eurographics Workshop on 3D Object Retrieval},
year = {2019},
publisher = {The Eurographics Association}
}
```
