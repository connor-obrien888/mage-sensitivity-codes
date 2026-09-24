# mage-sensitivity-codes
Notebooks, scripts, and plotting routines for analyzing initial results from MAGE input uncertainty sensitivity analysis. This repo is intended as a companion for the first results paper and as a workspace, not as a package. Any tools developed as part of this work will be integrated into [kaipy](https://kaipy-docs.readthedocs.io/).

Table of Contents:
- `combovideo.py`: Script that generates video frames from the three simulations (Supporting Information Movies 1-3)
- `ensemble_lib.py`: Helper function library used throughout the notebooks. Functions with broad utility have been integrated into [kaipy](https://kaipy-docs.readthedocs.io/).
- `geomag.ipynb`: Geomagnetic indices analysis, including MLT-binned SuperMAG indices figures.
- `GIC.ipynb`: Example notebook detailing GIC proxy codes developed for this study. Also creates Figure 6.
- `mesoscale.ipynb`: Figures pertaining to mesoscale magnetic field perturbations in the ionosphere (Section 3.3)
- `sim_slice.ipynb`: Notebook generating the simulation slice shown in Figure 7.

Requirements are the same as the [official kaiju python environment](https://kaiju-docs.readthedocs.io/en/latest/python/index.html).
