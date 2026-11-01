# Terrain opening spectra on hexagonal vs square grids

Revision materials for IEEE GRSL manuscript **GRSL-02429-2026**:
*Scale Stability of Terrain Opening Spectra on Hexagonal Versus Square Grids*.

Author: Yinggang Guo (Northwest Institute of Nuclear Technology, Xi'an).
Contact: fariel_gyg@163.com. ORCID: 0000-0002-8207-9941.

This repository is the public artifact for the revision. It contains the
Python code, study-window definitions, and tabulated spectra used in the
letter. Copernicus GLO-30 GeoTIFFs are **not** redistributed (Copernicus
licence); bounding boxes and PROJ strings are in `paper_a/config.yaml`.

## Reproduce the revision tables

```text
conda env create -f paper_a/environment.yml   # or pip install -r paper_a/requirements.txt
# place COP30 clips as paper_a/data/yanhe_loess_cop30.tif
#                     paper_a/data/sichuan_hills_cop30.tif
cd paper_a
python run_revision.py
python run_revision_e2_figures.py
```

CSV/JSON under `paper_a/outputs/revision/` are the numbers in Table I and
Figs. 3–4. The letter source is `grsl_hex_spectrum/main.tex`.

## Licence

Code is released under MIT (see `LICENSE`). DEM data remain Copernicus /
ESA / DLR / Airbus.
