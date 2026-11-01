GRSL-02429-2026 supplementary code and tables
Yinggang Guo, 1 November 2026

Python 3.12 environment: paper_a (numpy, scipy, rasterio, matplotlib).
Place Copernicus GLO-30 clips in paper_a/data as in config.yaml,
then from paper_a/:

  python run_revision.py
  python run_revision_e2_figures.py
  python run_tile_bootstrap.py

CSV/JSON in outputs/ are the numbers used in Table I and Fig. 3--4.
tile_w1_bootstrap.json: spatial-block bootstrap of nine 30-to-60 m tiles
(B=10000; hex median 95% CI [2.51, 10.62] m; square [10.31, 11.63] m;
hex-minus-square median CI includes 0).
DEM GeoTIFFs are not redistributed (Copernicus licence); window
coordinates are in config.yaml.
