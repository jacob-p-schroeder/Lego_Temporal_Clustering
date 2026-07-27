# LEGO Set Clustering Pipeline

This project builds a per-set feature table from a Rebrickable-style LEGO
SQLite database, then uses that feature table to explore clustering
structure across time (elbow/silhouette-based `k` selection, temporal
cluster drift, and dimensionality-reduction/VIF diagnostics).

## Project structure

```
.
├── data/
│   ├── build_lego_features.py     # feature engineering script (this is the only .py file)
│   └── db/
│       └── lego.db                # input SQLite database (not included — see below)
├── outputs/                       # created automatically; all plots are written here
├── cluster_determination.ipynb    # elbow / silhouette k selection, DBSCAN eps estimation
├── temporal_clustering.ipynb      # per-window KMeans + DBSCAN agreement, centroid drift over time
├── feature_analysis.ipynb         # VIF multicollinearity check + UMAP embeddings
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Input data

`data/build_lego_features.py` expects a SQLite database at `data/db/lego.db`
containing the standard Rebrickable-style tables:
`sets`, `inventories`, `inventory_parts`, `inventory_minifigs`, `minifigs`,
`parts`, `part_categories`, `colors`.

## Pipeline order

Run these in order — each step's output is consumed by the next.

### 1. Build features — `data/build_lego_features.py`

```bash
cd data
python build_lego_features.py
```

For every set, this computes part-count/diversity/composition statistics
(part & color entropy, technic/printed/transparent/rare-part proportions,
minifig ratios, etc.), buckets sets into multi-year "cohort" windows sized
by `MIN_COHORT_SIZE`/`MAX_WINDOW_YEARS`, and optionally scales the numeric
columns with a `RobustScaler`.

Key config constants at the top of the file:

| Constant | Purpose |
|---|---|
| `MIN_PARTS` | Drop sets with fewer parts than this (0 = no filter) |
| `HISTOGRAM` | Save a `num_parts` distribution histogram to sanity-check the `MIN_PARTS` cutoff |
| `FILTER_SETS` | Restrict to "standard" numbered sets (drops accessory/gear/promo set-number patterns) |
| `NORMALIZE` | Also emit a `RobustScaler`-normalized feature table |
| `RARE_THRESHOLD` | A part is "rare" if it appears in fewer than this many distinct sets |
| `MIN_COHORT_SIZE` / `MAX_WINDOW_YEARS` | Control how years are grouped into clustering windows |

Outputs (written into `data/`, and also written back into `lego.db` as tables):

- `num_parts_hist.png` (if `HISTOGRAM=True`)
- `lego_cleaned_features.csv` / table `set_features`
- `lego_normalized_features.csv` / table `norm_set_features` (if `NORMALIZE=True`)

### 2. Choose k / DBSCAN eps — `cluster_determination.ipynb`

Reads `data/lego_normalized_features.csv`, filters to the second-most-recent
time window, and:
- Runs KMeans across a range of `k` and reports inertia + silhouette score.
- Picks `k` via both an elbow heuristic and the best silhouette score.
- Estimates a DBSCAN `eps` via a k-distance elbow plot.

Plots are saved to `outputs/` (`elbow_plt.png`, `silhouette_plt.png`,
`combined_plt.png`, `cluster_sizes.png`, `k_distance_plt.png`).

### 3. Temporal clustering & drift — `temporal_clustering.ipynb`

Reads `data/lego_normalized_features.csv`, fits KMeans (k=4) and DBSCAN per
time window, compares them (ARI, NMI, noise fraction), then matches
centroids across windows (Hungarian algorithm) to measure centroid drift
both consecutively and against a fixed baseline window, flagging windows
whose drift exceeds `mean + c*std`.

Plots are saved to `outputs/` (`centroid_drift_consecutive.png`,
`centroid_drift_vs_baseline_0.png`).

### 4. Feature diagnostics & UMAP — `feature_analysis.ipynb`

Reads `data/lego_normalized_features.csv`, computes variance inflation
factors (VIF) for the full and the reduced ("independent") feature sets to
check multicollinearity, then produces 2D and 3D UMAP embeddings colored
by time window (overall, per-window small-multiples grid, and multiple 3D
viewing angles).

Plots are saved to `outputs/` (`UMAP_embedding.png`,
`UMAP_grid_by_window.png`, `UMAP_embedding_3d_multiangle.png`).

## Notes

- All three notebooks assume they are run with `data/` and `outputs/` as
  siblings of the notebook's working directory (i.e. run Jupyter from the
  project root).
- The notebooks read `lego_normalized_features.csv`, so `NORMALIZE=True`
  must be set when running `build_lego_features.py`.
- `outputs/` is created implicitly by `matplotlib`'s `savefig`; create it
  ahead of time (`mkdir outputs`) if it doesn't already exist.