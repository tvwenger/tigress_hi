# tigress_hi
Decomposing synthetic 21-cm spectra from TIGRESS

## Installation

To run the notebooks:

```bash
conda create --name astro-tigress -c conda-forge python pip ipython jupyter
conda activate astro-tigress
pip install https://github.com/tvwenger/astro-tigress/archive/installation.zip
```

To run `caribou_hi` (i.e., SLURM scripts):

```bash
conda create --name caribou_hi -c conda-forge pytensor pymc nutpie pip dill
conda activate caribou_hi
pip install caribou_hi
```

## Analysis

### `radiative_transfer.ipynb`

Extracts physical conditions of neutral gas from TIGRESS simulations, generates FITS cubes, performs radiative transfer, and generates synthetic 21-cm data cubes.

### `scripts/split_data.py`

Splits joint decomposition spectra output from `radiative_transfer.ipynb` into individual pickle files for parallel processing.

### `scripts/run_pencilbeam.py`

Optimizes `caribou_hi` model on single extracted "pencilbeam" spectrum.

### `scripts/run_smoothed.py`

Optimizes`caribou_hi` model on single extracted "smoothed" spectrum.

### `slurm/submit_pencilbeam.sh`

Runs `run_pencilbeam.py` on all "pencilbeam" spectra via SLURM.

### `slurm/submit_smoothed.sh`

Runs `run_smoothed.py` on all "smoothed" spectra via SLURM.
