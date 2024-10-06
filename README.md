# tigress_hi
Decomposing synthetic 21-cm spectra from TIGRESS

## Installation

To run the radiative transfer notebooks:

```bash
conda create --name astro-tigress -c conda-forge python pip ipython jupyter
conda activate astro-tigress
pip install https://github.com/tvwenger/astro-tigress/archive/installation.zip
```

To run the automated gaussian decomposition (`gausspy`) notebooks:

```bash
git clone git://github.com/tvwenger/gausspy.git
cd gausspy
conda env create -n gausspy --file conda-environment.yml
conda activate gausspy
python setup.py install
```

To run `caribou_hi` notebooks:

```bash
conda create --name caribou_hi -c conda-forge pymc pytensor nutpie pip dill
conda activate caribou_hi
pip install caribou_hi
```

## Analysis

### `radiative_transfer_ncr.ipynb`

Extracts physical conditions of neutral gas from TIGRESS-NCR simulation, generates FITS cubes, performs radiative transfer, and generates synthetic 21-cm data cubes.

### `agd.ipynb`

Runs `gausspy` automated Gaussian decomposition on TIGRESS-NCR spectra.

### `agd_results.ipynb`

Inspect `gausspy` results.

### `scripts/split_data.py`

Splits joint decomposition spectra output from `radiative_transfer_ncr.ipynb` into individual pickle files for parallel processing.

### `condor/run_caribou.sub`

Execute parallel processing of `caribou_hi` model fitting on a system with `condor`. For the three different "types" of spectra (`true`, `annulus`, and `annulus_error`) and three different values of FWHM (`1pix`, `3pix`, `10pix`), submit this file like

```bash
condor_submit spectype=true fwhm=1pix run_caribou.sub
condor_submit spectype=true fwhm=3pix run_caribou.sub
condor_submit spectype=true fwhm=10pix run_caribou.sub

condor_submit spectype=annulus fwhm=1pix run_caribou.sub
condor_submit spectype=annulus fwhm=3pix run_caribou.sub
condor_submit spectype=annulus fwhm=10pix run_caribou.sub

condor_submit spectype=annulus_error fwhm=1pix run_caribou.sub
condor_submit spectype=annulus_error fwhm=3pix run_caribou.sub
condor_submit spectype=annulus_error fwhm=10pix run_caribou.sub
```

## Archived Analysis

### `radiative_transfer.ipynb`

Extracts physical conditions of neutral gas from TIGRESS simulation, generates FITS cubes, performs radiative transfer, and generates synthetic 21-cm data cubes.

### `scripts/run_pencilbeam.py`

Optimizes `caribou_hi` model on single extracted "pencilbeam" spectrum.

### `scripts/run_smoothed.py`

Optimizes`caribou_hi` model on single extracted "smoothed" spectrum.

### `slurm/submit_pencilbeam.sh`

Runs `run_pencilbeam.py` on all "pencilbeam" spectra via SLURM.

### `slurm/submit_smoothed.sh`

Runs `run_smoothed.py` on all "smoothed" spectra via SLURM.