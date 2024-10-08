# tigress_hi
Decomposing synthetic 21-cm spectra from TIGRESS and 21-SPONGE

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

## `docker`

To build and push `docker` container:

```bash
docker build -t tvwenger/caribou_hi:v1.X.X .
docker push tvwenger/caribou_hi:v1.X.X
```

## TIGRESS-HI Analysis

### `radiative_transfer_ncr.ipynb`

Extracts physical conditions of neutral gas from TIGRESS-NCR simulation, generates FITS cubes, performs radiative transfer, and generates synthetic 21-cm data cubes.

### `agd_tigress.ipynb`

Runs `gausspy` automated Gaussian decomposition on TIGRESS-NCR spectra.

### `agd_results.ipynb`

Inspect `gausspy` results.

### `scripts/split_data.py`

Splits TIGRESS spectra output from `radiative_transfer_ncr.ipynb` into individual pickle files for parallel processing.

### `condor/run_caribou.sub`

Execute parallel processing of `caribou_hi` model fitting of TIGRESS data on a system with `condor`. For the three different "types" of spectra (`true`, `annulus`, and `annulus_error`) and three different values of FWHM (`1pix`, `3pix`, `10pix`), submit this file like

```bash
condor_submit spectype=true fwhm=1pix limit=1000 run_caribou.sub
condor_submit spectype=true fwhm=3pix limit=1000 run_caribou.sub
condor_submit spectype=true fwhm=10pix limit=1000 run_caribou.sub

condor_submit spectype=annulus fwhm=1pix limit=1000 run_caribou.sub
condor_submit spectype=annulus fwhm=3pix limit=1000 run_caribou.sub
condor_submit spectype=annulus fwhm=10pix limit=1000 run_caribou.sub

condor_submit spectype=annulus_error fwhm=1pix limit=1000 run_caribou.sub
condor_submit spectype=annulus_error fwhm=3pix limit=1000 run_caribou.sub
condor_submit spectype=annulus_error fwhm=10pix limit=1000 run_caribou.sub
```

### `scripts/run_caribou.sh` and `scripts/run_caribou.py`

Runs `caribou_hi` on a single TIGRESS spectrum. These scripts are invoked by `condor/run_caribou.sub`.

## 21-SPONGE Analysis

### `scripts/split_21sponge.py`

Splits 21-SPONGE spectra output into individual pickle files for parallel processing.

### `slurm/run_caribou.sub`

Execute parallel processing of `caribou_hi` model fitting of 21-SPONGE data on a system with `slurm`.

### `