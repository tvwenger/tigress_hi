import sys
import pickle

import numpy as np

import pymc as pm
import bayes_spec
import caribou_hi

from bayes_spec import SpecData, Optimize
from caribou_hi import EmissionAbsorptionModel, EmissionAbsorptionPhysicalModel


def fit_model(idx, data):
    result = {
        "idx": idx,
        "exception": "",
        "results": {},
    }
    try:
        # Initialize optimizer
        opt = Optimize(
            EmissionAbsorptionModel,
            data,
            max_n_clouds=8,
            baseline_degree=0,
            bg_temp=3.77,
            seed=1234,
            verbose=True,
        )
        opt.add_priors(
            prior_TB_fwhm=200.0,
            prior_tkin_factor=[2.0, 2.0],
            prior_sigma_log10_NHI=0.5,
            prior_fwhm2=200.0,
            prior_log10_nHI=[0.0, 1.5],
            prior_velocity=[-50.0, 50.0],
            prior_n_alpha=1.0e-6,
            prior_fwhm_L=None,
            prior_baseline_coeffs=None,
        )
        opt.add_likelihood()
        fit_kwargs = {
            "rel_tolerance": 0.005,
            "abs_tolerance": 0.005,
            "learning_rate": 0.001,
        }
        sample_kwargs = {
            "chains": 8,
            "cores": 8,
            "tune": 1000,
            "draws": 1000,
            "n_init": 200_000,
            "init_kwargs": fit_kwargs,
            "nuts_kwargs": {"target_accept": 0.9},
        }
        solve_kwargs = {
            "init_params": "random_from_data",
            "n_init": 10,
            "max_iter": 1_000,
            "kl_div_threshold": 0.1,
        }
        opt.optimize(
            bic_threshold=10.0,
            sample_kwargs=sample_kwargs,
            fit_kwargs=fit_kwargs,
            solve_kwargs=solve_kwargs,
            approx=False,
            start_spread={"velocity_norm": [0.25, 0.75]},
        )

        # save BICs and results for each model
        results = {0: {"bic": opt.best_model.null_bic()}}
        for n_gauss, model in opt.models.items():
            results[n_gauss] = {"bic": np.inf, "solutions": {}}
            if model.solutions is None:
                continue
            for solution in model.solutions:
                # get BIC
                bic = model.bic(solution=solution)

                # get summary
                summary = pm.summary(model.trace[f"solution_{solution}"])

                # check convergence
                converged = summary["r_hat"].max() < 1.05

                if converged and bic < results[n_gauss]["bic"]:
                    results[n_gauss]["bic"] = bic

                # save posterior samples
                data_vars = list(model.trace[f"solution_{solution}"].data_vars)

                # only save posterior samples if converged
                results[n_gauss]["solutions"][solution] = {
                    "bic": bic,
                    "summary": summary,
                    "converged": converged,
                    "trace": model.trace[f"solution_{solution}"][data_vars],
                }

        result["results"] = results
        return result

    except Exception as ex:
        result["exception"] = ex
        return result


def fit_physical_model(idx, data):
    result = {
        "idx": idx,
        "exception": "",
        "results": {},
    }
    try:
        # Initialize optimizer
        opt = Optimize(
            EmissionAbsorptionPhysicalModel,
            data,
            max_n_clouds=8,
            baseline_degree=0,
            depth_nth_fwhm_power=1 / 3,
            bg_temp=3.77,
            seed=1234,
            verbose=True,
        )
        opt.add_priors(
            prior_ff_NHI=1.0e21,
            prior_fwhm2_thermal_fraction=[2.0, 2.0],
            prior_sigma_log10_NHI=0.5,
            prior_fwhm2=200.0,
            prior_velocity=[-50.0, 50.0],
            prior_n_alpha=1.0e-6,
            prior_nth_fwhm_1pc=[1.75, 0.25],
            prior_fwhm_L=None,
            prior_baseline_coeffs=None,
        )
        opt.add_likelihood()
        fit_kwargs = {
            "rel_tolerance": 0.005,
            "abs_tolerance": 0.005,
            "learning_rate": 0.001,
        }
        sample_kwargs = {
            "chains": 8,
            "cores": 8,
            "tune": 1000,
            "draws": 1000,
            "n_init": 200_000,
            "init_kwargs": fit_kwargs,
            "nuts_kwargs": {"target_accept": 0.9},
        }
        solve_kwargs = {
            "init_params": "random_from_data",
            "n_init": 10,
            "max_iter": 1_000,
            "kl_div_threshold": 0.1,
        }
        opt.optimize(
            bic_threshold=10.0,
            sample_kwargs=sample_kwargs,
            fit_kwargs=fit_kwargs,
            solve_kwargs=solve_kwargs,
            approx=False,
            start_spread={"velocity_norm": [0.25, 0.75]},
        )

        # save BICs and results for each model
        results = {0: {"bic": opt.best_model.null_bic()}}
        for n_gauss, model in opt.models.items():
            results[n_gauss] = {"bic": np.inf, "solutions": {}}
            if model.solutions is None:
                continue
            for solution in model.solutions:
                # get BIC
                bic = model.bic(solution=solution)

                # get summary
                summary = pm.summary(model.trace[f"solution_{solution}"])

                # check convergence
                converged = summary["r_hat"].max() < 1.05

                if converged and bic < results[n_gauss]["bic"]:
                    results[n_gauss]["bic"] = bic

                # save posterior samples
                data_vars = list(model.trace[f"solution_{solution}"].data_vars)

                # only save posterior samples if converged
                results[n_gauss]["solutions"][solution] = {
                    "bic": bic,
                    "summary": summary,
                    "converged": converged,
                    "trace": model.trace[f"solution_{solution}"][data_vars],
                }

        result["results"] = results
        return result

    except Exception as ex:
        result["exception"] = ex
        return result


def main(idx):
    print(f"pymc version: {pm.__version__}")
    print(f"bayes_spec version: {bayes_spec.__version__}")
    print(f"caribou_hi version: {caribou_hi.__version__}")

    # load data
    with open(f"{idx}.pkl", "rb") as f:
        datum = pickle.load(f)

    # get data
    emission_velocity = datum["x_values_em"]
    emission_spectrum = datum["data_list_em"]
    rms_emission = datum["errors_em"]
    absorption_velocity = datum["x_values"]
    absorption_spectrum = datum["data_list"]
    rms_absorption = datum["errors"]

    # save
    emission = SpecData(
        emission_velocity,
        emission_spectrum,
        rms_emission,
        xlabel=r"$V_{\rm LSR}$ (km s$^{-1}$)",
        ylabel=r"$T_B$ (mK)",
    )
    absorption = SpecData(
        absorption_velocity,
        absorption_spectrum,
        rms_absorption,
        xlabel=r"$V_{\rm LSR}$ (km s$^{-1}$)",
        ylabel=r"$1-\exp(-\tau)$",
    )
    data = {"emission": emission, "absorption": absorption}

    result = fit_model(idx, data)
    if result["exception"] != "":
        print(result["exception"])
    fname = f"{idx}_em_abs.pkl"
    with open(fname, "wb") as f:
        pickle.dump(result, f)

    result_physical = fit_physical_model(idx, data)
    if result_physical["exception"] != "":
        print(result_physical["exception"])
    fname = f"{idx}_em_abs_physical.pkl"
    with open(fname, "wb") as f:
        pickle.dump(result_physical, f)


if __name__ == "__main__":
    idx = int(sys.argv[1])
    main(idx)
