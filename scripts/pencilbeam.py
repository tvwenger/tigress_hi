import sys
import pickle
import dill

import numpy as np

import pymc as pm
import bayes_spec
import caribou_hi

from bayes_spec import SpecData, Optimize
from caribou_hi import EmissionAbsorptionModel


def main(idx):
    print(f"Starting job on idx = {idx}")
    print(f"pymc version: {pm.__version__}")
    print(f"bayes_spec version: {bayes_spec.__version__}")
    print(f"caribou_hi version: {caribou_hi.__version__}")
    result = {
        "idx": idx,
        "exception": "",
        "results": {},
    }

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
        ylabel=r"$\tau$",
    )
    data = {"emission": emission, "absorption": absorption}

    try:
        # Initialize optimizer
        opt = Optimize(
            EmissionAbsorptionModel,
            data,
            max_n_clouds=8,
            baseline_degree=0,
            seed=1234,
            verbose=True,
        )
        opt.add_priors(
            prior_log10_NHI=[20.0, 0.5],
            prior_log10_nHI=[1.0, 0.5],
            prior_log10_tkin=[2.0, 0.5],
            prior_log10_n_alpha=[-6.0, 0.5],
            prior_log10_larson_linewidth=[0.2, 0.1],
            prior_larson_power=[0.4, 0.1],
            prior_velocity=[0.0, 20.0],
            prior_rms_emission=0.1,
            prior_rms_absorption=0.01,
            ordered=False,
        )
        opt.add_likelihood()
        fit_kwargs = {
            "rel_tolerance": 0.01,
            "abs_tolerance": 0.1,
            "learning_rate": 1e-2,
        }
        sample_kwargs = {
            "chains": 8,
            "cores": 8,
            "tune": 2000,
            "draws": 1000,
            "init_kwargs": fit_kwargs,
            "nuts_kwargs": {"target_accept": 0.8},
        }
        opt.optimize(
            bic_threshold=10.0,
            sample_kwargs=sample_kwargs,
            fit_kwargs=fit_kwargs,
            approx=False,
        )

        # save BICs and results for each model
        results = {0: {"bic": opt.best_model.null_bic()}}
        for n_gauss, model in opt.models.items():
            results[n_gauss] = {"bic": np.inf, "solutions": {}}
            for solution in model.solutions:
                # get BIC
                bic = model.bic(solution=solution)

                # get summary
                summary = pm.summary(model.trace[f"solution_{solution}"])

                # check convergence
                converged_chain = len(model.trace[f"solution_{solution}"].chain) > 1
                converged_rhat = summary["r_hat"].max() < 1.05
                converged = converged_chain and converged_rhat

                if converged and bic < results[n_gauss]["bic"]:
                    results[n_gauss]["bic"] = bic

                # save posterior samples for un-normalized params (except baseline)
                data_vars = list(model.trace[f"solution_{solution}"].data_vars)
                data_vars = [
                    data_var
                    for data_var in data_vars
                    if ("baseline" in data_var) or not ("norm" in data_var)
                ]

                # only save posterior samples if converged
                results[n_gauss]["solutions"][solution] = {
                    "bic": bic,
                    "summary": summary,
                    "converged": converged,
                    "trace": (
                        model.trace[f"solution_{solution}"][data_vars].sel(
                            draw=slice(None, None, 10)
                        )
                        if converged
                        else None
                    ),
                }

        result["results"] = results
        return result

    except Exception as ex:
        result["exception"] = ex
        return result


if __name__ == "__main__":
    idx = int(sys.argv[1])
    output = main(idx)
    if output["exception"] != "":
        print(output["exception"])

    # save results
    fname = f"{idx}_pencilbeam.pkl"
    with open(fname, "wb") as f:
        dill.dump(output, f)
