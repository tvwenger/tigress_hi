import os
import sys
import pickle

import numpy as np

import pymc as pm
import bayes_spec
import caribou_hi

from bayes_spec import SpecData, Optimize
from caribou_hi import EmissionAbsorptionModel


def main(dirname, idx):
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
    with open(f"data/{dirname}/{idx:06d}.pkl", "rb") as f:
        datum = pickle.load(f)

    # get data
    emission_velocity = datum["x_values_em"]
    emission_spectrum = datum["data_list_em"]
    rms_emission = datum["errors_em"]
    absorption_velocity = datum["x_values"]
    absorption_spectrum = datum["data_list"]
    rms_absorption = datum["errors"]

    # skip if there does not appear to be any signal
    if not np.any(emission_spectrum > 3.0 * rms_emission) and not np.any(absorption_spectrum > 3.0 * rms_absorption):
        result["exception"] = "no apparent signal"
        return result

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
            max_n_clouds=5,
            baseline_degree=0,
            seed=1234,
            verbose=True,
        )
        opt.add_priors(
            prior_log10_NHI=[20.0, 0.5],
            prior_log10_nHI=[1.0, 0.5],
            prior_log10_tkin=[2.0, 0.5],
            prior_log10_n_alpha=[-6.0, 0.5],
            prior_log10_larson_linewidth=[0.2, 0.05],
            prior_larson_power=[0.4, 0.05],
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
            "chains": 4,
            "cores": 4,
            "init_kwargs": fit_kwargs,
            "nuts_kwargs": {"target_accept": 0.8},
        }
        opt.optimize(bic_threshold=10.0, sample_kwargs=sample_kwargs, fit_kwargs=fit_kwargs, approx=False)

        # save BICs and results for each model
        results = {0: {"bic": opt.best_model.null_bic()}}
        for n_gauss, model in opt.models.items():
            results[n_gauss] = {}
            if len(model.solutions) > 1:
                results[n_gauss]["exception"] = "multiple solutions"
            elif len(model.solutions) == 1:
                results[n_gauss]["bic"] = model.bic(solution=0)
                results[n_gauss]["summary"] = pm.summary(model.trace.solution_0)
            else:
                results[n_gauss]["exception"] = "no solution"
        result["results"] = results
        return result

    except Exception as ex:
        result["exception"] = ex
        return result


if __name__ == "__main__":
    dirname = sys.argv[1]
    idx = int(sys.argv[2])
    output = main(dirname, idx)
    if output["exception"] != "":
        print(output["exception"])

    # save results
    outdirname = f"results/{dirname}_results"
    if not os.path.isdir(outdirname):
        os.mkdir(outdirname)
    fname = f"{outdirname}/{idx:06d}.pkl"
    with open(fname, "wb") as f:
        pickle.dump(output, f)
