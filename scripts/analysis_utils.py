import os
import pickle
import copy
import itertools

import arviz as az

import numpy as np
from scipy.optimize import least_squares

from kdetools import gaussian_kde

from typing import Iterable

print("analysis_utils version 1.6")

# assumed CMB background temperature
_T_CMB = 3.77

_SIM_PHASES = {
    "CNM": {"color": "blue", "spin_temp_min": 0.0, "spin_temp_max": 400.0},
    "LNM": {"color": "green", "spin_temp_min": 400.0, "spin_temp_max": 4000.0},
    "WNM": {"color": "orange", "spin_temp_min": 4000.0, "spin_temp_max": np.inf},
    "All": {"color": "black", "spin_temp_min": 0.0, "spin_temp_max": np.inf},
}

_CARIBOU_SINGLE_RESULT = {
    "All_log10_NHI_tau": -np.inf,
    "All_log10_NHI_TB": -np.inf,
    "CNM_log10_NHI_tau": -np.inf,
    "CNM_log10_NHI_TB": -np.inf,
    "LNM_log10_NHI_tau": -np.inf,
    "LNM_log10_NHI_TB": -np.inf,
    "WNM_log10_NHI_tau": -np.inf,
    "WNM_log10_NHI_TB": -np.inf,
    "CNM_fraction_tau": np.nan,
    "CNM_fraction_TB": np.nan,
    "LNM_fraction_tau": np.nan,
    "LNM_fraction_TB": np.nan,
    "WNM_fraction_tau": np.nan,
    "WNM_fraction_TB": np.nan,
    "tau_weighted_tspin_tau": np.nan,
    "tau_weighted_tspin_TB": np.nan,
    "log10_NHI": [],
    "log10_depth": [],
    "log10_pressure": [],
    "log10_nHI": [],
    "log10_tkin": [],
    "tspin": [],
    "fwhm_thermal": [],
    "fwhm_nonthermal": [],
    "tau_weights": [],
    "TB_weights": [],
    "filling_factor": [],
    "absorption_weight": [],
    "log10_NHI_samples": [],
    "log10_depth_samples": [],
    "log10_pressure_samples": [],
    "log10_nHI_samples": [],
    "log10_tkin_samples": [],
    "tspin_samples": [],
    "fwhm_thermal_samples": [],
    "fwhm_nonthermal_samples": [],
    "tau_weights_samples": [],
    "TB_weights_samples": [],
    "filling_factor_samples": [],
    "absorption_weight_samples": [],
    "BIC": np.nan,
    "n_gauss": 0,
    "n_solutions": np.nan,
}

_CARIBOU_RESULT = {
    "median": copy.deepcopy(_CARIBOU_SINGLE_RESULT),
    "eti_16%": copy.deepcopy(_CARIBOU_SINGLE_RESULT),
    "eti_84%": copy.deepcopy(_CARIBOU_SINGLE_RESULT),
    "predicted_absorption": {},
    "predicted_emission": {},
}

_GAUSSPY_SINGLE_RESULT = {
    "All_log10_NHI": -np.inf,
    "CNM_log10_NHI": -np.inf,
    "LNM_log10_NHI": -np.inf,
    "WNM_log10_NHI": -np.inf,
    "CNM_fraction": np.nan,
    "LNM_fraction": np.nan,
    "WNM_fraction": np.nan,
    "tau_weighted_tspin": np.nan,
    "log10_NHI": [],
    "log10_tkin": [],
    "tspin": [],
    "rchi2": np.nan,
    "n_gauss_abs": 0,
    "n_gauss_em": 0,
}

_GAUSSPY_RESULT = {
    "mean": copy.deepcopy(_GAUSSPY_SINGLE_RESULT),
    "sd": copy.deepcopy(_GAUSSPY_SINGLE_RESULT),
    "predicted_absorption": {},
    "predicted_absorption_em_ax": {},
    "predicted_emission": {},
}


def gaussian(x, amp, center, fwhm):
    return amp * np.exp(-4.0 * np.log(2.0) * (x - center) ** 2.0 / fwhm**2.0)


def calc_line_profile(
    velo_axis: Iterable[float], velocity: Iterable[float], fwhm: Iterable[float]
) -> Iterable[float]:
    """Evaluate the Gaussian line profile, ensuring normalization.

    Parameters
    ----------
    velo_axis : Iterable[float]
        Observed velocity axis (km s-1; length S)
    velocity : Iterable[float]
        Cloud center velocity (km s-1; length C x N)
    fwhm : Iterable[float]
        Cloud FWHM line widths (km s-1; length C x N)

    Returns
    -------
    Iterable[float]
        Line profile (MHz-1; shape S x N)
    """
    amp = np.sqrt(4.0 * np.log(2.0) / (np.pi * fwhm**2.0))
    profile = gaussian(velo_axis[:, None], amp, velocity, fwhm)

    # normalize
    channel_size = np.abs(velo_axis[1] - velo_axis[0])
    profile_int = np.sum(profile, axis=0)
    norm = profile_int * channel_size
    norm[profile_int < 1.0e-6] = 1.0
    return profile / norm


def calc_optical_depth(
    velo_axis: Iterable[float],
    velocity: Iterable[float],
    NHI: Iterable[float],
    tspin: Iterable[float],
    fwhm: Iterable[float],
) -> Iterable[float]:
    """Evaluate the optical depth spectra following Marchal et al. (2019) eq. 15
    assuming a homogeneous and isothermal cloud.

    Parameters
    ----------
    velo_axis : Iterable[float]
        Observed velocity axis (km s-1) (length S)
    velocity : Iterable[float]
        Cloud velocities (km s-1) (length N)
    NHI : Iterable[float]
        HI column density (cm-2) (length N)
    tspin : Iterable[float]
        Spin tempearture (K) (length N)
    fwhm : Iterable[float]
        FWHM line width (km s-1)

    Returns
    -------
    Iterable[float]
        Optical depth spectra (shape S x N)
    """
    # Evaluate line profile
    line_profile = calc_line_profile(velo_axis, velocity, fwhm)

    # Evaluate the optical depth spectra
    const = 1.82243e18  # cm-2 (K km s-1)-1
    optical_depth = NHI * line_profile / tspin / const
    return optical_depth


def radiative_transfer(
    tau: Iterable[float],
    tspin: Iterable[float],
    filling_factor: Iterable[float],
    bg_temp: float,
) -> Iterable[float]:
    """Evaluate the radiative transfer to predict the emission spectrum. The emission
    spectrum is ON - OFF, where ON includes the attenuated emission of the background and
    the clouds, and the OFF is the emission of the background. Order of N clouds is
    assumed to be [nearest, ..., farthest]. The contribution of each cloud is diluted by the
    filling factor, a number between zero and one.

    Parameters
    ----------
    tau : Iterable[float]
        Optical depth spectra (shape S x ... x N)
    tspin : Iterable[float]
        Spin temperatures (K) (shape ... x N)
    filling_factor : Iterable[float]
        Filling factor (between zero and one) (shape ... x N)
    bg_temp : float
        Assumed background temperature

    Returns
    -------
    Iterable[float]
        Predicted emission brightness temperature spectrum (K) (length S)
    """
    front_tau = np.zeros_like(tau[..., 0:1])
    # cumulative optical depth through clouds
    sum_tau = np.concatenate([front_tau, np.cumsum(tau, axis=-1)], axis=-1)

    # radiative transfer, assuming filling factor = 1.0
    emission_bg_attenuated = bg_temp * np.exp(-sum_tau[..., -1])
    emission_clouds = filling_factor * tspin * (1.0 - np.exp(-tau))
    emission_clouds_attenuated = emission_clouds * np.exp(-sum_tau[..., :-1])
    emission = emission_bg_attenuated + emission_clouds_attenuated.sum(axis=-1)

    # ON - OFF
    return emission - bg_temp


def get_best_model(result, bic_threshold=10.0):
    """Determine the best caribou_hi result, and return the best solution."""
    # keep only best model
    best_bic = np.inf
    best_n_gauss = 0
    best_solution = 0
    best_num_solutions = 0
    for n_gauss in result["results"].keys():
        this_bic = np.inf
        this_solution = None
        this_num_solutions = 0
        if "bic" in result["results"][n_gauss]:
            this_bic = result["results"][n_gauss]["bic"]

        if "solutions" in result["results"][n_gauss].keys():
            this_num_solutions = len(result["results"][n_gauss]["solutions"])
            for solution in result["results"][n_gauss]["solutions"].keys():
                converged = result["results"][n_gauss]["solutions"][solution][
                    "converged"
                ]
                bic = result["results"][n_gauss]["solutions"][solution]["bic"]
                if converged and bic <= this_bic:
                    this_bic = bic
                    this_solution = solution
        if np.isinf(best_bic) or this_bic < (best_bic - bic_threshold):
            best_bic = this_bic
            best_n_gauss = n_gauss
            best_solution = this_solution
            best_num_solutions = this_num_solutions
    if best_n_gauss in result["results"].keys():
        result["results"] = result["results"][best_n_gauss]
        if "solutions" in result["results"].keys() and best_solution is not None:
            result["results"] = result["results"]["solutions"][best_solution]
            return result, best_bic, best_n_gauss, best_num_solutions
    return None, best_bic, best_n_gauss, best_num_solutions


def compile_caribou_results(
    fname,
    x_values,
    x_values_em,
    predict=True,
    condition_ff=False,
    thin=10,
):
    """Compile results from a caribou_hi output"""
    result = copy.deepcopy(_CARIBOU_RESULT)

    if not os.path.exists(fname):
        print(fname)
        return result

    with open(fname, "rb") as f:
        output = pickle.load(f)

    # get best result
    output, bic, n_gauss, n_sols = get_best_model(output)
    result["median"]["BIC"] = bic
    result["median"]["n_gauss"] = n_gauss
    result["median"]["n_solutions"] = n_sols

    if output is None:
        return result

    trace = output["results"]["trace"]

    if condition_ff and "filling_factor" in trace:
        # Condition posterior on filling_factor = 1 and absorption weight = 1 by fitting a KDE to
        # posterior samples.
        varnames = [
            "log10_NHI",
            "log10_depth",
            "log10_pressure",
            "tspin",
            "fwhm_thermal",
            "fwhm_nonthermal",
            "filling_factor",
            "absorption_weight",
        ]
        conditional_trace = {
            varname: np.ones(trace[varname].shape) * np.nan for varname in varnames
        }
        for cloud in trace.cloud:
            samples = az.extract(trace[varnames].sel(cloud=cloud))
            median_tspin = samples["tspin"].median()
            samples = np.stack([samples[varname].to_numpy() for varname in varnames])
            if median_tspin > _SIM_PHASES["WNM"]["spin_temp_min"]:
                kde = gaussian_kde(samples)
                x_cond = np.array([[1.0]])
                dims_cond = [varnames.index("filling_factor")]
                conditional_samples = kde.conditional_resample(
                    samples.shape[1], x_cond=x_cond, dims_cond=dims_cond
                )[0]
                new_varnames = [
                    varname for varname in varnames if varname not in ["filling_factor"]
                ]
                for i, varname in enumerate(new_varnames):
                    conditional_trace[varname][:, :, cloud] = conditional_samples[
                        :, i
                    ].reshape(conditional_trace[varname].shape[0:2])
                conditional_trace["filling_factor"][:, :, cloud] = np.ones(
                    conditional_trace["filling_factor"].shape[0:2]
                )
            else:
                for i, varname in enumerate(varnames):
                    conditional_trace[varname][:, :, cloud] = samples[i].reshape(
                        conditional_trace[varname].shape[0:2]
                    )

        # derived quantities
        conditional_trace["log10_nHI"] = (
            conditional_trace["log10_NHI"] - conditional_trace["log10_depth"] - 18.48935
        )
        conditional_trace["log10_tkin"] = (
            conditional_trace["log10_pressure"]
            - conditional_trace["log10_nHI"]
            - np.log10(1.1)
        )

        # replace samples in trace
        for varname in varnames:
            trace[varname].data = conditional_trace[varname]

    # For phase fractions, we have to assign each cloud to a phase. Clouds
    # can't move between phases because the posterior sample dimensions would
    # not be constant. Instead we investigate posterior predictive samples of
    # the cumulative distribution function(s)
    median_tspin = trace["tspin"].median(dim=["chain", "draw"])

    for phase in _SIM_PHASES.keys():
        # find clouds in this phase
        clouds = np.array(
            [
                i
                for i in range(len(trace.cloud))
                if (median_tspin[i] > _SIM_PHASES[phase]["spin_temp_min"])
                and (median_tspin[i] <= _SIM_PHASES[phase]["spin_temp_max"])
            ]
        )

        # save column density
        trace[f"{phase}_log10_NHI_tau"] = np.log10(
            (
                (trace["absorption_weight"] if "absorption_weight" in trace else 1.0)
                * 10.0 ** trace["log10_NHI"]
            )
            .sel(cloud=clouds)
            .sum(dim="cloud")
        )
        trace[f"{phase}_log10_NHI_TB"] = np.log10(
            (
                (trace["filling_factor"] if "filling_factor" in trace else 1.0)
                * 10.0 ** trace["log10_NHI"]
            )
            .sel(cloud=clouds)
            .sum(dim="cloud")
        )

    # add phase fractions to trace
    for phase in ["CNM", "LNM", "WNM"]:
        trace[f"{phase}_fraction_tau"] = (
            10.0 ** trace[f"{phase}_log10_NHI_tau"] / 10.0 ** trace["All_log10_NHI_tau"]
        )
        trace[f"{phase}_fraction_TB"] = (
            10.0 ** trace[f"{phase}_log10_NHI_TB"] / 10.0 ** trace["All_log10_NHI_TB"]
        )

    # add tau weights and TB weights to trace
    filling_factor = trace["filling_factor"] if "filling_factor" in trace else 1.0
    absorption_weight = (
        trace["absorption_weight"] if "absorption_weight" in trace else 1.0
    )
    trace["tau_weights"] = (
        absorption_weight
        * 10.0 ** trace["log10_nHI"]
        / trace["tspin"]
        / np.sqrt(trace["fwhm_thermal"] ** 2.0 + trace["fwhm_nonthermal"] ** 2.0)
    )
    trace["TB_weights"] = (
        filling_factor
        * 10.0 ** trace["log10_nHI"]
        / np.sqrt(trace["fwhm_thermal"] ** 2.0 + trace["fwhm_nonthermal"] ** 2.0)
    )
    trace["tau_weighted_tspin_tau"] = (trace["tspin"] * trace["tau_weights"]).sum(
        dim="cloud"
    ) / trace["tau_weights"].sum(dim="cloud")
    trace["tau_weighted_tspin_TB"] = (trace["tspin"] * trace["TB_weights"]).sum(
        dim="cloud"
    ) / trace["TB_weights"].sum(dim="cloud")

    # "mean" is not a good statistic for highly-skewed distributions
    # Let's use median statistics

    # save phase column densities, fractions
    varnames = [
        "All_log10_NHI_tau",
        "All_log10_NHI_TB",
        "CNM_log10_NHI_tau",
        "CNM_log10_NHI_TB",
        "LNM_log10_NHI_tau",
        "LNM_log10_NHI_TB",
        "WNM_log10_NHI_tau",
        "WNM_log10_NHI_TB",
        "CNM_fraction_tau",
        "CNM_fraction_TB",
        "LNM_fraction_tau",
        "LNM_fraction_TB",
        "WNM_fraction_tau",
        "WNM_fraction_TB",
        "tau_weighted_tspin_tau",
        "tau_weighted_tspin_TB",
    ]
    summary = az.summary(
        trace,
        var_names=varnames,
        kind="stats",
        stat_focus="median",
        hdi_prob=0.68,
    )
    for stat in ["median", "eti_16%", "eti_84%"]:
        for varname in varnames:
            result[stat][varname] = summary[stat][varname]

    # save physical conditions and weights
    varnames = [
        "log10_NHI",
        "log10_depth",
        "log10_pressure",
        "log10_nHI",
        "log10_tkin",
        "tspin",
        "fwhm_thermal",
        "fwhm_nonthermal",
        "tau_weights",
        "TB_weights",
    ]
    if "filling_factor" in trace:
        varnames += ["filling_factor"]
    if "absorption_weight" in trace:
        varnames += ["absorption_weight"]
    for cloud in range(len(trace.cloud)):
        summary = az.summary(
            trace.sel(cloud=cloud).drop_vars("cloud"),
            var_names=varnames,
            kind="stats",
            stat_focus="median",
            hdi_prob=0.68,
            coords={"draw": trace.draw, "chain": trace.chain},
        )
        samples = az.extract(trace[varnames]).sel(
            cloud=cloud, sample=slice(None, None, thin)
        )
        for varname in varnames:
            for stat in ["median", "eti_16%", "eti_84%"]:
                result[stat][varname].append(summary[stat][varname])
            result["median"][f"{varname}_samples"].append(samples[varname].to_numpy())

    if predict:
        # evaluate posterior predictive samples to predict emission and absorption
        varnames = ["velocity", "log10_NHI", "tspin", "fwhm"]
        if "absorption_weight" in trace:
            varnames += ["absorption_weight"]
        if "filling_factor" in trace:
            varnames += ["filling_factor"]
        trace = az.extract(trace[varnames]).sel(sample=slice(None, None, thin))

        absorption_weight = (
            trace["absorption_weight"].data.T if "absorption_weight" in trace else 1.0
        )
        absorption_optical_depth = absorption_weight * calc_optical_depth(
            x_values[:, None],
            trace["velocity"].data.T,
            10.0 ** trace["log10_NHI"].data.T,
            trace["tspin"].data.T,
            trace["fwhm"].data.T,
        )

        for cloud in trace.cloud.data:
            result["predicted_absorption"][cloud] = 1.0 - np.exp(
                -absorption_optical_depth[:, :, cloud]
            )

        emission_optical_depth = calc_optical_depth(
            x_values_em[:, None],
            trace["velocity"].data.T,
            10.0 ** trace["log10_NHI"].data.T,
            trace["tspin"].data.T,
            trace["fwhm"].data.T,
        )

        for cloud in trace.cloud.data:
            cloud_tspin = trace["tspin"].data.copy()
            this_cloud = np.zeros(len(trace.cloud), dtype=bool)
            this_cloud[cloud] = True
            cloud_tspin[~this_cloud] = 0.0
            filling_factor = (
                trace["filling_factor"].data.T if "filling_factor" in trace else 1.0
            )
            result["predicted_emission"][cloud] = radiative_transfer(
                emission_optical_depth,
                cloud_tspin.T,
                filling_factor,
                _T_CMB,
            )

    return result


def compile_gausspy_results(
    fname,
    data,
    pbar,
    num_samples=100,
    seed=1234,
):
    """Compile gausspy results"""
    rng = np.random.RandomState(seed=seed)
    results = {}

    with open(fname, "rb") as f:
        agd_results = pickle.load(f)

    results = {}
    for idx in range(len(agd_results["amplitudes_fit_em"])):
        results[idx] = copy.deepcopy(_GAUSSPY_RESULT)

        n_gauss_abs = results[idx]["mean"]["n_gauss_abs"] = len(
            agd_results["amplitudes_fit"][idx]
        )
        n_gauss_em = results[idx]["mean"]["n_gauss_em"] = len(
            agd_results["amplitudes_fit_em"][idx]
        )
        results[idx]["mean"]["rchi2"] = agd_results["best_fit_rchi2"][idx][0]

        results[idx]["predicted_absorption"] = gaussian(
            data["x_values"][idx][:, None],
            np.array(agd_results["amplitudes_fit"][idx]),
            np.array(agd_results["means_fit"][idx]),
            np.array(agd_results["fwhms_fit"][idx]),
        )

        results[idx]["predicted_absorption_em_ax"] = gaussian(
            data["x_values_em"][idx][:, None],
            np.array(agd_results["amplitudes_fit"][idx]),
            np.array(agd_results["means_fit"][idx]),
            np.array(agd_results["fwhms_fit"][idx]),
        )

        results[idx]["predicted_emission"] = gaussian(
            data["x_values_em"][idx][:, None],
            np.array(agd_results["amplitudes_fit_em"][idx]),
            np.array(agd_results["means_fit_em"][idx]),
            np.array(agd_results["fwhms_fit_em"][idx]),
        )

        good_chans = np.where(
            data["data_list_em"][idx] > (3.0 * data["errors_em"][idx])
        )[0]
        if len(good_chans) > 0:
            start_chan = good_chans[0]
            end_chan = good_chans[-1]
            tau_weighted_tspin = np.sum(
                data["data_list"][idx][start_chan:end_chan]
                * data["data_list_em"][idx][start_chan:end_chan]
                / (1.0 - np.exp(-data["data_list"][idx][start_chan:end_chan]))
            ) / np.sum(data["data_list"][idx][start_chan:end_chan])
        else:
            tau_weighted_tspin = np.nan
        results[idx]["mean"]["tau_weighted_tspin"] = tau_weighted_tspin

        # catch missing errors. Why are there missing errors?
        for key in agd_results.keys():
            assumed_errors = {
                "amplitudes_fit_err": 0.1,
                "amplitudes_fit_err_em": 1.0,
                "fwhms_fit_err": 5.0,
                "fwhms_fit_err_em": 5.0,
            }
            if key in assumed_errors.keys():
                agd_results[key][idx] = [
                    val if val is not None else assumed_errors[key]
                    for val in agd_results[key][idx]
                ]

        # kinetic temperature upper limit
        fwhms_abs = list(agd_results["fwhms_fit"][idx])
        fwhms_abs_err = list(agd_results["fwhms_fit_err"][idx])
        fwhms_em = list(agd_results["fwhms_fit_em"][idx][n_gauss_abs:n_gauss_em])
        fwhms_em_err = list(
            agd_results["fwhms_fit_err_em"][idx][n_gauss_abs:n_gauss_em]
        )
        fwhms = np.array(fwhms_abs + fwhms_em)
        fwhms_err = np.array(fwhms_abs_err + fwhms_em_err)
        tkin_max = 21.866 * fwhms**2.0  # K
        log10_tkin_max_err = fwhms_err * 2.0 / np.log(10.0) / fwhms
        results[idx]["mean"]["log10_tkin"] += list(np.log10(tkin_max))
        results[idx]["sd"]["log10_tkin"] += list(log10_tkin_max_err)

        # HT03 method for spin temperature estimate
        # emission-only contribution, permutations on Fk
        total_tau = np.sum(results[idx]["predicted_absorption_em_ax"], axis=1)
        Fks = np.array([0.0, 0.5, 1.0])
        all_Fks = itertools.product(Fks, repeat=n_gauss_em)
        # YIKES
        TB_em_AGD = np.array(
            [
                np.sum(
                    [
                        (Fk[i] + (1.0 - Fk[i]) * np.exp(-total_tau))
                        * results[idx]["predicted_emission"][:, i]
                        for i in range(n_gauss_abs, n_gauss_em)
                    ],
                    axis=0,
                )
                for Fk in all_Fks
            ]
        )

        # absorption-only contribution, permutations on cloud order
        cloud_orders = list(itertools.permutations(np.arange(n_gauss_abs)))

        # sample random subset of trial permutations
        TB_em_AGD_sample_idxs = rng.randint(0, len(TB_em_AGD), size=num_samples)
        cloud_order_sample_idxs = rng.randint(0, len(cloud_orders), size=num_samples)

        spin_temp_trial_temps = np.ones((num_samples, n_gauss_abs)) * np.nan
        spin_temp_trial_temp_errs = np.ones((num_samples, n_gauss_abs)) * np.nan
        spin_temp_trial_residuals = np.ones(num_samples) * np.nan

        for trial_idx, (TB_em_AGD_sample_idx, cloud_order_sample_idx) in enumerate(
            zip(TB_em_AGD_sample_idxs, cloud_order_sample_idxs)
        ):
            cloud_order = cloud_orders[cloud_order_sample_idx]
            TB_em = TB_em_AGD[TB_em_AGD_sample_idx]

            def residual(spin_temp):
                # DOUBLE YIKES
                TB_abs = np.sum(
                    [
                        spin_temp[i]
                        * (
                            1.0
                            - np.exp(-results[idx]["predicted_absorption_em_ax"][:, i])
                        )
                        * np.exp(
                            -np.sum(
                                [
                                    results[idx]["predicted_absorption_em_ax"][:, j]
                                    for j in cloud_order
                                    if j != i
                                ],
                                axis=0,
                            )
                        )
                        for i in cloud_order
                    ],
                    axis=0,
                )
                res = data["data_list_em"][idx] - TB_em - TB_abs
                return res / data["errors_em"][idx]

            try:
                spin_temp_guess = np.ones(len(cloud_order)) * 500.0
                bounds = (
                    np.zeros_like(spin_temp_guess),
                    np.ones_like(spin_temp_guess) * np.inf,
                )
                res = least_squares(residual, spin_temp_guess, bounds=bounds)
                spin_temp_trial_residuals[trial_idx] = np.std(residual(res.x))
                spin_temp_trial_temps[trial_idx] = res.x
                cov = np.linalg.inv(res.jac.T @ res.jac)
                spin_temp_trial_temp_errs[trial_idx] = np.sqrt(np.diag(cov))
            except:
                pass

        bad = np.any(
            np.isnan(spin_temp_trial_temps + spin_temp_trial_temp_errs), axis=1
        )
        weights = spin_temp_trial_residuals[~bad]
        if np.sum(weights) == 0:
            mean_tspin = np.ones(n_gauss_abs) * np.nan
            err_tspin = np.ones(n_gauss_abs) * np.nan
        else:
            mean_tspin = np.average(
                spin_temp_trial_temps[~bad], weights=weights, axis=0
            )
            err_tspin = (
                np.average(
                    (spin_temp_trial_temps - mean_tspin)[~bad] ** 2.0
                    + spin_temp_trial_temp_errs[~bad] ** 2.0,
                    weights=weights,
                    axis=0,
                )
                * num_samples
                / (num_samples - 1.0)
            )
        # catch bad spin temperatures, replace with inf
        isnan = np.isnan(mean_tspin)
        mean_tspin[isnan] = np.inf
        err_tspin[isnan] = np.inf

        results[idx]["mean"]["tspin"] += list(mean_tspin)
        results[idx]["sd"]["tspin"] += list(err_tspin)

        # upper limit on WNM cloud spin temp is kinetic temp
        # results[idx]["mean"]["tspin"] += list(tkin_max[n_gauss_abs:n_gauss_em])
        results[idx]["mean"]["tspin"] += [np.inf] * len(fwhms_em)
        results[idx]["sd"]["tspin"] += [np.inf] * len(fwhms_em)

        # phase column densities
        for phase in _SIM_PHASES.keys():
            # find clouds in this phase
            abs_clouds = np.array(
                [
                    i
                    for i in range(n_gauss_abs)
                    if (
                        results[idx]["mean"]["tspin"][i]
                        > _SIM_PHASES[phase]["spin_temp_min"]
                    )
                    and (
                        results[idx]["mean"]["tspin"][i]
                        <= _SIM_PHASES[phase]["spin_temp_max"]
                    )
                ]
            )

            em_clouds = np.array([])
            if phase in ["WNM", "All"]:
                em_clouds = np.arange(n_gauss_abs, n_gauss_em)

            # column density of each cloud
            # absorption clouds
            # using a noise threshold for the integration to calculate NHI leads to a
            # systematic bias that down-weights NHI for warm (wide) clouds of same NHI
            # this would tend to... overpredict fCNM
            NHI = [
                1.064
                * 1.823e18
                * agd_results["amplitudes_fit"][idx][i]
                * agd_results["fwhms_fit"][idx][i]
                * mean_tspin[i]
                for i in abs_clouds
            ] + [
                1.064
                * 1.823e18
                * agd_results["amplitudes_fit_em"][idx][i]
                * agd_results["fwhms_fit_em"][idx][i]
                for i in em_clouds
            ]
            NHI = np.array(NHI)
            log10_NHI_err = [
                np.sqrt(
                    (
                        agd_results["amplitudes_fit_err"][idx][i]
                        / agd_results["amplitudes_fit"][idx][i]
                        / np.log(10.0)
                    )
                    ** 2.0
                    + (
                        agd_results["fwhms_fit_err"][idx][i]
                        / agd_results["fwhms_fit"][idx][i]
                        / np.log(10.0)
                    )
                    ** 2.0
                    + (err_tspin[i] / mean_tspin[i] / np.log(10.0)) ** 2.0
                )
                for i in abs_clouds
            ] + [
                np.sqrt(
                    (
                        agd_results["amplitudes_fit_err_em"][idx][i]
                        / agd_results["amplitudes_fit_em"][idx][i]
                        / np.log(10.0)
                    )
                    ** 2.0
                    + (
                        agd_results["fwhms_fit_err_em"][idx][i]
                        / agd_results["fwhms_fit_em"][idx][i]
                        / np.log(10.0)
                    )
                    ** 2.0
                )
                for i in em_clouds
            ]
            log10_NHI_err = np.array(log10_NHI_err)
            total_NHI = NHI.sum()
            log10_total_NHI_err = np.sqrt(
                np.sum((log10_NHI_err * NHI / total_NHI) ** 2.0)
            )

            if phase == "All":
                results[idx]["mean"]["log10_NHI"] += list(np.log10(NHI))
                results[idx]["sd"]["log10_NHI"] += list(log10_NHI_err)
            results[idx]["mean"][f"{phase}_log10_NHI"] = np.log10(total_NHI)
            results[idx]["sd"][f"{phase}_log10_NHI"] = log10_total_NHI_err

        # phase fractions
        for phase in ["CNM", "LNM", "WNM"]:
            phase_fraction = results[idx]["mean"][f"{phase}_fraction"] = (
                10.0 ** results[idx]["mean"][f"{phase}_log10_NHI"]
                / 10.0 ** results[idx]["mean"]["All_log10_NHI"]
            )
            results[idx]["sd"][f"{phase}_fraction"] = (
                phase_fraction
                * np.log(10.0)
                * np.sqrt(
                    results[idx]["sd"][f"{phase}_log10_NHI"] ** 2.0
                    + results[idx]["sd"]["All_log10_NHI"] ** 2.0
                    - 2.0
                    * phase_fraction
                    * results[idx]["sd"][f"{phase}_log10_NHI"]
                    * results[idx]["sd"]["All_log10_NHI"]
                )
            )

        pbar.update()
    return results
