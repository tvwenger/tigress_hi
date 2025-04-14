import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib import cm
import numpy as np
from scipy.stats import gaussian_kde

_SOURCES = ["tigress", "caribou_hi", "gausspy"]

print("analysis_plots version 0.1")


class Histogram:
    def __init__(self, ax, results, sources, drop_sightlines=[], drop_clouds={}):
        self.ax = ax
        self.results = results
        for source in sources:
            if source not in _SOURCES:
                raise ValueError(f"Invalid source: {source}")
        self.sources = sources
        self.drop_sightlines = drop_sightlines
        self.drop_clouds = drop_clouds

    def num_clouds(self, bins=None, **kwargs):
        legend = False
        for results, source in zip(self.results, self.sources):
            if source == "caribou_hi":
                num = np.array(
                    [result["median"]["n_gauss"] for result in results.values()]
                )
                self.ax.hist(num, bins=bins, color="gray", edgecolor="k")
            elif source == "gausspy":
                num_em = np.array(
                    [result["mean"]["n_gauss_em"] for result in results.values()]
                )
                num_abs = np.array(
                    [result["mean"]["n_gauss_abs"] for result in results.values()]
                )
                self.ax.hist(
                    num_em,
                    bins=bins,
                    color="gray",
                    hatch=r"\\",
                    edgecolor="k",
                    alpha=0.5,
                    label="Emission",
                )
                self.ax.hist(
                    num_abs,
                    bins=bins,
                    color="gray",
                    hatch="//",
                    edgecolor="k",
                    alpha=0.5,
                    label="Absorption",
                )
                legend = True
            else:
                raise ValueError(f"{source} not supported")
        return self.ax, legend, None, None

    def parameter(
        self,
        param="All_log10_NHI",
        bins=None,
        xlim=None,
        ylim=None,
        sightline=False,
    ):
        for results, source in zip(self.results, self.sources):
            if source == "caribou_hi":
                if sightline:
                    data = np.array([result["median"][param] for result in results.values()])
                else:
                    data = np.concatenate([result["median"][param] for result in results.values()])
                self.ax.hist(
                    data,
                    bins=bins,
                    color="gray",
                    edgecolor="k",
                )

            elif source in ["tigress", "gausspy"]:
                if sightline:
                    data = np.array([result[param] for result in results.values()])
                else:
                    data = np.concatenate([result[param] for result in results.values()])
                self.ax.hist(
                    data,
                    bins=bins,
                    color="gray",
                    edgecolor="k",
                )

            else:
                raise ValueError(f"Invalid source: {source}")
        return self.ax, False, None, None


class Scatter:
    def __init__(self, ax, results, sources, drop_sightlines=[], drop_clouds={}):
        self.ax = ax
        self.results = results
        for source in sources:
            if source not in _SOURCES:
                raise ValueError(f"Invalid source: {source}")
        self.sources = sources
        self.drop_sightlines = drop_sightlines
        self.drop_clouds = drop_clouds

    def parameter(
        self,
        params=["All_log10_NHI"],
        color_param=None,
        num_bins=0,
        relative=None,
        vmin=None,
        vmax=None,
        **kwargs,
    ):
        data = []
        error = []
        color_data = None
        if len(params) == 1:
            params = params * len(self.results)
        elif len(params) != len(self.results):
            raise ValueError("params must have length 1 or len(results)")

        for results, source, param in zip(self.results, self.sources, params):
            if source == "caribou_hi":
                # if "_tau" in param or "_TB" in param:
                if False:
                    avg_data = np.ones(len(results)) * np.nan
                    avg_error = np.ones((len(results), 2)) * np.nan
                    for idx, result in results.items():
                        if idx in self.drop_sightlines:
                            continue
                        if "_tau" in param:
                            weights = np.array(result["median"]["tau_weights"])
                            weights = weights / weights.sum()
                            avg_data[idx] = np.average(
                                result["median"][param], weights=weights
                            )
                            avg_error_low = (
                                result["median"][param] - result["eti_16%"][param]
                            )
                            avg_error_high = (
                                result["eti_84%"][param] - result["median"][param]
                            )
                            avg_error[idx] = np.sqrt(
                                np.sum(
                                    np.vstack([avg_error_low, avg_error_high]) ** 2.0
                                    * weights**2.0
                                )
                            )
                else:
                    if isinstance(results[next(iter(results))]["median"][param], list):
                        data.append(
                            np.concatenate(
                                [
                                    result["median"][param]
                                    for idx, result in results.items()
                                    if idx not in self.drop_sightlines
                                ]
                            )
                        )
                        err_low = np.concatenate(
                            [
                                np.array(result["median"][param])
                                - np.array(result["eti_16%"][param])
                                for idx, result in results.items()
                                if idx not in self.drop_sightlines
                            ]
                        )
                        err_high = np.concatenate(
                            [
                                np.array(result["eti_84%"][param])
                                - np.array(result["median"][param])
                                for idx, result in results.items()
                                if idx not in self.drop_sightlines
                            ]
                        )
                        if color_param is not None:
                            color_data = np.concatenate(
                                [
                                    result["median"][color_param]
                                    for idx, result in results.items()
                                    if idx not in self.drop_sightlines
                                ]
                            )
                    else:
                        data.append(
                            np.array(
                                [
                                    result["median"][param]
                                    for idx, result in results.items()
                                    if idx not in self.drop_sightlines
                                ]
                            )
                        )
                        err_low = [
                            np.array(result["median"][param])
                            - np.array(result["eti_16%"][param])
                            for idx, result in results.items()
                            if idx not in self.drop_sightlines
                        ]
                        err_high = [
                            np.array(result["eti_84%"][param])
                            - np.array(result["median"][param])
                            for idx, result in results.items()
                            if idx not in self.drop_sightlines
                        ]
                        if color_param is not None:
                            color_data = np.array(
                                [
                                    result["median"][color_param]
                                    for idx, result in results.items()
                                    if idx not in self.drop_sightlines
                                ]
                            )
                    err = np.vstack([err_low, err_high])
                    bad = np.isnan(err) + (err < 0.0)
                    err[bad] = 0.0
                    error.append(err)

            elif source == "gausspy":
                if isinstance(results[next(iter(results))]["mean"][param], list):
                    data.append(
                        np.concatenate(
                            [
                                result["mean"][param]
                                for idx, result in results.items()
                                if idx not in self.drop_sightlines
                            ]
                        )
                    )
                    error.append(
                        np.concatenate(
                            [
                                result["sd"][param]
                                for idx, result in results.items()
                                if idx not in self.drop_sightlines
                            ]
                        )
                    )
                    if color_param is not None:
                        color_data = np.concatenate(
                            [
                                result["mean"][color_param]
                                for idx, result in results.items()
                                if idx not in self.drop_sightlines
                            ]
                        )
                else:
                    data.append(
                        np.array(
                            [
                                result["mean"][param]
                                for idx, result in results.items()
                                if idx not in self.drop_sightlines
                            ]
                        )
                    )
                    error.append(
                        np.array(
                            [
                                result["sd"][param]
                                for idx, result in results.items()
                                if idx not in self.drop_sightlines
                            ]
                        )
                    )
                    if color_param is not None:
                        color_data = np.concatenate(
                            [
                                result["mean"][color_param]
                                for idx, result in results.items()
                                if idx not in self.drop_sightlines
                            ]
                        )

            elif source == "tigress":
                data.append(
                    np.array(
                        [
                            result[param]
                            for idx, result in results.items()
                            if idx not in self.drop_sightlines
                        ]
                    )
                )
                error.append(None)
                if color_param is not None:
                    color_data = np.array(
                        [
                            result[color_param]
                            for idx, result in results.items()
                            if idx not in self.drop_sightlines
                        ]
                    )

            else:
                raise ValueError(f"Invalid source: {source}")

        if relative == "absolute":
            data[1] = data[1] - data[0]
            err0 = error[0] if error[0] is not None else 0.0
            err1 = error[1] if error[1] is not None else 0.0
            error[1] = np.sqrt(err0**2.0 + err1**2.0)
        elif relative == "fractional":
            data[1] = (data[1] - data[0]) / data[0]
            err0 = error[0] if error[0] is not None else 0.0
            err1 = error[1] if error[1] is not None else 0.0
            error[1] = np.sqrt(err0**2.0 + err1**2.0) / np.abs(data[0])

        if len(data[1]) == 0:
            data[1] = np.ones_like(data[0]) * np.nan
            error[1] = None

        cax = None
        if color_data is not None:
            cax = self.ax.scatter(
                data[0],
                data[1],
                marker=".",
                c=color_data,
                alpha=0.25,
                vmin=vmin,
                vmax=vmax,
            )
        else:
            self.ax.errorbar(
                data[0],
                data[1],
                xerr=error[0],
                yerr=error[1],
                marker=".",
                c="k",
                linestyle="none",
                alpha=0.25,
                errorevery=1,
            )

        # indicate outliers with arrows
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        xscale = self.ax.get_xscale()
        yscale = self.ax.get_yscale()

        # start arrows 10% away from axis edge
        low_x = ~np.isinf(data[0]) * (data[0] < xlim[0])
        high_x = ~np.isinf(data[0]) * (data[0] > xlim[1])
        low_y = ~np.isinf(data[1]) * (data[1] < ylim[0])
        high_y = ~np.isinf(data[1]) * (data[1] > ylim[1])

        if xscale == "linear":
            left_dx = -0.1 * (xlim[1] - xlim[0])
            right_dx = 0.1 * (xlim[1] - xlim[0])
        else:
            left_dx = -xlim[0] * 10.0 ** (0.1 * np.log10(xlim[1] / xlim[0])) + xlim[0]
            right_dx = xlim[1] / 10.0 ** (0.1 * np.log10(xlim[1] / xlim[0])) - xlim[1]
        if yscale == "linear":
            bottom_dy = -0.1 * (ylim[1] - ylim[0])
            top_dy = 0.1 * (ylim[1] - ylim[0])
        else:
            bottom_dy = -ylim[0] * 10.0 ** (0.1 * np.log10(ylim[1] / ylim[0])) + ylim[0]
            top_dy = ylim[1] / 10.0 ** (0.1 * np.log10(ylim[1] / ylim[0])) - ylim[1]
        left_x = np.zeros_like(data[0][low_x]) + xlim[0] - 1.1 * left_dx
        right_x = np.zeros_like(data[0][high_x]) + xlim[1] - 1.1 * right_dx
        bottom_y = np.zeros_like(data[1][low_y]) + ylim[0] - 1.1 * bottom_dy
        top_y = np.zeros_like(data[1][high_y]) + ylim[1] - 1.1 * top_dy

        xgroups = [left_x, right_x, data[0][low_y], data[0][high_y]]
        ygroups = [data[1][low_x], data[1][high_x], bottom_y, top_y]
        dxgroups = [
            np.ones_like(left_x) * left_dx,
            np.ones_like(right_x) * right_dx,
            np.zeros_like(bottom_y),
            np.zeros_like(top_y),
        ]
        dygroups = [
            np.zeros_like(left_x),
            np.zeros_like(right_x),
            np.ones_like(bottom_y) * bottom_dy,
            np.ones_like(top_y) * top_dy,
        ]
        for xgrp, ygrp, dxgrp, dygrp in zip(xgroups, ygroups, dxgroups, dygroups):
            for x, y, dx, dy in zip(xgrp, ygrp, dxgrp, dygrp):
                arrow_start = (x, y)
                arrow_end = (x + dx, y + dy)
                self.ax.annotate(
                    "",
                    arrow_end,
                    arrow_start,
                    arrowprops=dict(
                        width=1.0,
                        headwidth=3.0,
                        headlength=3.0,
                        alpha=0.1,
                        color="k",
                    ),
                )

        # add box plot
        if num_bins > 0:
            bin_start = np.nanmax([np.nanmin(data[0]), xlim[0]])
            bin_end = np.nanmin([np.nanmax(data[0]), xlim[1]])
            bin_width = (bin_end - bin_start) / num_bins
            bin_edges = np.linspace(bin_start, bin_end, num_bins + 1, endpoint=True)
            bin_centers = bin_edges[:-1] + bin_width / 2.0
            idxs = np.digitize(data[0], bin_edges)
            idxs = idxs[~np.isnan(data[0])]
            box_data = [data[1][idxs == i + 1] for i in range(len(bin_centers))]
            box_data = [box_datum[~np.isnan(box_datum)] for box_datum in box_data]
            bin_centers = [
                bin_center
                for bin_center, box_datum in zip(bin_centers, box_data)
                if len(box_datum) > 0
            ]
            box_data = [box_datum for box_datum in box_data if len(box_datum) > 0]
            parts = self.ax.violinplot(
                box_data,
                bin_centers,
                widths=0.85 * bin_width,
                showmedians=True,
                showextrema=False,
            )
            for pc in parts["bodies"]:
                pc.set_facecolor("orange")
                pc.set_edgecolor("orange")
                pc.set_alpha(0.9)
            parts["cmedians"].set_colors("red")

        return self.ax, False, cax, None


class Cumulative:
    def __init__(
        self,
        ax,
        results,
        sources,
        drop_sightlines=[],
        drop_clouds={},
    ):
        self.ax = ax
        self.results = results
        for source in sources:
            if source not in _SOURCES:
                raise ValueError(f"Invalid source: {source}")
        self.sources = sources
        self.drop_sightlines = drop_sightlines
        self.drop_clouds = drop_clouds

    def parameter(
        self,
        params=["tspin"],
        num_samples=100,
        labels=None,
        volume=False,
        **kwargs,
    ):
        if len(params) == 1:
            params = params * len(self.results)
        elif len(params) != len(self.results):
            raise ValueError("params must have length 1 or len(results)")

        tigress_linestyles = (ls for ls in ["-", "--", ":"])
        caribou_hi_linestyles = (ls for ls in ["-", "--", ":"])

        rng = np.random.RandomState(seed=1234)
        if labels is None:
            labels = self.sources
        for results, source, label, param in zip(
            self.results, self.sources, labels, params
        ):
            if source == "tigress":
                # TIGRESS resolution = 4 pc
                res = 4.0
                xdata = np.concatenate(
                    [
                        result[param]
                        for idx, result in results.items()
                        if idx not in self.drop_sightlines
                    ]
                )
                if volume:
                    xdata[np.isnan(xdata)] = -np.inf
                    ydata = np.ones_like(xdata) * res
                else:
                    NHI_param = "log10_NHI"
                    if "_tau" in param:
                        NHI_param = "log10_NHI_tau"
                    if "_TB" in param:
                        NHI_param = "log10_NHI_TB"
                    ydata = np.concatenate(
                        [
                            10.0 ** np.array(result[NHI_param])
                            for idx, result in results.items()
                            if idx not in self.drop_sightlines
                        ]
                    )

                sort_idx = np.argsort(xdata)
                cdf_x = xdata[sort_idx]
                ydata = ydata[sort_idx]
                if volume:
                    cdf_y = np.nancumsum(ydata) / np.nansum(ydata)
                else:
                    cdf_y = np.log10(np.nancumsum(ydata))
                self.ax.step(
                    cdf_x,
                    cdf_y,
                    where="post",
                    color="k",
                    linestyle=next(tigress_linestyles),
                    label=label,
                )

            elif source == "caribou_hi":
                my_param = param.replace("_tau", "").replace("_TB", "")

                absorption_weight_free = False
                for result in results.values():
                    if (
                        result["median"]["n_gauss"] > 0
                        and len(result["median"]["absorption_weight"]) > 0
                    ):
                        absorption_weight_free = True
                        break

                filling_factor_free = False
                for result in results.values():
                    if (
                        result["median"]["n_gauss"] > 0
                        and len(result["median"]["filling_factor"]) > 0
                    ):
                        filling_factor_free = True
                        break

                # posterior samples
                sample_idx = [
                    (
                        rng.randint(
                            np.array(result["median"]["tspin_samples"]).shape[1],
                            size=num_samples,
                        )
                        if result["median"]["n_gauss"] > 0
                        else None
                    )
                    for result in results.values()
                ]

                for i in range(num_samples):
                    xdata = np.concatenate(
                        [
                            np.array(
                                [
                                    np.array(result["median"][f"{my_param}_samples"])[
                                        cloud_i, sample_i[i]
                                    ]
                                    for cloud_i in range(result["median"]["n_gauss"])
                                    if cloud_i not in self.drop_clouds.get(idx, [])
                                ]
                            )
                            for (idx, result), sample_i in zip(
                                results.items(), sample_idx
                            )
                            if sample_i is not None and idx not in self.drop_sightlines
                        ]
                    )
                    if "_tau" in param:
                        ydata = np.concatenate(
                            [
                                np.array(
                                    [
                                        (
                                            np.array(
                                                result["median"][
                                                    "absorption_weight_samples"
                                                ]
                                            )[cloud_i, sample_i[i]]
                                            if absorption_weight_free
                                            else 1.0
                                        )
                                        * 10.0
                                        ** np.array(
                                            result["median"]["log10_NHI_samples"]
                                        )[cloud_i, sample_i[i]]
                                        for cloud_i in range(
                                            result["median"]["n_gauss"]
                                        )
                                        if cloud_i not in self.drop_clouds.get(idx, [])
                                    ]
                                )
                                for (idx, result), sample_i in zip(
                                    results.items(), sample_idx
                                )
                                if sample_i is not None
                                and idx not in self.drop_sightlines
                            ]
                        )
                    elif "_TB" in param:
                        ydata = np.concatenate(
                            [
                                np.array(
                                    [
                                        (
                                            np.array(
                                                result["median"][
                                                    "filling_factor_samples"
                                                ]
                                            )[cloud_i, sample_i[i]]
                                            if filling_factor_free
                                            else 1.0
                                        )
                                        * 10.0
                                        ** np.array(
                                            result["median"]["log10_NHI_samples"]
                                        )[cloud_i, sample_i[i]]
                                        for cloud_i in range(
                                            result["median"]["n_gauss"]
                                        )
                                        if cloud_i not in self.drop_clouds.get(idx, [])
                                    ]
                                )
                                for (idx, result), sample_i in zip(
                                    results.items(), sample_idx
                                )
                                if sample_i is not None
                                and idx not in self.drop_sightlines
                            ]
                        )
                    sort_idx = np.argsort(xdata)
                    cdf_x = xdata[sort_idx]
                    cdf_y = np.log10(np.nancumsum(ydata[sort_idx]))
                    self.ax.step(
                        cdf_x,
                        cdf_y,
                        where="post",
                        color="orange",
                        alpha=0.5,
                        linewidth=1.0,
                        linestyle=":",
                    )

                # posterior point estimate
                xdata = np.concatenate(
                    [
                        np.array(
                            [
                                result["median"][my_param][cloud_i]
                                for cloud_i in range(result["median"]["n_gauss"])
                                if cloud_i not in self.drop_clouds.get(idx, [])
                            ]
                        )
                        for idx, result in results.items()
                        if idx not in self.drop_sightlines
                    ]
                )
                if "_tau" in param:
                    ydata = np.concatenate(
                        [
                            np.array(
                                [
                                    (
                                        np.array(
                                            result["median"]["absorption_weight"][
                                                cloud_i
                                            ]
                                        )
                                        if absorption_weight_free
                                        else 1.0
                                    )
                                    * 10.0 ** result["median"]["log10_NHI"][cloud_i]
                                    for cloud_i in range(result["median"]["n_gauss"])
                                    if cloud_i not in self.drop_clouds.get(idx, [])
                                ]
                            )
                            for idx, result in results.items()
                            if idx not in self.drop_sightlines
                        ]
                    )
                elif "_TB" in param:
                    ydata = np.concatenate(
                        [
                            np.array(
                                [
                                    (
                                        np.array(
                                            result["median"]["filling_factor"][cloud_i]
                                        )
                                        if filling_factor_free
                                        else 1.0
                                    )
                                    * 10.0 ** result["median"]["log10_NHI"][cloud_i]
                                    for cloud_i in range(result["median"]["n_gauss"])
                                    if cloud_i not in self.drop_clouds.get(idx, [])
                                ]
                            )
                            for idx, result in results.items()
                            if idx not in self.drop_sightlines
                        ]
                    )
                sort_idx = np.argsort(xdata)
                cdf_x = xdata[sort_idx]
                cdf_y = np.log10(np.nancumsum(ydata[sort_idx]))
                self.ax.step(
                    cdf_x,
                    cdf_y,
                    where="post",
                    color="r",
                    label=label,
                    linestyle=next(caribou_hi_linestyles),
                )

            elif source == "gausspy":
                # draw samples from uncorrelated posterior distributions
                xdata_mean = np.concatenate(
                    [
                        result["mean"][param]
                        for idx, result in results.items()
                        if idx not in self.drop_sightlines
                    ]
                )
                xdata_sd = np.concatenate(
                    [
                        result["sd"][param]
                        for idx, result in results.items()
                        if idx not in self.drop_sightlines
                    ]
                )

                ydata_mean = np.concatenate(
                    [
                        10.0 ** np.array(result["mean"]["log10_NHI"])
                        for idx, result in results.items()
                        if idx not in self.drop_sightlines
                    ]
                )
                ydata_sd = np.concatenate(
                    [
                        result["sd"]["log10_NHI"]
                        for idx, result in results.items()
                        if idx not in self.drop_sightlines
                    ]
                )

                for i in range(num_samples):
                    xdata = rng.normal(loc=xdata_mean, scale=xdata_sd)
                    ydata = rng.normal(loc=ydata_mean, scale=ydata_sd)

                    sort_idx = np.argsort(xdata)
                    cdf_x = xdata[sort_idx]
                    cdf_y = np.log10(np.nancumsum(ydata[sort_idx]))
                    self.ax.step(
                        cdf_x,
                        cdf_y,
                        where="post",
                        color="green",
                        alpha=0.5,
                        linewidth=1.0,
                        linestyle=":",
                    )

                xdata = np.concatenate(
                    [
                        result["mean"][param]
                        for idx, result in results.items()
                        if idx not in self.drop_sightlines
                    ]
                )
                ydata = np.concatenate(
                    [
                        10.0 ** np.array(result["mean"]["log10_NHI"])
                        for idx, result in results.items()
                        if idx not in self.drop_sightlines
                    ]
                )
                bad = (
                    np.isnan(xdata)
                    + np.isinf(xdata)
                    + np.isnan(ydata)
                    + np.isinf(ydata)
                )
                xdata = xdata[~bad]
                ydata = ydata[~bad]
                sort_idx = np.argsort(xdata)
                cdf_x = xdata[sort_idx]
                cdf_y = np.log10(np.nancumsum(ydata[sort_idx]))
                self.ax.step(
                    cdf_x,
                    cdf_y,
                    where="post",
                    color="b",
                    label=label,
                    linestyle="--",
                )

            else:
                raise ValueError(f"Invalid source: {source}")
        return self.ax, True, None, None


class Cumulative2D:
    def __init__(
        self,
        ax,
        results,
        sources,
        drop_sightlines=[],
        drop_clouds={},
    ):
        self.ax = ax
        self.results = results
        for source in sources:
            if source not in _SOURCES:
                raise ValueError(f"Invalid source: {source}")
        self.sources = sources
        self.drop_sightlines = drop_sightlines
        self.drop_clouds = drop_clouds

    def parameter(
        self,
        params=["log10_tkin", "log10_pressure"],
        labels=None,
        volume=False,
        gridsize=20,
        vmin=None,
        vmax=None,
        xlim=(-np.inf, np.inf),
        ylim=(-np.inf, np.inf),
        contour_tigress=False,
        contour_levels=[1e21],
    ):
        if len(params) != 2:
            raise ValueError("params must have length 2")

        if labels is None:
            labels = self.sources

        conax = None

        for res_idx, (results, source, label) in enumerate(
            zip(self.results, self.sources, labels)
        ):
            if source == "tigress":
                # TIGRESS resolution = 4 pc
                res = 4.0

                xdata = np.concatenate(
                    [
                        result[params[0]]
                        for idx, result in results.items()
                        if idx not in self.drop_sightlines
                    ]
                )
                ydata = np.concatenate(
                    [
                        result[params[1]]
                        for idx, result in results.items()
                        if idx not in self.drop_sightlines
                    ]
                )

                if volume:
                    zdata = np.ones_like(xdata) * res
                else:
                    NHI_param = "log10_NHI"
                    if "_tau" in params[0] and "_tau" in params[1]:
                        NHI_param = "log10_NHI_tau"
                    if "_TB" in params[0] and "_TB" in params[1]:
                        NHI_param = "log10_NHI_TB"
                    zdata = np.concatenate(
                        [
                            10.0 ** np.array(result[NHI_param])
                            for idx, result in results.items()
                            if idx not in self.drop_sightlines
                        ]
                    )

                # drop NaNs and Infs
                bad = (
                    np.isnan(xdata)
                    + np.isinf(xdata)
                    + np.isnan(ydata)
                    + np.isinf(ydata)
                    + np.isnan(zdata)
                    + np.isinf(zdata)
                )
                xdata = xdata[~bad]
                ydata = ydata[~bad]
                zdata = zdata[~bad]

                if res_idx > 0 and contour_tigress:
                    xbins = np.linspace(xlim[0], xlim[1], gridsize)
                    ybins = np.linspace(ylim[0], ylim[1], gridsize)
                    contourZ, contourX, contourY = np.histogram2d(
                        xdata, ydata, weights=zdata, bins=(xbins, ybins)
                    )
                    xaxis = contourX[:-1] + (contourX[1] - contourX[0]) / 2.0
                    yaxis = contourY[:-1] + (contourY[1] - contourY[0]) / 2.0
                    xgrid, ygrid = np.meshgrid(xaxis, yaxis, indexing="ij")
                    if volume:
                        conax = self.ax.contour(
                            xgrid,
                            ygrid,
                            contourZ / contourZ.sum(),
                            levels=contour_levels,
                            colors="k",
                        )
                    else:
                        conax = self.ax.contour(
                            xgrid,
                            ygrid,
                            contourZ,
                            levels=contour_levels,
                            colors="k",
                        )

                else:
                    # Drop out-of-bounds
                    bad = (
                        (xdata < xlim[0])
                        + (xdata > xlim[1])
                        + (ydata < ylim[0])
                        + (ydata > ylim[1])
                    )
                    xdata = xdata[~bad]
                    ydata = ydata[~bad]
                    zdata = zdata[~bad]

                    if volume:
                        cax = self.ax.hexbin(
                            xdata,
                            ydata,
                            C=zdata / zdata.sum(),
                            bins="log",
                            gridsize=gridsize,
                            vmin=vmin,
                            vmax=vmax,
                            reduce_C_function=np.sum,
                        )
                    else:
                        cax = self.ax.hexbin(
                            xdata,
                            ydata,
                            C=zdata,
                            bins="log",
                            gridsize=gridsize,
                            vmin=vmin,
                            vmax=vmax,
                            reduce_C_function=np.sum,
                        )

            elif source == "caribou_hi":
                my_param_0 = params[0].replace("_tau", "").replace("_TB", "")
                my_param_1 = params[1].replace("_tau", "").replace("_TB", "")

                absorption_weight_free = False
                for result in results.values():
                    if (
                        result["median"]["n_gauss"] > 0
                        and len(result["median"]["absorption_weight"]) > 0
                    ):
                        absorption_weight_free = True
                        break

                filling_factor_free = False
                for result in results.values():
                    if (
                        result["median"]["n_gauss"] > 0
                        and len(result["median"]["filling_factor"]) > 0
                    ):
                        filling_factor_free = True
                        break

                # posterior point estimate
                xdata = np.concatenate(
                    [
                        np.array(
                            [
                                result["median"][my_param_0][cloud_i]
                                for cloud_i in range(result["median"]["n_gauss"])
                                if cloud_i not in self.drop_clouds.get(idx, [])
                            ]
                        )
                        for idx, result in results.items()
                        if idx not in self.drop_sightlines
                    ]
                )
                ydata = np.concatenate(
                    [
                        np.array(
                            [
                                result["median"][my_param_1][cloud_i]
                                for cloud_i in range(result["median"]["n_gauss"])
                                if cloud_i not in self.drop_clouds.get(idx, [])
                            ]
                        )
                        for idx, result in results.items()
                        if idx not in self.drop_sightlines
                    ]
                )
                if volume:
                    zdata = np.concatenate(
                        [
                            np.array(
                                [
                                    10.0 ** result["median"]["log10_depth"][cloud_i]
                                    for cloud_i in range(result["median"]["n_gauss"])
                                    if cloud_i not in self.drop_clouds.get(idx, [])
                                ]
                            )
                            for idx, result in results.items()
                            if idx not in self.drop_sightlines
                        ]
                    )
                elif "_tau" in params[0] and "_tau" in params[1]:
                    zdata = np.concatenate(
                        [
                            np.array(
                                [
                                    (
                                        np.array(
                                            result["median"]["absorption_weight"][
                                                cloud_i
                                            ]
                                        )
                                        if absorption_weight_free
                                        else 1.0
                                    )
                                    * 10.0 ** result["median"]["log10_NHI"][cloud_i]
                                    for cloud_i in range(result["median"]["n_gauss"])
                                    if cloud_i not in self.drop_clouds.get(idx, [])
                                ]
                            )
                            for idx, result in results.items()
                            if idx not in self.drop_sightlines
                        ]
                    )
                elif "_TB" in params[0] and "_TB" in params[1]:
                    zdata = np.concatenate(
                        [
                            np.array(
                                [
                                    (
                                        np.array(
                                            result["median"]["filling_factor"][cloud_i]
                                        )
                                        if filling_factor_free
                                        else 1.0
                                    )
                                    * 10.0 ** result["median"]["log10_NHI"][cloud_i]
                                    for cloud_i in range(result["median"]["n_gauss"])
                                    if cloud_i not in self.drop_clouds.get(idx, [])
                                ]
                            )
                            for idx, result in results.items()
                            if idx not in self.drop_sightlines
                        ]
                    )
                else:
                    raise ValueError("both parameters must be _tau or _TB")

                # Drop out-of-bounds
                bad = (
                    (xdata < xlim[0])
                    + (xdata > xlim[1])
                    + (ydata < ylim[0])
                    + (ydata > ylim[1])
                )
                xdata = xdata[~bad]
                ydata = ydata[~bad]
                zdata = zdata[~bad]

                if volume:
                    cax = self.ax.hexbin(
                        xdata,
                        ydata,
                        C=zdata / zdata.sum(),
                        bins="log",
                        gridsize=gridsize,
                        vmin=vmin,
                        vmax=vmax,
                        reduce_C_function=np.sum,
                    )
                else:
                    cax = self.ax.hexbin(
                        xdata,
                        ydata,
                        C=zdata,
                        bins="log",
                        gridsize=gridsize,
                        vmin=vmin,
                        vmax=vmax,
                        reduce_C_function=np.sum,
                    )

        return self.ax, False, cax, conax


def plot_one(
    plottype,
    plotdata,
    results,
    sources,
    xlabel=None,
    ylabel=None,
    title=None,
    xscale="linear",
    yscale="linear",
    xlim=None,
    ylim=None,
    legend_loc="best",
    **kwargs,
):
    fig, ax = plt.subplots(layout="constrained")
    plot = plottype(ax, results, sources)
    func = getattr(plot, plotdata)
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    ax, legend = func(xlim=xlim, ylim=ylim, **kwargs)
    if legend:
        ax.legend(loc=legend_loc, fontsize=12)

    return fig


def plot_grid(
    plottype,
    plotdata,
    results,
    sources,
    drop_sightlines=None,
    drop_clouds=None,
    xlabel=None,
    ylabel=None,
    title=None,
    xscale="linear",
    yscale="linear",
    xlim=None,
    ylim=None,
    equal_line=True,
    label_loc="upper right",
    legend_loc="best",
    cbar_label="",
    vlines=[],
    vline_labels=[],
    vline_label_offset=0,
    **kwargs,
):
    legend_placed = False
    cbar_placed = False
    vline_labels_placed = False

    if all(source == "tigress" for source in sources):
        fig = plt.figure(constrained_layout=True, figsize=(12, 3))
        axes = fig.subplots(nrows=1, ncols=3, sharex=True, sharey=True)
        for fwhm_idx, fwhm in enumerate(results[0].keys()):
            plot = plottype(
                axes[fwhm_idx], [result[fwhm] for result in results], sources
            )
            func = getattr(plot, plotdata)

            if xlim is not None:
                axes[fwhm_idx].set_xlim(xlim)
            if ylim is not None:
                axes[fwhm_idx].set_ylim(ylim)

            axes[fwhm_idx].set_xscale(xscale)
            axes[fwhm_idx].set_yscale(yscale)

            axes[fwhm_idx], legend, cax, conax = func(xlim=xlim, ylim=ylim, **kwargs)
            if legend and not legend_placed:
                axes[fwhm_idx].legend(loc=legend_loc, fontsize=12)
                legend_placed = True

            if cax is not None and not cbar_placed:
                cbar = fig.colorbar(
                    cax,
                    ax=axes,
                    location="right",
                    aspect=5,
                    pad=0.015,
                    extend="both",
                )
                if conax is not None:
                    cbar.add_lines(conax)
                cbar.set_label(cbar_label)
                cbar_placed = True

            if fwhm == 1:
                label = "pencil-beam"
            else:
                label = r"$\theta$" + f" = {fwhm} pix"
            label_locx = 0.25
            label_locy = 0.95
            if label_loc == "upper right":
                label_locx = 0.75
            if label_loc == "lower left":
                label_locy = 0.15
            if label_loc == "lower right":
                label_locx = 0.75
                label_locy = 0.15
            axes[fwhm_idx].text(
                label_locx,
                label_locy,
                label,
                ha="center",
                va="top",
                transform=axes[fwhm_idx].transAxes,
                bbox=dict(
                    facecolor="white", edgecolor="black", alpha=0.5, boxstyle="round"
                ),
            )

            if equal_line:
                start = np.min([xlim[0], ylim[0]])
                end = np.max([xlim[1], ylim[1]])
                axes[fwhm_idx].plot([start, end], [start, end], "k-")

    else:
        fig, axes = plt.subplots(
            2, 3, sharex=True, sharey=True, figsize=(12, 6), layout="constrained"
        )
        for datatype_idx, datatype in enumerate(["mismatched", "matched"]):
            for fwhm_idx, fwhm in enumerate([1, 3, 10]):
                my_results = []
                for result, source in zip(results, sources):
                    if source == "tigress":
                        my_results.append(result[fwhm])
                    elif source in ["caribou_hi", "gausspy"]:
                        my_results.append(result[datatype][fwhm])

                plot = plottype(
                    axes[datatype_idx, fwhm_idx],
                    my_results,
                    sources,
                    drop_sightlines=(
                        drop_sightlines[datatype][fwhm]
                        if drop_sightlines is not None
                        else []
                    ),
                    drop_clouds=(
                        drop_clouds[datatype][fwhm] if drop_clouds is not None else {}
                    ),
                )
                func = getattr(plot, plotdata)

                if xlim is not None:
                    axes[datatype_idx, fwhm_idx].set_xlim(xlim)
                if ylim is not None:
                    axes[datatype_idx, fwhm_idx].set_ylim(ylim)

                axes[datatype_idx, fwhm_idx].set_xscale(xscale)
                axes[datatype_idx, fwhm_idx].set_yscale(yscale)

                axes[datatype_idx, fwhm_idx], legend, cax, conax = func(
                    xlim=xlim, ylim=ylim, **kwargs
                )
                if legend and not legend_placed:
                    axes[datatype_idx, fwhm_idx].legend(loc=legend_loc, fontsize=12)
                    legend_placed = True

                if cax is not None and not cbar_placed:
                    cbar = fig.colorbar(
                        cax,
                        ax=axes,
                        location="right",
                        aspect=5,
                        pad=0.015,
                        extend="both",
                    )
                    if conax is not None:
                        cbar.add_lines(conax)
                    cbar.set_label(cbar_label)
                    cbar_placed = True

                if datatype == "mismatched" and fwhm == 1:
                    label = "True\npencil-beam"
                elif datatype == "matched" and fwhm == 1:
                    label = "Annulus\n" + "pencil-beam"
                elif datatype == "mismatched":
                    label = "True\n" + r"$\theta$" + f" = {fwhm} pix"
                else:
                    label = "Annulus\n" + r"$\theta$" + f" = {fwhm} pix"
                label_locx = 0.25
                label_locy = 0.95
                if label_loc == "upper right":
                    label_locx = 0.75
                if label_loc == "lower left":
                    label_locy = 0.35
                if label_loc == "lower right":
                    label_locx = 0.75
                    label_locy = 0.35
                axes[datatype_idx, fwhm_idx].text(
                    label_locx,
                    label_locy,
                    label,
                    ha="center",
                    va="top",
                    transform=axes[datatype_idx, fwhm_idx].transAxes,
                    bbox=dict(
                        facecolor="white",
                        edgecolor="black",
                        alpha=0.5,
                        boxstyle="round",
                    ),
                )

                if equal_line:
                    start = np.min([xlim[0], ylim[0]])
                    end = np.max([xlim[1], ylim[1]])
                    axes[datatype_idx, fwhm_idx].plot([start, end], [start, end], "k-")

                for vline, vline_label in zip(vlines, vline_labels):
                    axes[datatype_idx, fwhm_idx].axvline(vline, color="k", alpha=0.75)
                    if not vline_labels_placed:
                        trans = transforms.blended_transform_factory(
                            axes[datatype_idx, fwhm_idx].transData,
                            axes[datatype_idx, fwhm_idx].transAxes,
                        )
                        if xscale == "log":
                            axes[datatype_idx, fwhm_idx].text(
                                vline * vline_label_offset,
                                0.5,
                                vline_label,
                                transform=trans,
                            )
                        else:
                            axes[datatype_idx, fwhm_idx].text(
                                vline + vline_label_offset,
                                0.5,
                                vline_label,
                                transform=trans,
                            )
                if not vline_labels_placed:
                    vline_labels_placed = True

    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    fig.suptitle(title)

    return fig


def plot_spectra(data, results, source, idx):
    if source == "caribou_hi":
        fig, axes = plt.subplots(
            4, 3, sharex=True, figsize=(12, 6), layout="constrained"
        )
        for type_idx, key in enumerate(data.keys()):
            for fwhm_idx, fwhm in enumerate(data[key].keys()):
                axes[2 * type_idx, fwhm_idx].plot(
                    data[key][fwhm]["x_values_em"][idx],
                    data[key][fwhm]["data_list_em"][idx],
                    "k-",
                )
                axes[2 * type_idx + 1, fwhm_idx].plot(
                    data[key][fwhm]["x_values"][idx],
                    data[key][fwhm]["data_list"][idx],
                    "k-",
                )

                # evaluate cloud-based contributions to data from posterior samples
                if results[key][fwhm][idx]["median"]["n_gauss"] > 0:
                    colors = cm.rainbow(
                        np.linspace(
                            0,
                            1,
                            results[key][fwhm][idx]["median"]["n_gauss"],
                        )
                    )
                    for i, color in enumerate(colors):
                        axes[2 * type_idx, fwhm_idx].plot(
                            data[key][fwhm]["x_values_em"][idx],
                            results[key][fwhm][idx]["predicted_emission"][i],
                            "-",
                            color=color,
                            alpha=0.1,
                        )
                        axes[2 * type_idx + 1, fwhm_idx].plot(
                            data[key][fwhm]["x_values"][idx],
                            results[key][fwhm][idx]["predicted_absorption"][i],
                            "-",
                            color=color,
                            alpha=0.1,
                        )
                label = (
                    r"$\log_{10} N_{\rm HI, Em}$ = "
                    + f"{results[key][fwhm][idx]["median"]["All_log10_NHI_TB"]:.2f}"
                    + "\n"
                )
                label += (
                    r"$\log_{10} N_{\rm CNM, Em}$ = "
                    + f"{results[key][fwhm][idx]["median"]["CNM_log10_NHI_TB"]:.2f}"
                    + "\n"
                )
                label += (
                    r"$\log_{10} N_{\rm LNM, Em}$ = "
                    + f"{results[key][fwhm][idx]["median"]["LNM_log10_NHI_TB"]:.2f}"
                    + "\n"
                )
                label += (
                    r"$\log_{10} N_{\rm WNM, Em}$ = "
                    + f"{results[key][fwhm][idx]["median"]["WNM_log10_NHI_TB"]:.2f}"
                )
                axes[2 * type_idx, fwhm_idx].text(
                    0.02,
                    0.98,
                    label,
                    ha="left",
                    va="top",
                    transform=axes[2 * type_idx, fwhm_idx].transAxes,
                    fontsize=8,
                )
                label = ""
                for i in range(len(results[key][fwhm][idx]["median"]["log10_NHI"])):
                    label += f"[{i}] "
                    label += r"$\log_{10} N_{\rm HI}$ = "
                    label += f"{results[key][fwhm][idx]["median"]["log10_NHI"][i]:.2f} "
                    label += r"$\log_{10} T_{k}$ = "
                    label += (
                        f"{results[key][fwhm][idx]["median"]["log10_tkin"][i]:.1f} "
                    )
                    label += r"$T_{s}$ = "
                    label += f"{results[key][fwhm][idx]["median"]["tspin"][i]:.1f} "
                    if i != len(results[key][fwhm][idx]["median"]["log10_NHI"]) - 1:
                        label += "\n"
                axes[2 * type_idx + 1, fwhm_idx].text(
                    0.02,
                    0.98,
                    label,
                    ha="left",
                    va="top",
                    transform=axes[2 * type_idx + 1, fwhm_idx].transAxes,
                    fontsize=8,
                )
                axes[2 * type_idx, fwhm_idx].set_ylabel(r"$T_B$ (K)")
                axes[2 * type_idx + 1, fwhm_idx].set_ylabel(r"1 - exp(-$\tau$)")
        fig.suptitle(f"{idx}")
        fig.supxlabel(r"$V_{\rm LSR}$ (km s$^{-1}$)")
    return fig
