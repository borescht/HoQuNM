"""Provide optimisation routines and defined objective functions."""

from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from hoqunm.data_tools.base import OUTPUT_DIR_OPTIMISATION
from hoqunm.simulation.evaluators import EvaluationResults


class Optimator:
    """Optimator holding objective functions to assess EvaluationResults.

    :param results: Set of EvaluationResults to consider.
    :param max_capacities: Upper capacity bounds per ward.
    :param min_capacities: Lower capacity bounds per ward.
    """
    def __init__(self,
                 results: List[EvaluationResults],
                 max_capacities: Optional[np.ndarray] = None,
                 min_capacities: Optional[np.ndarray] = None):
        self._results = results.copy()
        self.results = results.copy()
        if max_capacities is not None:
            self.results = [
                r for r in self.results
                if np.all(r.hospital_specs.capacities < max_capacities)
            ]
        if min_capacities is not None:
            self.results = [
                r for r in self.results
                if np.all(r.hospital_specs.capacities > min_capacities)
            ]

    def plain_rejection(self) -> EvaluationResults:
        """Result with best rejection rate.

        :return: Result with best rejection rate.
        """
        return min(self.results, key=lambda r: np.max(r.rejection))

    def plain_utilisation(self) -> EvaluationResults:
        """Result with best utilisation.

        :return: Result with best utilisation.
        """
        return max(self.results, key=lambda r: np.min(r.utilisation()))

    def utilisation_restricted(
        self, utilisation_constraints: np.ndarray = np.array(0.8)):
        """Result with best rejection rate under utilisation constraint.

        :param utilisation_constraints: Minimal utilisation per ward.

        :return: Result with best utilisation.
        """
        values = [
            np.max(r.rejection) if np.all(
                r.utilisation() > utilisation_constraints) else float("NaN")
            for r in self.results
        ]
        op_idx = np.nanargmin(values)
        return op_idx, values

    def profit_rejection(self,
                         bed_costs: np.ndarray = np.array(0.8),
                         rejection_costs: np.ndarray = np.array(0.25)):
        """Result with best profit + rejection costs.

        :param bed_costs: Costs for one bed.
        :param rejection_costs: Imaginary costs for rejecting a patient.

        :return: Result with best profit + rejection costs under given cost structure.
        """
        values = [
            np.sum(
                r.profit_rejection(bed_costs=bed_costs,
                                   rejection_costs=rejection_costs))
            for r in self.results
        ]
        op_idx = np.nanargmax(values)
        return op_idx, values

    def plot_results(self,
                     op_idx: int,
                     values: List[float],
                     op_idx_rel: List[int],
                     color_map: str = "YlGn",
                     angle: int = 15,
                     rotation: int = 300,
                     savepath: Optional[Path] = None,
                     filename: Optional[str] = None) -> None:
        """Plot the result for a given target function. Use only for
        3-dimensional wards!

        :param op_idx: The index to the computed optimum.
        :param values: Values for every instance in self.results.
        :param op_idx_rel: List of indices of relaxed optima.
        :param color_map: Color map to use for plotting.
        :param angle: Angle to use for 3d mapping.
        :param rotation: Rotation to use for 3d mapping.
        :param savepath: Path to save file to.
        :param filename: Name of the pdf file which contains the produced plot.
        """
        if savepath is None:
            savepath = OUTPUT_DIR_OPTIMISATION

        xyz = [
            np.array([
                r.hospital_specs.capacities for i, r in enumerate(self.results)
                if not np.isnan(values[i])
            ]).T,
            np.array([
                r.hospital_specs.capacities for i, r in enumerate(self.results)
                if not np.isnan(values[i]) and i not in op_idx_rel
                and i != op_idx
            ]).T
        ]

        val = [
            np.array([v for i, v in enumerate(values) if not np.isnan(v)]),
            np.array([
                v for i, v in enumerate(values)
                if not np.isnan(v) and i not in op_idx_rel and i != op_idx
            ])
        ]

        # sort according to last ward (z-axis)
        for i, xyzv in enumerate(zip(xyz, val)):
            if xyzv[0].shape[0] > 0:
                index = xyzv[0][2].argsort()
                xyz[i] = xyzv[0][:, index]
                val[i] = xyzv[1][index]

        _, cmap, min_value, max_value = _get_color_specs(color_map=color_map,
                                                         values=val[0])

        step = 5

        # pylint: disable=unsubscriptable-object
        count = (step + xyz[0][2][-1] - xyz[0][2][0]) // step

        fig = plt.figure(figsize=(16, 10 * count))

        n = 1
        for i in range(xyz[0][2][0], 1 + xyz[0][2][-1], step):
            ax = fig.add_subplot(count, 1, n, projection='3d')

            _plot_normal(ax=ax,
                         i=i,
                         step=step,
                         xyz=xyz[1],
                         values=val[1],
                         color_map=color_map)

            _plot_optima(ax=ax,
                         i=i,
                         step=step,
                         xyz=xyz[0],
                         results=self.results,
                         op_idx=op_idx,
                         op_idx_rel=op_idx_rel)

            _set_labels(ax=ax, results=self.results)

            _set_ticks(ax=ax, xyz=xyz[0], i=i, step=step)

            ax.grid()

            ax.view_init(angle, rotation)

            fig.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(
                min_value, max_value),
                                           cmap=cmap),
                         ax=ax,
                         shrink=0.7)

            n += 1

        fig.tight_layout()

        plt.savefig(savepath.joinpath(f"{filename}.pdf"))
        plt.close()


def _plot_normal(ax: plt.Axes, i: int, step: int, xyz: np.ndarray,
                 values: List[float], color_map: str) -> None:
    """Plot all combinations which get colored according to heatmap."""
    if xyz.shape[0] > 0:
        colors, _, _, _ = _get_color_specs(color_map=color_map, values=values)
        r1, r2 = _z_bounds(xyz=xyz, i=i, step=step)
        ax.scatter(xyz[0][r1:r2],
                   xyz[1][r1:r2],
                   xyz[2][r1:r2],
                   color=colors[r1:r2],
                   marker=".",
                   zorder=0)


def _plot_optima(ax: plt.Axes, i: int, step: int, xyz: np.ndarray,
                 results: List[EvaluationResults], op_idx: int,
                 op_idx_rel: List[int]):
    """Plot optimum and relaxed optima."""
    legend = False

    r1, r2 = _z_bounds(xyz=xyz, i=i, step=step)

    if (xyz[2][r1] <= results[op_idx].hospital_specs.capacities[2] and
        (r2 is None
         or results[op_idx].hospital_specs.capacities[2] < xyz[2][r2])):
        ax.scatter(results[op_idx].hospital_specs.capacities[0:1],
                   results[op_idx].hospital_specs.capacities[1:2],
                   results[op_idx].hospital_specs.capacities[2:],
                   color="r",
                   marker="*",
                   label="Optimum",
                   zorder=1)
        legend = True

    relaxed_legend = False
    for _, idx in enumerate(op_idx_rel):
        if (xyz[2][r1] <= results[idx].hospital_specs.capacities[2] and
            (r2 is None
             or results[idx].hospital_specs.capacities[2] < xyz[2][r2])):
            ax.scatter(results[idx].hospital_specs.capacities[0:1],
                       results[idx].hospital_specs.capacities[1:2],
                       results[idx].hospital_specs.capacities[2:],
                       color="darkred",
                       marker="*",
                       label="Relaxed optimum" if not relaxed_legend else None,
                       zorder=1)
            relaxed_legend = True

    if legend or relaxed_legend:
        ax.legend()


def _get_color_specs(
    color_map: str, values: List[float]
) -> Tuple[List[Tuple[float, float, float, float]], matplotlib.colors.Colormap,
           float, float]:
    """Compute color specifications."""
    min_value = min(values)
    max_value = max(values)
    cmap = cm.get_cmap(color_map)
    colors = [cmap((v - min_value) / (max_value - min_value)) for v in values]
    return colors, cmap, min_value, max_value


def _z_bounds(xyz: np.ndarray, i: int, step: int) -> Tuple[int, int]:
    r1 = np.where(xyz[2] == i)[0][0]
    r2 = np.where(
        xyz[2] == i +
        step)[0][0] if len(np.where(xyz[2] == i + step)[0]) != 0 else None
    return r1, r2


def _set_ticks(ax: plt.Axes, xyz: np.ndarray, i: int, step: int) -> None:
    """Set axis ticks."""
    r1, r2 = _z_bounds(xyz=xyz, i=i, step=step)
    r2_ = r2 - 1 if r2 is not None else -1
    ax.set_xticks(
        np.arange(xyz[0].min() - xyz[0].min() % 2, xyz[0].max() + 1, 2))
    ax.set_yticks(
        np.arange(xyz[1].min() - xyz[1].min() % 2, xyz[1].max() + 1, 2))
    ax.set_zticks(np.arange(xyz[2][r1], xyz[2][r2_] + 1, 1))


def _set_labels(ax: plt.Axes, results: List[EvaluationResults]) -> None:
    ax.set_xlabel(f"Capacity for {results[0].hospital_specs.ward_map[0]}")
    ax.set_ylabel(f"Capacity for {results[0].hospital_specs.ward_map[1]}")
    ax.set_zlabel(f"Capacity for {results[0].hospital_specs.ward_map[2]}")
