"""Implementation of main evaluators used to evaluate hospital with given
HospitalSpecs.

For the general purpose one should use DES implmeneted in
SimulationEvalautor. For more restricted use-cases, AnalyticEvaluator
and MarkovEvaluator can be used too.
"""

import datetime
import itertools
import json
import logging
import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.linalg
import scipy.sparse.linalg
import scipy.stats
import simpy as sm
from scipy.optimize import root
from scipy.special import factorial

from hoqunm.simulation.hospital import (Hospital, HospitalSpecs,
                                        ReporterDistributions)
from hoqunm.utils.distributions import entropy_max
from hoqunm.utils.utils import LOGGING_DIR, get_logger

MAX_PROCESSORS = mp.cpu_count()


class EvaluationResults:
    """Hold evaluation results and corresponding methods (especially for
    plotting).

    :param hospital_specs: Hospital_specs for which the results are.
    :param config: Configuration of created distributions (nalytical/simulation evaluator,...).
    :param distributions: Computed distributions.
    """
    name = "Results"

    def __init__(self,
                 hospital_specs: HospitalSpecs,
                 config: Optional[Dict[str, str]] = None,
                 distributions: Optional[ReporterDistributions] = None):
        self.hospital_specs = hospital_specs

        if config is None:
            self.config = dict()
        else:
            self.config = config

        if distributions is None:
            self.distributions = ReporterDistributions(
                hospital_specs.I, hospital_specs.U, hospital_specs.capacities)
        else:
            self.distributions = distributions

        self.full_occupancy = self.distributions.full_occupancy
        self.mean_occupancy = self.distributions.mean_occupancy
        self.utilisation = self.distributions.utilisation
        self.profit = self.distributions.profit
        self.profit_bed = self.distributions.profit_bed

    def rejection_costs(
            self, costs: np.ndarray = np.array(0.25)) -> np.ndarray:
        """Compute costs for patient rejection.

        :param costs: Imaginary costs for rejecting a patient.

        :return: The costs for each ward.
        """
        return self.hospital_specs.arrival_rate[:, :, 0].sum(
            axis=1) * self.rejection * costs

    def profit_rejection(
        self,
        bed_costs: np.ndarray = np.array(0.8),
        rejection_costs: np.ndarray = np.array(0.25)
    ) -> np.ndarray:
        """Compute costs for patient rejection.

        :param bed_costs: Costs for one bed.
        :param rejection_costs: Imaginary costs for rejecting a patient.

        :return: The costs for each ward.
        """
        profit = self.profit(costs=bed_costs)
        profit_rejection = profit - self.rejection_costs(costs=rejection_costs)

        return profit_rejection

    @property
    def occupancy(self) -> np.ndarray:
        """return distributions occupancy."""
        return self.distributions.occupancy

    @property
    def rejection(self) -> np.ndarray:
        """Return distributions rejection."""
        return self.distributions.rejection_out

    def plot_occupancy(self, figsize=(12, 6), ax=None) -> None:
        """Plot parts of the distributions for now only occupancy.

        :param figsize: The size of the figure, if ax=None.
        :param ax: The ax on which to plot on.
        """

        occupancy = self.occupancy.copy()
        occupancy[occupancy == 0] = np.nan

        if not ax:
            fig = plt.figure(figsize=figsize)

            occupancy_plot = fig.add_subplot(111)
        else:
            occupancy_plot = ax
        occupancy_plot.set_title("Occupancy distributions")
        occupancy_plot.set_ylabel("Probability")
        occupancy_plot.set_xlabel("# Patients")
        occupancy_plot.plot(np.arange(occupancy.shape[1]),
                            occupancy.transpose(),
                            "*",
                            markersize=6)
        occupancy_plot.legend(list(self.hospital_specs.ward_map_inv))
        occupancy_plot.set_xticks(np.arange(occupancy.shape[1]))
        occupancy_plot.set_yticks(np.arange(0, 1.1, 0.1))
        occupancy_plot.grid(True, axis="y")

    def plot_against(self,
                     others: List["EvaluationResults"],
                     top=False) -> None:
        """Plot the distributions against the distributions of other evaluators
        for now only occupancy.

        :param others: The other evaluators against self is plotted.
        :param top: Plot self at last so it appears on top.
        """

        color_cycle = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b", "#e377c2",
            "#7f7f7f", "#bcbd22", "#17becf"
        ]

        wards = self.hospital_specs.capacities.shape[0]

        if top:
            evaluators = others + [self]
            colors = [color_cycle[i % 6] for i, _ in enumerate(others)] + ["r"]
        else:
            evaluators = [self] + others
            colors = ["r"] + [color_cycle[i % 6] for i, _ in enumerate(others)]

        fig = plt.figure(figsize=(12, 6 * wards))
        for i in range(self.hospital_specs.I):
            occupancy_plot = fig.add_subplot(wards, 1, i + 1)
            occupancy_plot.set_title(
                "Occupancy distributions for ward {}".format(
                    self.hospital_specs.ward_map.get(i, i)))
            occupancy_plot.set_ylabel("Probability")
            occupancy_plot.set_xlabel("# Patients")
            for evaluator, color in zip(evaluators, colors):
                capacity = evaluator.hospital_specs.capacities[i]
                occupancy = evaluator.occupancy.copy()
                occupancy[occupancy == 0] = np.nan
                occupancy_plot.plot(np.arange(capacity + 1),
                                    occupancy[i, :capacity + 1],
                                    "*",
                                    color=color,
                                    markersize=6)

            if len(evaluators) < 20:
                occupancy_plot.legend(
                    [evaluator.name for evaluator in evaluators])
            max_capacity = max([
                evaluator.hospital_specs.capacities[i]
                for evaluator in evaluators
            ])
            min_occupancy = np.min([
                np.where(evaluator.occupancy[i] != 0)[0][0]
                if len(np.where(evaluator.occupancy[i] != 0)[0]) > 0 else
                max_capacity for evaluator in evaluators
            ])

            if max_capacity <= 50:
                occupancy_plot.set_xticks(
                    np.arange(min_occupancy, max_capacity + 1, 1))
            else:
                occupancy_plot.set_xticks(
                    np.arange(min_occupancy, max_capacity + 1, 5))
            occupancy_plot.set_yticks(
                np.arange(
                    0,
                    max([
                        np.nanmax(evaluator.occupancy)
                        for evaluator in evaluators
                    ]) + 0.1, 0.1))
            occupancy_plot.grid(True, axis="y")
        fig.tight_layout()

        return fig

    @staticmethod
    def plot_many(evaluation_results: List["EvaluationResults"],
                  colors: List[Union[Tuple[float, float, float, float], str]],
                  markers: List[str],
                  labels: Sequence[Optional[str]],
                  shifts: Optional[List[float]] = None) -> plt.Figure:
        """Plot multipe EvaluationResults onto one plot. Plot only occupancy.

        :param evaluation_results: Results to plot.
        :param colors: Color to use per result.
        :param markers: Marker to use per result.
        :param labels: Label to use per results.
        :param shifts: Shift factor to use per result.
        """
        wards = evaluation_results[0].hospital_specs.capacities.shape[0]
        if shifts is None:
            shifts = [0] * len(evaluation_results)
        fig = plt.figure(figsize=(12, 6 * wards))
        for i in range(evaluation_results[0].hospital_specs.I):
            occupancy_plot = fig.add_subplot(wards, 1, i + 1)
            occupancy_plot.set_title(
                "Occupancy distributions for ward {}".format(
                    evaluation_results[0].hospital_specs.ward_map.get(i, i)))
            occupancy_plot.set_ylabel("Probability")
            occupancy_plot.set_xlabel("# Patients")
            for j, evaluation_result in enumerate(evaluation_results):
                capacity = evaluation_result.hospital_specs.capacities[i]
                occupancy = evaluation_result.occupancy.copy()
                occupancy[occupancy == 0] = np.nan
                occupancy_plot.plot(np.arange(capacity + 1) + shifts[j],
                                    occupancy[i, :capacity + 1],
                                    linestyle="",
                                    color=colors[j],
                                    marker=markers[j],
                                    label=labels[j],
                                    markersize=6)

            if any([label for label in labels if labels is not None]):
                occupancy_plot.legend()
            max_capacity = max([
                evaluator.hospital_specs.capacities[i]
                for evaluator in evaluation_results
            ])
            min_occupancy = np.min([
                np.where(evaluation_result.occupancy[i] != 0)[0][0]
                if len(np.where(evaluation_result.occupancy[i] != 0)[0]) > 0
                else max_capacity for evaluation_result in evaluation_results
            ])

            if max_capacity <= 50:
                occupancy_plot.set_xticks(
                    np.arange(min_occupancy, max_capacity + 1, 1))
            else:
                min_occupancy += -min_occupancy % 5
                occupancy_plot.set_xticks(
                    np.arange(min_occupancy, max_capacity + 1, 5))
            occupancy_plot.set_yticks(
                np.arange(
                    0,
                    max([
                        np.nanmax(evaluator.occupancy)
                        for evaluator in evaluation_results
                    ]) + 0.1, 0.1))
            occupancy_plot.grid(True, axis="y")
        fig.tight_layout()
        return fig

    def save_dict(self) -> Dict[str, Any]:
        """Create argument value mapping for instance saving.

        :return: Argument value mapping.
        """
        arguments = {
            "hospital_specs": self.hospital_specs.save_dict(),
            "distributions": self.distributions.save_dict()
        }
        return arguments

    def save(self, filepath: Path = Path()) -> None:
        """Save to json via save_dict.

        :param filepath: Path to save to.
        """

        arguments_dict = self.save_dict()
        if filepath.suffix != ".json":
            filepath = filepath.parent.joinpath(filepath.name + ".json")
        with open(filepath, "w") as f:
            json.dump(arguments_dict, f)

    @staticmethod
    def load_dict(arguments: Dict[str, Any]) -> "EvaluationResults":
        """Create instance from given dict.

        :param arguments: Argument value mapping.

        :return: Class instance.
        """
        hospital_specs = HospitalSpecs.load_dict(arguments["hospital_specs"])
        distributions = ReporterDistributions.load_dict(
            arguments["distributions"])
        return EvaluationResults(hospital_specs=hospital_specs,
                                 distributions=distributions)

    @staticmethod
    def load(filepath: Path = Path()) -> "EvaluationResults":
        """Load from json via load_dict.

        :param filepath: Filepath to load from.

        :return: Loaded instance of self.
        """
        with open(filepath, "r") as f:
            arguments = json.load(f)
        return EvaluationResults.load_dict(arguments)

    def copy(self) -> "EvaluationResults":
        """Copy instance.

        :return: Copied instance.
        """
        out = EvaluationResults(hospital_specs=self.hospital_specs.copy(),
                                config=self.config,
                                distributions=self.distributions.copy())
        out.name = self.name
        return out


class Evaluator(EvaluationResults):
    """A generic evaluator class.

    :params hospital_specs: The hospital specifications.
    :param logger: Logger to use for information logging.
    """

    name = "Evaluator"

    def __init__(self,
                 hospital_specs: HospitalSpecs,
                 logger: Optional[logging.Logger] = None):
        super().__init__(hospital_specs=hospital_specs)
        self.logger = logger if logger is not None else get_logger(
            f"{__file__} - Evaluator",
            file_path=LOGGING_DIR.joinpath(
                f"{Path(__file__).resolve().stem}.log"))

    def reset(self) -> None:
        """Reset the evaluator to its initial state."""
        self.distributions = ReporterDistributions(
            self.hospital_specs.I, self.hospital_specs.U,
            self.hospital_specs.capacities)


class SimulationEvaluatorSpecs:
    """A class which holds all information corresponding to a simulation
    evaluator.

    :param  init_mode: The init_mode for the hospital. 1 is random, 0 is all on zero.
    :param processors: How many processors to use. if processors=1,
    multiprocessing will not be used.
    :pram eval_time: The basis eval_time for one hospital.
    :param max_eval_time: The time one hospital is run for the longest.
    :param run_size: The basis number of runs which are simulated together.
    For multiprocessing, should be the same as processors.
    :param max_runs: The maximal number of runs performed.
    :param error_bound_single: The error_bound for a single run to terminate.
    :param error_bound: The error bound of all runs to terminate.
    :param loss: The function, with which the loss will be computed.
    """
    def __init__(self,
                 init_mode: int = 1,
                 processors: int = MAX_PROCESSORS,
                 eval_time: int = 500,
                 max_eval_time: int = 2**4 * 500,
                 run_size: int = MAX_PROCESSORS,
                 max_runs: int = 2**2 * MAX_PROCESSORS,
                 error_bound_single: float = 1e-4,
                 error_bound: float = 1e-4,
                 loss: Callable = entropy_max):
        self.init_mode = init_mode
        self.processors = processors
        self.eval_time = eval_time
        self.max_eval_time = max_eval_time
        self.run_size = run_size
        self.max_runs = max_runs
        self.error_bound_single = error_bound_single
        self.error_bound = error_bound
        self.loss = loss


class SimulationEvaluator(Evaluator):
    """The simulation evaluator which evaluates the metrics of the hospital via
    simulation.

    :param hospital_specs: The hospital specifications.
    :param simulation_evaluator_specs: The simulation evaluator specifications.
    :param logger: Logger to use for information logging.
    """

    name = "Simulation Evaluator"

    def __init__(self,
                 hospital_specs: HospitalSpecs,
                 simulation_evaluator_specs:
                 SimulationEvaluatorSpecs = SimulationEvaluatorSpecs(),
                 logger: Optional[logging.Logger] = None):

        logger = logger if logger is not None else get_logger(
            f"{__file__} - Simulation Evaluator",
            file_path=LOGGING_DIR.joinpath(
                f"{__file__} - Simulation Evaluator.log"))
        super().__init__(hospital_specs, logger)
        self.specs = simulation_evaluator_specs
        self.runs = 0
        self.error = float("inf")

    def plot_rejection(self,
                       figsize: Tuple[int, int] = (12, 6),
                       ax: Optional[plt.Axes] = None) -> None:
        """Plot rejection and waiting_rejection of the distributions.

        :param figsize: Figure size.
        :param ax: Axes object to plot to.
        """

        rejection = self.distributions.rejection_out.copy()
        rejection[rejection == 0] = np.nan

        if not ax:
            fig = plt.figure(figsize=figsize)

            rejection_plot = fig.add_subplot(111)
        else:
            rejection_plot = ax

        rejection_plot.set_ylabel("Probabilities")
        rejection_plot.plot(
            np.arange(0, self.hospital_specs.capacities.shape[0]), rejection,
            "*")
        rejection_plot.legend(["Rejection after waiting", "Overall rejection"])
        rejection_plot.set_xticklabels([
            self.hospital_specs.ward_map[st]
            for st, _ in enumerate(self.hospital_specs.capacities)
        ])
        rejection_plot.set_xticks(
            np.arange(0, self.hospital_specs.capacities.shape[0]))
        rejection_plot.set_yticks(np.arange(0, 1.1, 0.1))
        rejection_plot.grid(True, axis="y")

    def plot(self, figsize: Tuple[int, int] = (12, 14)) -> None:
        """General plot method to combine for now plot_occupancy and
        plot_rejection.

        :param figsize: Figure size to use.
        """

        fig = plt.figure(figsize=figsize)

        occupancy_plot = fig.add_subplot(211)
        self.plot_occupancy(figsize, occupancy_plot)

        rejection_plot = fig.add_subplot(212)
        self.plot_rejection(figsize, rejection_plot)

        fig.tight_layout()

    def _initialize_hospital(self, eval_time: float) -> Hospital:
        """For simulation, initialize the hospital.

        :param eval_time: The initial evaluation time to run.

        :return: The initialized hospital.
        """

        env = sm.Environment()

        hospital = Hospital(env, self.hospital_specs)

        hospital.setup(self.specs.init_mode)

        # run the hospital first in warm-up mode
        hospital.run(eval_time)
        hospital.reporter.reset_utilities(hospital.env.now)

        return hospital

    def reset(self):
        """Reset the evaluator to its initial state."""
        super().reset()
        self.error = float("inf")
        self.runs = 0

    def evaluate(self) -> ReporterDistributions:
        """Evaluate the hospital.

        :return: The observed distributions.
        """
        occupancy_ = np.zeros_like(self.distributions.occupancy)
        k = 0

        self.logger.info("Start evaluation at {}".format(
            datetime.datetime.now()))

        while self.error >= self.specs.error_bound and self.runs < self.specs.max_runs:
            # while error is large and we did not made too many runs we simulate

            if self.specs.processors > 1:
                # start simulation in parallel mode, async
                pool = mp.Pool(processes=self.specs.processors)

                for _ in range(self.specs.run_size * max(2**(k - 1), 1)):
                    pool.apply_async(self.evaluate_single,
                                     args=(np.random.randint(0, 2**16), ),
                                     callback=self.evaluate_callback,
                                     error_callback=self.error_callback)

                pool.close()
                pool.join()

            else:
                # otherwise one by one
                for _ in range(self.specs.run_size * max(2**(k - 1), 1)):
                    result = self.evaluate_single(np.random.randint(0, 2**16))
                    self.evaluate_callback(result)

            # we increase the run size
            self.runs += self.specs.run_size * max(2**(k - 1), 1)
            k += 1

            # compute the error
            if occupancy_.any():
                self.error = self.specs.loss(
                    occupancy_.T, self.distributions.occupancy.T / self.runs)

            # get latest occupancy distributions
            occupancy_ = self.distributions.occupancy / self.runs

            self.logger.info(
                f"Multi Simulation runner finished iteration {k} with: "
                f"k={k - 1}, error={self.error}, "
                f"#(finished single simulations)={self.runs}.")

        # finished, so set the distributions
        self.distributions /= self.runs

        self.logger.info(f"Finished Multi Simulation: "
                         f"k={k - 1}, error={self.error}, "
                         f"#(finished single simulations)={self.runs}.")

        return self.distributions

    def evaluate_single(
            self,
            seed: int = None) -> Tuple[ReporterDistributions, float, float]:
        """Evaluate a single run.

        :param seed: The seed with which numpy should be seeded. Useful in multiprocessing mode!

        :return: The observed distributions, the error, and the hospital object.
        """

        # seed to not evaluate the same in parallel mode
        np.random.seed(seed)

        # create the model
        hospital = self._initialize_hospital(self.specs.eval_time)

        # start running in reporting mode
        error = np.inf

        k = 1

        hospital.run(hospital.env.now + self.specs.eval_time)

        reporter_k_1 = hospital.reporter

        while (error >= self.specs.error_bound_single
               and hospital.env.now < self.specs.max_eval_time):
            # while error is large and we did not run too long we simulate

            reporter_k = reporter_k_1.copy()

            hospital.reporter.reset_utilities(hospital.env.now)
            hospital.run(hospital.env.now + self.specs.eval_time * (2**k))
            reporter_k_1 = hospital.reporter
            occupancy_k = reporter_k.occupancy
            occupancy_k_1 = reporter_k_1.occupancy
            occupancy_full = occupancy_k / 3 + occupancy_k_1 * (2 / 3)

            error = self.specs.loss(occupancy_k.T, occupancy_full.T)

            k += 1

        distributions = reporter_k.distributions / 3 + reporter_k_1.distributions * (
            2 / 3)

        return distributions, error, hospital.env.now

    def evaluate_callback(
            self, result: Tuple[ReporterDistributions, float, float]) -> None:
        """Set the attributes obtained in one run from evaluate_single.

        :param result: Containes distributions reuslt, the current error and the current time.
        """
        distributions = result[0]
        error = result[1]
        time = result[2]
        self.distributions += distributions
        self.logger.info(f"Single Simulation runner: "
                         f"error={error}, simulation time={time}.")

    def error_callback(self, error: BaseException) -> None:
        """Callback function for multiprocessing.

        :param error: The error received.
        """
        self.logger.warning(error)


class MarkovEvaluator(Evaluator):
    """A class, which creates the explicit Q matrix to solve for pi.

    :param hospital_specs: The hospital specifications.
    :param truncation: The truncation of the state space.
    Exp.:If the state space gets too large, the solution is not computable. If it
    is known, that pi has positive probability only for high occupancies, it does make sense and
    thus computation still feasible, to not include lower capacities in the computation.
    Note: If MemoryError is received during Q-Matrix building (indicating that state space is too
    large), truncation will be automatically be setted higher.
    :param logger: Logger to use for information logging.
    """
    name = "Markov_Evaluator"

    def __init__(self,
                 hospital_specs: HospitalSpecs,
                 truncation: float = 1.,
                 logger: Optional[logging.Logger] = None):

        logger = logger if logger is not None else get_logger(
            f"{__file__} - Markov Evaluator",
            file_path=LOGGING_DIR.joinpath(
                f"{__file__} - Markov Evaluator.log"))
        super().__init__(hospital_specs, logger=logger)
        if np.any(self.hospital_specs.holdings):
            self.logger.warning(
                "Markov Evaluator cannot handle holding logic, "
                "thus will ignore holdings. Holding information is "
                f"{self.hospital_specs.holdings}.")
        if any(len(w) > 0 for w in self.hospital_specs.waitings.values()):
            self.logger.warning(
                "Markov Evaluator cannot handle waiting logic, "
                "thus will ignore waiting. Waiting information is "
                f"{self.hospital_specs.waitings}.")

        if self.hospital_specs.U != 1:
            self.logger.warning(
                "Markov Evaluator cannot handle multiple classes, "
                f"thus will ignore all classes besides 1. "
                f"Number of classes is {self.hospital_specs.U}")
        if np.any(self.hospital_specs.arrival[:, 0, 1] != 0):
            self.logger.warning(
                "Markov Evaluator cannot handle internal arrival, "
                "thus will ignore this information.")

        self.logger.info(
            "Just taking arrival rate. Do not differentiate between exponential "
            "or hypererlang.")
        self.arrival_rate = self.hospital_specs.arrival_rate[:, 0,
                                                             0].reshape(-1)
        self.service_rate = self.hospital_specs.service_rate[:, 0].reshape(-1)
        # allow truncation, i.e. only observing the upper states.

        self.Q = np.zeros((1, ))
        self.full_pi = np.zeros((1, ))
        self._truncation = 1.
        self.set_truncated(truncation)
        self.pi = np.zeros((self.hospital_specs.capacities.shape[0],
                            self.hospital_specs.capacities.max() + 1))

        self.shape = self.arrival_rate.shape

    def _build_plain_truncated_Q(self, truncation: float) -> np.ndarray:
        """Build the truncated Q-matrix.

        :param truncation: The truncation to apply.

        :return: The truncated Q matrix.
        """
        Q = np.zeros(np.concatenate(
            ((self.hospital_specs.capacities + 1) * truncation,
             (self.hospital_specs.capacities + 1) * truncation)).astype("int"),
                     dtype="float32")

        return Q

    def _build_plain_truncated_full_pi(self, truncation: float) -> np.ndarray:
        """Build the truncated full_pi array.

        :param truncation: The truncation to apply.

        :return: The truncated full_pi array.
        """
        full_pi = np.zeros(((((self.hospital_specs.capacities + 1) *
                              truncation).astype("int").prod())))

        return full_pi

    def set_truncated(self, truncation: float) -> None:
        """Apply new truncation.

        :param truncation: The truncation to apply.
        """

        created = False
        while not created and truncation > 0.1:
            try:
                self.Q = self._build_plain_truncated_Q(truncation)
                self.full_pi = self._build_plain_truncated_full_pi(truncation)
                created = True
            except MemoryError:
                truncation = self._memory_error(truncation)

        self._truncation = truncation

    def _memory_error(self, truncation: float) -> float:
        """Adjust truncation downwards if an memory error occurs."""
        truncation = truncation * 0.9
        self.logger.warning(
            f"Memroy error. Setting truncation to {truncation}.")

        return truncation

    def evaluate(self):
        """Run evaluation."""

        self.compute_pi()
        self.distributions.occupancy = self.pi

    # pylint: disable=too-many-nested-blocks
    def build_Q(self) -> None:
        """Build the Q matrix explicitly."""

        capacities = self.hospital_specs.capacities

        routing = self.hospital_specs.routing[:, 0, :, 0].reshape(
            (capacities.shape[0], capacities.shape[0] + 1))
        routing[:, -1] = 1 - np.sum(routing[:, :-1], axis=1)

        for index in np.ndindex(
                tuple(((capacities + 1) * self._truncation).astype("int"))):
            for i, j in itertools.product(np.arange(capacities.shape[0]),
                                          repeat=2):

                real_index = (np.ceil(
                    (capacities + 1) *
                    (1 - self._truncation))).astype("int") + np.array(index)

                if i == j:
                    # out in the wild
                    index_ = np.array(index)
                    if index[i] > 0:
                        index_[i] -= 1
                        self.Q[tuple(np.concatenate(
                            (index, index_)))] += routing[i, -1] * min(
                                real_index[i],
                                capacities[i]) * self.service_rate[i]
                        index_[i] += 1

                    # coming from the wild
                    if real_index[i] < capacities[i]:
                        index_[i] += 1
                        self.Q[tuple(np.concatenate(
                            (index, index_)))] += self.arrival_rate[i]
                        index_[i] -= 1

                else:
                    index_ = np.array(index)
                    if index[i] > 0:
                        index_[i] -= 1
                        if real_index[j] < capacities[j]:
                            index_[j] += 1

                        self.Q[tuple(np.concatenate(
                            (index, index_)))] += routing[i, j] * min(
                                real_index[i],
                                capacities[i]) * self.service_rate[i]

        self.Q = self.Q.reshape(
            (((capacities + 1) * self._truncation).astype("int").prod(),
             ((capacities + 1) * self._truncation).astype("int").prod()))
        self.Q -= np.diag(np.sum(self.Q, axis=1))

    def compute_full_pi(self) -> None:
        """Compute the full_pi according to Q matrix."""

        capacities = self.hospital_specs.capacities

        self.build_Q()

        # adjust it, such that the LGS gets solvable
        Q = self.Q.copy()
        Q[:, -1] = np.ones(
            (((capacities + 1) * self._truncation).astype("int").prod()))
        b = np.zeros(
            (((capacities + 1) * self._truncation).astype("int").prod()))
        b[-1] = 1

        # solve it
        self.full_pi = scipy.linalg.solve(Q.T, b)

        self.full_pi = self.full_pi.reshape(
            ((capacities + 1) * self._truncation).astype("int"))

    def compute_pi(self) -> None:
        """Compute pi per ward according to full_pi."""

        capacities = self.hospital_specs.capacities

        while not self.full_pi.any() and self._truncation > 0.1:
            try:
                self.compute_full_pi()
            except MemoryError:
                truncation = self._memory_error(self._truncation)
                self.set_truncated(truncation)

        # compute individual wardary distributions
        for i, n in enumerate((capacities + 1) - (
            (capacities + 1) * self._truncation).astype("int")):
            pi_i = np.sum(
                self.full_pi,
                axis=tuple([j for j in range(capacities.shape[0]) if j != i]))
            pi_i[capacities[i] - n] = pi_i[capacities[i] - n:].sum()
            self.pi[i, n:capacities[i] + 1] = pi_i[:capacities[i] - n + 1]


class AnalyticEvaluator(Evaluator):
    """A class for solving pi analytically.

    :param hospital_specs: The hospital specifications.
    :param logger: Logger to use for information logging.
    """

    name = "Analytic_Evaluator"

    def __init__(self,
                 hospital_specs: HospitalSpecs,
                 logger: Optional[logging.Logger] = None):

        logger = logger if logger is not None else get_logger(
            f"{__file__} - Analytic Evaluator",
            file_path=LOGGING_DIR.joinpath(
                f"{__file__} - Analytic Evaluator.log"))
        super().__init__(hospital_specs, logger=logger)
        if np.any(self.hospital_specs.holdings):
            self.logger.warning(
                "Analytic Evaluator cannot handle holding logic, "
                "thus will ignore holdings. Holding information is "
                f"{self.hospital_specs.holdings}.")
        if any(len(w) > 0 for w in self.hospital_specs.waitings.values()):
            self.logger.warning(
                "Analytic Evaluator cannot handle waiting logic, "
                "thus will ignore waiting. Waiting information is "
                f"{self.hospital_specs.waitings}.")
        if np.any(self.hospital_specs.arrival[:, :, 1] != 0):
            self.logger.warning(
                "Analytic Evaluator cannot handle internal arrival, "
                "thus will ignore this information.")
        self.arrival_rate = self.hospital_specs.arrival_rate[:, :, 0]
        self.service_rate = self.hospital_specs.service_rate[:, :self.
                                                             hospital_specs.U]
        self.nz_service_rate = self.service_rate.copy()
        self.nz_service_rate[self.nz_service_rate == 0] = 1
        self.shape = self.arrival_rate.shape
        self.alpha_lin = np.zeros_like(self.arrival_rate)
        self.alpha_res = np.zeros_like(self.arrival_rate)
        self.alpha_res_success = False
        self.alpha_res_message = None
        self.pi_lin = np.zeros((self.service_rate.shape[0],
                                self.hospital_specs.capacities.max() + 1),
                               dtype="float")
        self.pi_res = np.zeros((self.service_rate.shape[0],
                                self.hospital_specs.capacities.max() + 1),
                               dtype="float")

    def evaluate(self) -> None:
        """Run evaluation."""

        self.solve_pi()
        self.distributions.occupancy = self.pi_res

    @property
    def occupancy(self) -> np.ndarray:
        """Redefine occupancy method.

        :return: Occupancy.
        """
        out = self.pi_res.copy()
        for i, val in enumerate(out):
            cap = self.hospital_specs.capacities[i]
            out[i, cap] = val[cap:].sum()
        out = out[:, :self.hospital_specs.capacities.max() + 1]
        return out

    def build_single_class(self) -> "AnalyticEvaluator":
        """Build a single class model from a multi-class model using the
        computed solution for the multiclass model.

        :return: New AnalyticalEvaluator instance with new hospital specs.
        """

        alpha = self.alpha_res
        arrival = np.zeros(shape=(self.hospital_specs.I, 1), dtype="O")
        service = np.zeros(shape=(self.hospital_specs.I, 1), dtype="O")
        routing = np.zeros(shape=(self.hospital_specs.I, 1,
                                  self.hospital_specs.I + 1, 1),
                           dtype="float")

        # compute arrival
        for i, ward_arrival in enumerate(self.hospital_specs.arrival):
            lambdas = np.array([
                1 / distribution.mean() for distribution in ward_arrival
                if distribution != 0
            ])
            mean = 1 / lambdas.sum()
            arrival[i, 0] = scipy.stats.expon(scale=mean)

        # compute service
        for i, ward_alpha in enumerate(alpha):
            if ward_alpha.sum():
                mean = np.sum(ward_alpha * np.array([
                    distribution.mean() if distribution != 0 else 0
                    for distribution in self.hospital_specs.service[i]
                ])) / ward_alpha.sum()
            else:
                mean = 1
            service[i, 0] = scipy.stats.expon(scale=mean)

        # compute routing
        for i, ward_alpha in enumerate(alpha):
            for j, r in enumerate(
                    self.hospital_specs.routing.sum(axis=3).transpose(
                        (0, 2, 1))[i]):
                if ward_alpha.sum():
                    r_ = np.sum(r * ward_alpha) / ward_alpha.sum()
                else:
                    r_ = 0
                routing[i, 0, j, 0] = r_
            if routing[i, 0, :, 0].sum() == 0:
                routing[i, 0, -1, 0] = 1

        hospital_specs = HospitalSpecs(self.hospital_specs.capacities, arrival,
                                       service, routing)

        return AnalyticEvaluator(hospital_specs)

    def solve_alpha_lin(self) -> None:
        """Solve linear traffic equation."""

        routing = self.hospital_specs.routing[:, :, :-1, :].reshape(
            (self.hospital_specs.routing.shape[0] *
             self.hospital_specs.routing.shape[1],
             self.hospital_specs.routing.shape[0] *
             self.hospital_specs.routing.shape[1])) - np.eye(
                 self.hospital_specs.routing.shape[0] *
                 self.hospital_specs.routing.shape[1])

        arrival_rate = self.arrival_rate.reshape(-1)

        self.alpha_lin = np.linalg.solve(routing.T, -arrival_rate)

        self.alpha_lin = self.alpha_lin.reshape(self.shape)

    def solve_alpha_res(self) -> None:
        """Solve adjusted traffic equation."""

        if not self.alpha_lin.any():
            self.solve_alpha_lin()

        alpha_ = self.alpha_lin.reshape(self.shape)

        sol = root(self.alpha_func, alpha_, jac=self.alpha_jac, tol=1e-10)
        self.alpha_res_success = sol.success
        self.alpha_res_message = sol.message
        self.alpha_res = sol.x.reshape(self.shape)
        self.logger.info(f"Solved for alpha_res with success: {sol.success}, "
                         f"message: {sol.message}.")

    def alpha_func(self, alpha: np.ndarray) -> np.ndarray:
        """Traffic equation as function. alpha_func = 0 has to hold if traffic equation holds.

        :param alpha: The alpha, where to evaluate.
        """

        alpha = alpha.reshape(self.shape)

        pi = self.create_pi(alpha)
        not_full = np.array([
            1 - pi[i, capacity]
            for i, capacity in enumerate(self.hospital_specs.capacities)
        ])

        routing = ((self.hospital_specs.routing[:, :, :-1, :].transpose(
            (2, 3, 1, 0)) * not_full).transpose((0, 1, 3, 2)))

        out = self.arrival_rate + (routing * alpha).sum(axis=(2, 3)) - alpha
        return out.reshape(-1)

    def alpha_jac(self, alpha: np.ndarray) -> np.ndarray:
        """Explicit jacobian of alpha_func. This makes the root searching much
        faster and more efficient.

        :param alpha: The alpha where to evaluate.
        """

        alpha = alpha.reshape(self.shape)
        pi = self.create_pi(alpha)
        not_full = np.array([
            1 - pi[i, capacity]
            for i, capacity in enumerate(self.hospital_specs.capacities)
        ])
        routing = self.hospital_specs.routing[:, :, :-1, :].transpose(
            (2, 3, 0, 1))
        coefs = (alpha / self.nz_service_rate).sum(axis=1)

        jac = np.array([np.sum(routing * alpha, axis=3)] *
                       routing.shape[3]).transpose((1, 2, 3, 0))
        quo_der = np.zeros_like(self.service_rate, dtype="float")
        for index, capacity in enumerate(self.hospital_specs.capacities):
            k = np.arange(capacity + 1)
            sum_params = coefs[index]**k / factorial(k)
            quo_der[index, :] = (1 / self.nz_service_rate[index]) * (
                np.sum(sum_params[:-1]) * sum_params[-1] -
                np.sum(sum_params) * sum_params[-2]) / (np.sum(sum_params)**2)

        jac *= quo_der
        jac += (routing.transpose((0, 1, 3, 2)) * not_full).transpose(
            (0, 1, 3, 2))

        jac = jac.reshape((jac.shape[0] * jac.shape[1], jac.shape[0] *
                           jac.shape[1])) - np.eye(jac.shape[0] * jac.shape[1])

        return jac

    def create_pi(self, alpha: np.ndarray) -> np.ndarray:
        """Create the pi for given alpha assuming M/M/K/K queue.

        :param alpha: The incoming traffic at each node.
        """

        pi = np.zeros((self.service_rate.shape[0],
                       self.hospital_specs.capacities.max() + 1),
                      dtype="float")

        coefs = (alpha / self.nz_service_rate).sum(axis=1)

        for index, capacity in enumerate(self.hospital_specs.capacities):
            k = np.arange(capacity + 1)
            pi[index, :capacity + 1] = coefs[index]**k / factorial(k)
        pi = (pi.transpose() / pi.sum(axis=1)).transpose()

        return pi

    def solve_pi(self) -> None:
        """Solve for pi."""

        if not self.alpha_lin.any():
            self.solve_alpha_lin()

        if not self.alpha_res.any():
            self.solve_alpha_res()

        self.pi_lin = self.create_pi(self.alpha_lin)
        self.pi_res = self.create_pi(self.alpha_res)
