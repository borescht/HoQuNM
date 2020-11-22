"""Command line tools for optimisation."""

import datetime
import json
import logging
from pathlib import Path
from typing import List

import click
import matplotlib.pyplot as plt
import numpy as np

from hoqunm.data_tools.base import (EXAMPLE_FILEPATH_OPTIMISATION_COMPUTATION,
                                    EXAMPLE_FILEPATH_OPTIMISATION_SIMULATION,
                                    EXAMPLE_MODEL_NO_CART,
                                    OUTPUT_DIR_OPTIMISATION)
from hoqunm.data_tools.modelling import HospitalModel
from hoqunm.optimisation.optimators import Optimator
from hoqunm.simulation.evaluators import (EvaluationResults,
                                          SimulationEvaluator,
                                          SimulationEvaluatorSpecs)
from hoqunm.utils.utils import get_logger

# pylint: disable=too-many-locals
# pylint: disable=broad-except


def _create_plots(results: List[EvaluationResults], rejection: bool,
                  profit: bool, logger: logging.Logger,
                  utilisation_constraints: np.ndarray,
                  rejection_costs: np.ndarray, bed_costs: np.ndarray,
                  eps_rejection: float, eps_profit_rejection: float,
                  output_dir: Path) -> None:
    optimator = Optimator(results)

    if rejection:
        logger.info("Get optimum w.r.t. minimal rejection.")
        try:
            idx, values = optimator.utilisation_restricted(
                utilisation_constraints=utilisation_constraints)

            result = optimator.results[idx]
            logger.info(f"Chosen opitimum for rejection: \n"
                        f"capacities: {result.hospital_specs.capacities}\n"
                        f"rejection: {result.rejection}\n"
                        f"utilisation: {result.utilisation()}\n"
                        f"profit: {result.profit()}\n"
                        f"profit_rejection: {result.profit_rejection()}")

            logger.info(
                f"Chosen opitimum for rejection range: "
                f"{values[idx] - eps_rejection}, {values[idx]}, {values[idx] + eps_rejection}"
            )

            for angle in [3, 15]:
                for rotation in [300, 60]:
                    idx_ = [
                        i for i, v in enumerate(values)
                        if abs(values[idx] - v) < eps_rejection and i != idx
                    ]
                    optimator.plot_results(
                        op_idx=idx,
                        values=values,
                        op_idx_rel=idx_,
                        color_map="viridis_r",
                        angle=angle,
                        rotation=rotation,
                        savepath=output_dir,
                        filename=f"rejection - "
                        f"utilisation_constraints[{utilisation_constraints}] - "
                        f"angle[{angle}]_rotation[{rotation}]")
                    plt.close()
        except ValueError:
            logger.warning(f"Failed for rejection.")

    if profit:
        try:
            idx, values = optimator.profit_rejection(
                bed_costs=bed_costs, rejection_costs=rejection_costs)

            result = optimator.results[idx]

            logger.info(f"Chosen opitimum for profit_rejection: \n"
                        f"capacities: {result.hospital_specs.capacities}\n"
                        f"rejection: {result.rejection}\n"
                        f"utilisation: {result.utilisation()}\n"
                        f"profit: {result.profit()}\n"
                        f"profit_rejection: {result.profit_rejection()}")

            logger.info(
                f"Chosen opitimum for profit_rejection range: "
                f"{values[idx] - eps_profit_rejection}, {values[idx]}, "
                f"{values[idx] + eps_profit_rejection}")

            for angle in [3, 15]:
                for rotation in [300, 60]:
                    idx_ = [
                        i for i, v in enumerate(values)
                        if abs(values[idx] -
                               v) < eps_profit_rejection and i != idx
                    ]

                    optimator.plot_results(
                        op_idx=idx,
                        values=values,
                        op_idx_rel=idx_,
                        color_map="viridis_r",
                        angle=angle,
                        rotation=rotation,
                        savepath=output_dir,
                        filename=
                        f"profit_rejection - angle[{angle}]_rotation[{rotation}]"
                    )
                    plt.close()
        except ValueError as e:
            logger.warning(f"Failed for profit. {e}")

    logger.info("Finished optimisation computation.")


def _get_evaluation_results(results_path: Path,
                            logger: logging.Logger) -> List[EvaluationResults]:
    evaluation_results: List[EvaluationResults] = []
    for file in results_path.glob("*.json"):
        try:
            evaluation_results.append(EvaluationResults.load(file))
        except BaseException as e:
            logger.warning(f"Not able to load {file}. {e}.")
    return evaluation_results


@click.command()
@click.option(
    "--specsfile",
    "-s",
    type=click.Path(exists=True),
    default=str(EXAMPLE_FILEPATH_OPTIMISATION_SIMULATION),
    required=True,
    help="Filepath to specifications for model building. "
    f"Default can be found in {EXAMPLE_FILEPATH_OPTIMISATION_SIMULATION}.")
@click.option("--model",
              "-m",
              type=click.Choice(["1", "2", "3"]),
              default="1",
              required=True,
              help="Model to evaluate. You can choose between 1,2 and 3.\n"
              "Default: 1")
@click.option(
    "--waiting",
    "-w",
    is_flag=True,
    help="If waiting shall be assessed according to given waiting map.")
@click.option("--rejection",
              "-r",
              is_flag=True,
              help="Optimise according to minimal rejection.")
@click.option("--profit",
              "-p",
              is_flag=True,
              help="Optimise according to profit.")
def simulate_optimum(specsfile: str, model: str, waiting: bool,
                     rejection: bool, profit: bool):
    """Analyse different capacity combinations according to the specified
    optimisation problem."""

    with open(specsfile, "r") as f:
        specs = json.load(f)

    output_dir = Path(
        specs["output_dir"]
    ) if specs["output_dir"] is not None else OUTPUT_DIR_OPTIMISATION
    if not output_dir.is_dir():
        output_dir.mkdir()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    logger = get_logger(
        "simulate_optimum",
        output_dir.joinpath(f"simulate_optimum - {timestamp}.log"))

    if specs["modelfile"] is None:
        logger.info(
            f"No model file given. Recursing to default models from {EXAMPLE_MODEL_NO_CART}."
        )
        modelfile = EXAMPLE_MODEL_NO_CART
    else:
        modelfile = Path(specs["modelfile"])

    if not modelfile.is_file():
        raise FileNotFoundError(modelfile)

    wards = specs["wards"]
    capacities = specs["capacities"]
    adjust_int_rates = specs["adjust_int_rates"]
    service_name = specs["service_name"]
    if service_name not in ["expon", "hypererlang"]:
        raise ValueError(
            f"service_name has to be one of [expon, hypererlang]. Current value: {service_name}."
        )
    waitings = specs["waitings"] if waiting else dict()
    simulation_evaluator_specs = SimulationEvaluatorSpecs(**specs["DES_specs"])
    optimisation_specs = specs["optimisation_specs"]
    lower_capacities = np.array(optimisation_specs["lower_capacities"])
    upper_capacities = np.array(optimisation_specs["upper_capacities"])
    utilisation_constraints = np.array(
        optimisation_specs["utilisation_constraints"])
    bed_costs = np.array(optimisation_specs["bed_costs"])
    rejection_costs = np.array(optimisation_specs["rejection_costs"])
    eps_rejection = optimisation_specs["eps_rejection"]
    eps_profit_rejection = optimisation_specs["eps_profit_rejection"]

    results: List[EvaluationResults] = []

    combinations = np.prod(upper_capacities - lower_capacities + 1)
    logger.info(f"Start simulation of possible capacity combinations. "
                f"# combinations={combinations}.")

    ward_capacity = dict(zip(wards, capacities))

    hospital_model = HospitalModel.load(filepath=modelfile, logger=logger)
    hospital_specs = hospital_model.get_model(
        model=int(model),
        capacities=ward_capacity,
        service_name=service_name,
        adjust_int_rates=adjust_int_rates,
        waitings=waitings)

    if not (profit or rejection):
        logger.info("No optimisation routine specified.")
        raise ValueError("No optimisation routine specified.")

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = output_dir.joinpath("Simulation results - " + timestamp)
    results_dir.mkdir()

    for i, index in enumerate(
            np.ndindex(*(upper_capacities - lower_capacities + 1))):
        capacities_ = lower_capacities + np.array(index)

        ward_capacity = dict(zip(wards, capacities_))

        hospital_specs.set_capacities(**ward_capacity)

        logger.info(
            f"Simulate model on capacities {capacities_}. {i + 1} of {combinations}"
        )

        simulation_evaluator = SimulationEvaluator(
            hospital_specs=hospital_specs,
            simulation_evaluator_specs=simulation_evaluator_specs,
            logger=logger)
        simulation_evaluator.evaluate()

        key = f"{modelfile.name},model:{model},service:{service_name},waiting:{waiting}"
        simulation_evaluator.name = key

        results.append(simulation_evaluator)

        simulation_evaluator.save(
            results_dir.joinpath(
                f"simulation_result-capacities"
                f"{list(simulation_evaluator.hospital_specs.capacities)}.json")
        )

    _create_plots(results=results,
                  rejection=rejection,
                  profit=profit,
                  logger=logger,
                  utilisation_constraints=utilisation_constraints,
                  rejection_costs=rejection_costs,
                  bed_costs=bed_costs,
                  eps_rejection=eps_rejection,
                  eps_profit_rejection=eps_profit_rejection,
                  output_dir=output_dir)


@click.command()
@click.option(
    "--specsfile",
    "-s",
    type=click.Path(exists=True),
    default=str(EXAMPLE_FILEPATH_OPTIMISATION_COMPUTATION),
    required=True,
    help="Filepath to specifications for model building. "
    f"Default can be found in {EXAMPLE_FILEPATH_OPTIMISATION_COMPUTATION}.")
@click.option("--rejection",
              "-r",
              is_flag=True,
              help="Optimise according to minimal rejection.")
@click.option("--profit",
              "-p",
              is_flag=True,
              help="Optimise according to profit.")
def compute_optimum(specsfile: str, rejection: bool, profit: bool):
    """Analyse different capacity combinations according to the specified
    optimisation problem.

    Take results from specified direcotry.
    """

    with open(specsfile, "r") as f:
        specs = json.load(f)

    output_dir = Path(
        specs["output_dir"]
    ) if specs["output_dir"] is not None else OUTPUT_DIR_OPTIMISATION
    if not output_dir.is_dir():
        output_dir.mkdir()

    input_dir = Path(specs["input_dir"])
    if not input_dir.is_dir():
        raise NotADirectoryError(input_dir)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    logger = get_logger(
        "compute_optimum",
        output_dir.joinpath(f"compute_optimum - {timestamp}.log"))

    optimisation_specs = specs["optimisation_specs"]
    utilisation_constraints = np.array(
        optimisation_specs["utilisation_constraints"])
    bed_costs = np.array(optimisation_specs["bed_costs"])
    rejection_costs = np.array(optimisation_specs["rejection_costs"])
    eps_rejection = optimisation_specs["eps_rejection"]
    eps_profit_rejection = optimisation_specs["eps_profit_rejection"]

    if not (profit or rejection):
        logger.info("No optimisation routine specified.")
        raise ValueError("No optimisation routine specified.")

    results = _get_evaluation_results(Path(input_dir), logger=logger)

    _create_plots(results=results,
                  rejection=rejection,
                  profit=profit,
                  logger=logger,
                  utilisation_constraints=utilisation_constraints,
                  rejection_costs=rejection_costs,
                  bed_costs=bed_costs,
                  eps_rejection=eps_rejection,
                  eps_profit_rejection=eps_profit_rejection,
                  output_dir=output_dir)
