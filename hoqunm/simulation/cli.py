"""Command line tools for model simulation."""

import datetime
import json
from pathlib import Path
from typing import Any, Dict, List

import click
import matplotlib.pyplot as plt

from hoqunm.data_tools.base import (EXAMPLE_FILEPATH_ASSESS,
                                    EXAMPLE_FILEPATH_VARIANTS,
                                    EXAMPLE_MODEL_CART, EXAMPLE_MODEL_NO_CART,
                                    OUTPUT_DIR_ASSESS_CAPACITIES,
                                    OUTPUT_DIR_MODEL_VARIANTS)
from hoqunm.data_tools.modelling import HospitalModel
from hoqunm.simulation.evaluators import (EvaluationResults,
                                          SimulationEvaluator,
                                          SimulationEvaluatorSpecs)
from hoqunm.utils.utils import MODEL_COLORS, get_logger


@click.command()
@click.option("--specsfile",
              "-s",
              type=click.Path(exists=True),
              default=str(EXAMPLE_FILEPATH_VARIANTS),
              required=True,
              help="Filepath to specifications for model building. "
              f"Default can be found in {EXAMPLE_FILEPATH_VARIANTS}.")
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
def analyse_model_variants(specsfile: str, model: str, waiting: bool):
    """Analyse the model under its different variants."""
    with open(specsfile, "r") as f:
        specs = json.load(f)

    output_dir = specs(
        specs["output_dir"]
    ) if specs["output_dir"] is not None else OUTPUT_DIR_MODEL_VARIANTS
    if not output_dir.is_dir():
        output_dir.mkdir()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logger = get_logger(
        "analyse_model_variants",
        output_dir.joinpath(f"analyse_model_variants - {timestamp}.log"))

    if specs["modelfiles"] is None or all(f is None
                                          for f in specs["modelfiles"]):
        logger.info(
            f"No model file given. Recursing to default models from {EXAMPLE_MODEL_NO_CART} and "
            f"{EXAMPLE_MODEL_CART}.")
        modelfiles = [EXAMPLE_MODEL_NO_CART, EXAMPLE_MODEL_CART]
    else:
        modelfiles = [Path(modelfile) for modelfile in specs["modelfiles"]]

    if any(not modelfile.is_file() for modelfile in modelfiles):
        raise FileNotFoundError(modelfiles)

    wards = specs["wards"]
    capacities = specs["capacities"]
    ward_capacity = dict(zip(wards, capacities))
    adjust_int_rates = specs["adjust_int_rates"]
    waitings = specs["waitings"] if waiting else dict()
    simulation_evaluator_specs = SimulationEvaluatorSpecs(**specs["DES_specs"])

    results: Dict[str, Any] = dict()
    evaluators: List[EvaluationResults] = []

    for modelfile in modelfiles:
        logger.info(f"Assessing model from file {modelfile}.")
        hospital_model = HospitalModel.load(filepath=modelfile, logger=logger)
        for service_name in list(
                hospital_model.wards.values())[0].service.__dict__.keys():
            hospital_specs = hospital_model.get_model(
                model=int(model),
                capacities=ward_capacity,
                service_name=service_name,
                adjust_int_rates=adjust_int_rates,
                waitings=waitings)
            simulation_evaluator = SimulationEvaluator(
                hospital_specs=hospital_specs,
                simulation_evaluator_specs=simulation_evaluator_specs,
                logger=logger)
            simulation_evaluator.evaluate()

            key = f"{modelfile},service:{service_name},waiting:{waiting}"
            results[key] = simulation_evaluator.save_dict()

            simulation_evaluator.name = key
            evaluators.append(simulation_evaluator)

    results_file = output_dir.joinpath(f"model_results_{model}.json")

    with open(results_file, "w") as f:
        json.dump(results, f)

    hospital_model = HospitalModel.load(filepath=modelfiles[0], logger=logger)
    real_observation = hospital_model.occupancy_as_evaluator(**ward_capacity)
    real_observation.plot_against(evaluators, top=True)
    plt.savefig(
        output_dir.joinpath(f"Occupancy distribution - model {model}.pdf"))
    plt.close()

    logger.info(
        f"Finished successfully. You can find the results in {results_file}.")


@click.command()
@click.option("--specsfile",
              "-s",
              type=click.Path(exists=True),
              default=str(EXAMPLE_FILEPATH_ASSESS),
              required=True,
              help="Filepath to specifications for model building. "
              f"Default can be found in {EXAMPLE_FILEPATH_ASSESS}.")
@click.option(
    "--waiting",
    "-w",
    is_flag=True,
    help="If waiting shall be assessed according to given waiting map.")
def assess_capacities(specsfile: str, waiting: bool):
    """Assess the capacities for the different models."""

    with open(specsfile, "r") as f:
        specs = json.load(f)

    output_dir = specs(
        specs["output_dir"]
    ) if specs["output_dir"] is not None else OUTPUT_DIR_ASSESS_CAPACITIES
    if not output_dir.is_dir():
        output_dir.mkdir()

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    logger = get_logger(
        "assess_capacities",
        output_dir.joinpath(f"assess_capacities - {timestamp}.log"))

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
    ward_capacity = dict(zip(wards, capacities))
    adjust_int_rates = specs["adjust_int_rates"]
    service_name = specs["service_name"]
    if service_name not in ["expon", "hypererlang"]:
        raise ValueError(
            f"service_name has to be one of [expon, hypererlang]. Current value: {service_name}."
        )
    waitings = specs["waitings"] if waiting else dict()
    simulation_evaluator_specs = SimulationEvaluatorSpecs(**specs["DES_specs"])

    results: Dict[str, Any] = dict()

    logger.info(f"Assessing hospital model from {modelfile}.")
    evaluators: List[EvaluationResults] = []

    for model in range(1, 4):
        logger.info(f"Assessing model {model}.")
        hospital_model = HospitalModel.load(filepath=modelfile, logger=logger)
        hospital_specs = hospital_model.get_model(
            model=int(model),
            capacities=ward_capacity,
            service_name=service_name,
            adjust_int_rates=adjust_int_rates,
            waitings=waitings)
        simulation_evaluator = SimulationEvaluator(
            hospital_specs=hospital_specs,
            simulation_evaluator_specs=simulation_evaluator_specs,
            logger=logger)
        simulation_evaluator.evaluate()

        key = f"{modelfile.name},model:{model},service:{service_name},waiting:{waiting}"
        results[key] = simulation_evaluator.save_dict()

        simulation_evaluator.name = key
        evaluators.append(simulation_evaluator)

    hospital_model = HospitalModel.load(filepath=modelfile, logger=logger)
    real_observation = hospital_model.occupancy_as_evaluator(**ward_capacity)

    colors = MODEL_COLORS + ["r"] * 3
    shifts = [-1 / 6, 0, 1 / 6] + [-1 / 6, 0, 1 / 6]
    markers = ["*"] * 3 + ["*"] * 3
    labels = [e.name for e in evaluators
              ] + [real_observation.name, None, None]  # type: ignore
    EvaluationResults.plot_many(evaluation_results=evaluators +
                                [real_observation] * 3,
                                colors=colors,
                                shifts=shifts,
                                markers=markers,
                                labels=labels)

    plt.savefig(
        output_dir.joinpath(
            f"Model results - {modelfile.stem}_wards{wards}_capacities{capacities}_"
            f"service[{service_name}]_waiting[{waiting}].pdf"))
    plt.close()

    # finally save the results
    results_file = output_dir.joinpath(
        f"Model results - {modelfile.stem}_wards{wards}_capacities{capacities}_"
        f"service[{service_name}]_waiting[{waiting}].json")

    with open(results_file, "w") as f:
        json.dump(results, f)
