"""Command line tools for data handling."""

import datetime
import json
from pathlib import Path

import click

from hoqunm.data_tools.analysis import CartSpecs, advanced_analysis
from hoqunm.data_tools.base import (EXAMPLE_FILEPATH_ANALYSE,
                                    EXAMPLE_FILEPATH_BUILD,
                                    EXAMPLE_FILEPATH_PREPROCESS,
                                    OUTPUT_DIR_DATA, OUTPUT_DIR_DATA_ANALYSIS,
                                    OUTPUT_DIR_MODEL_BUILD)
from hoqunm.data_tools.modelling import make_hospital_model
from hoqunm.data_tools.preprocessing import adjust_data, preprocess
from hoqunm.utils.distributions import HypererlangSpecs

SAVEFILE_ANALYSIS = OUTPUT_DIR_DATA.joinpath("preprocessed_data.csv")


@click.command()
@click.option("--specsfile",
              "-s",
              type=click.Path(exists=True),
              default=str(EXAMPLE_FILEPATH_PREPROCESS),
              required=True,
              help="Filepath to specifications for model building. "
              f"Default can be found in {EXAMPLE_FILEPATH_PREPROCESS}.")
def preprocess_data(specsfile: str):
    """Clean data obtained from hospital."""
    with open(Path(specsfile), "r") as f:
        specs = json.load(f)

    files = [Path(file) for file in specs["files"]]
    if specs.get("filepath", None) is not None:
        files = [Path(specs["filepath"]).joinpath(file) for file in files]
    if any(not file.is_file() for file in files):
        raise FileNotFoundError(files)

    y, m, d = specs["startdate"].split("-")
    startdate = datetime.datetime(int(y), int(m), int(d))

    y, m, d = specs["enddate"].split("-")
    enddate = datetime.datetime(int(y), int(m), int(d))

    ward_key_map = specs["ward_key_map"]
    internal_prefix = specs["internal_prefix"]
    urgency_split_string = specs["urgency_split_string"]
    birth_format = specs["birth_format"]
    flow_split_string = specs["flow_split_string"]
    timestamp_format = specs["timestamp_format"]

    save_file = specs[
        "save_file"] if specs["save_file"] is not None else SAVEFILE_ANALYSIS

    if not Path(save_file).parent.is_dir():
        Path(save_file).parent.mkdir()

    preproessor = preprocess(filepath=files,
                             startdate=startdate,
                             enddate=enddate,
                             ward_key_map=ward_key_map,
                             internal_prefix=internal_prefix,
                             urgency_split_string=urgency_split_string,
                             birth_format=birth_format,
                             flow_split_string=flow_split_string,
                             timestamp_format=timestamp_format)

    print("Save processed data.")
    preproessor.write(save_file, index=False)

    print(f"Finished successfully. You can find your file in {save_file}.")


@click.command()
@click.option("--specsfile",
              "-s",
              type=click.Path(exists=True),
              default=str(EXAMPLE_FILEPATH_ANALYSE),
              required=True,
              help="Filepath to specifications for model building. "
              f"Default can be found in {EXAMPLE_FILEPATH_ANALYSE}.")
def analyse_data(specsfile: str):
    """Analyse the data very briefly w/o building of the model."""
    with open(specsfile, "r") as f:
        specs = json.load(f)

    datafile = Path(specs["datafile"]
                    ) if specs["datafile"] is not None else SAVEFILE_ANALYSIS
    if not datafile.is_file():
        raise FileNotFoundError(datafile)

    wards = specs["wards"]
    capacities = specs["capacities"]

    y, m, d = specs["startdate"].split("-")
    startdate = datetime.datetime(int(y), int(m), int(d))

    y, m, d = specs["enddate"].split("-")
    enddate = datetime.datetime(int(y), int(m), int(d))

    cart_specs = CartSpecs.load_dict(specs["cart"])

    window_size = specs["window_size"]

    output_dir = Path(
        specs["output_dir"]
    ) if specs["output_dir"] is not None else OUTPUT_DIR_DATA_ANALYSIS
    if not output_dir.is_dir():
        output_dir.mkdir()

    adjust_pacu_occupancy = specs["adjust_pacu_occupancy"]

    print("Start analysing the data.")

    preprocessor = adjust_data(filepath=datafile,
                               wards=wards,
                               startdate=startdate,
                               enddate=enddate,
                               keep_internal=True)

    advanced_analysis(preprocessor=preprocessor,
                      wards=wards,
                      capacities=capacities,
                      adjust_pacu_occupancy=adjust_pacu_occupancy,
                      cart_specs=cart_specs,
                      window_size=window_size,
                      output_dir=output_dir)

    print(
        f"Finished analysis successfully. You can find all the files in {output_dir}."
    )


@click.command()
@click.option("--specsfile",
              "-s",
              type=click.Path(exists=True),
              default=str(EXAMPLE_FILEPATH_BUILD),
              required=True,
              help="Filepath to specifications for model building. "
              f"Default can be found in {EXAMPLE_FILEPATH_BUILD}.")
def build_model(specsfile: str):
    """Build the model under different variants.

    This includes analysis as it is necessary.
    """
    with open(specsfile, "r") as f:
        specs = json.load(f)

    datafile = Path(specs["datafile"]
                    ) if specs["datafile"] is not None else SAVEFILE_ANALYSIS
    if not datafile.is_file():
        raise FileNotFoundError(datafile)

    wards = specs["wards"]
    capacities = specs["capacities"]

    y, m, d = specs["startdate"].split("-")
    startdate = datetime.datetime(int(y), int(m), int(d))

    y, m, d = specs["enddate"].split("-")
    enddate = datetime.datetime(int(y), int(m), int(d))

    cart_specs = CartSpecs.load_dict(specs["cart"])
    hypererlang_specs = HypererlangSpecs.load_dict(specs["hypererlang"])
    adjust_pacu_occupancy = specs["adjust_pacu_occupancy"]

    output_dir = Path(
        specs["output_dir"]
    ) if specs["output_dir"] is not None else OUTPUT_DIR_MODEL_BUILD
    if not output_dir.is_dir():
        output_dir.mkdir()

    print("Start building the hospital model.")

    make_hospital_model(filepath=datafile,
                        wards=wards,
                        capacities=capacities,
                        startdate=startdate,
                        enddate=enddate,
                        cart_specs=cart_specs,
                        hypererlang_specs=hypererlang_specs,
                        adjust_pacu_occupancy=adjust_pacu_occupancy,
                        output_dir=output_dir)

    print(
        f"Finished successfully. You can find all the files in {output_dir}.")
