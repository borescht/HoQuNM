"""Basic functionalities and default values for data preprocessing, analysis
and modelling."""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Tuple, Union

import pandas as pd
from dataenforce import Dataset, validate

pd.options.mode.use_inf_as_na = True

GLOB_BEGIN = "AUFNDAT"
GLOB_END = "ENTLDAT"
BEGIN = "INTE_BEGINN"
END = "INTE_ENDE"
FA_BEGIN = "Aufn_FA"
FA_CURRENT = "Akt_FA"
STATUS = "Fallstatus"
DIAGNR = "DIAGNR"
ICPM = "ICPM"
OP_DATE = "OPDatum"
URGENCY = "Dringlichkeit"
REC_TYPE = "Aufnahmeart"
PATIENT = "Fallnummer"
SEX = "Geschlecht"
BIRTH = "Geburtsdatum"
AGE = "Alter"
FLOW = "sta_verlegung"

# mine
PRE_WARD = "Preceding"
CURRENT_WARD = "Current"
POST_WARD = "Next"
INTER_ARRIVAL = "Zwischenankunft"
SERVICE = "Pflegezeit"
EXTERNAL = "External"
INTERNAL = "Internal"
PRE_CLASS = "Pre_Typ"
CURRENT_CLASS = "Typ"
POST_CLASS = "Post_Typ"
SIGN = "Sign"
TIMESTAMP = "Zeitstempel"
OCCUPANCY = "Belegung"
TIME = "Dauer"
MIN = "Minimum"
MAX = "Maximum"
MEAN = "Mean"
STD = "Std"
WEEK = "Week"
WEEKDAY = "Weekday"

DATECOLUMNS = [BEGIN, END, GLOB_BEGIN, GLOB_END]

HEAD_ORG = [
    GLOB_BEGIN, PATIENT, SEX, BIRTH, FA_BEGIN, FA_CURRENT, FLOW, STATUS, BEGIN,
    END, DIAGNR, ICPM, OP_DATE, URGENCY, REC_TYPE, GLOB_END
]
HEAD = [
    GLOB_BEGIN, PATIENT, SEX, BIRTH, FA_BEGIN, FA_CURRENT, PRE_WARD,
    CURRENT_WARD, POST_WARD, PRE_CLASS, CURRENT_CLASS, STATUS, BEGIN, END,
    DIAGNR, ICPM, OP_DATE, URGENCY, REC_TYPE, GLOB_END, AGE
]

MIN_HEAD = [PATIENT, PRE_WARD, CURRENT_WARD, POST_WARD, BEGIN, END]
Min_Head_Name = Dataset[PATIENT, PRE_WARD, CURRENT_WARD, POST_WARD, BEGIN, END,
                        ...]

CART_COLUMNS = [SEX, AGE, STATUS, WEEK]
CAT_COLUMNS = [SEX, STATUS]
CART_COLUMNS_TRANSLATION = {
    SEX: "Sex",
    AGE: "Age",
    STATUS: "Status",
    WEEK: "Week"
}

OUTPUT_DIR = Path(os.path.expanduser("~")).resolve().joinpath("hoqunm_files")
if not OUTPUT_DIR.is_dir():
    OUTPUT_DIR.mkdir()

OUTPUT_DIR_DATA = OUTPUT_DIR.joinpath("data")
if not OUTPUT_DIR_DATA.is_dir():
    OUTPUT_DIR_DATA.mkdir()

OUTPUT_DIR_DATA_ANALYSIS = OUTPUT_DIR.joinpath("data_analysis")
if not OUTPUT_DIR_DATA_ANALYSIS.is_dir():
    OUTPUT_DIR_DATA_ANALYSIS.mkdir()

OUTPUT_DIR_MODEL_BUILD = OUTPUT_DIR.joinpath("model_build")
if not OUTPUT_DIR_MODEL_BUILD.is_dir():
    OUTPUT_DIR_MODEL_BUILD.mkdir()

OUTPUT_DIR_SIMULATION = OUTPUT_DIR.joinpath("simulation")
if not OUTPUT_DIR_SIMULATION.is_dir():
    OUTPUT_DIR_SIMULATION.mkdir()

OUTPUT_DIR_ASSESS_CAPACITIES = OUTPUT_DIR_SIMULATION.joinpath(
    "assess_capacities")
if not OUTPUT_DIR_ASSESS_CAPACITIES.is_dir():
    OUTPUT_DIR_ASSESS_CAPACITIES.mkdir()

OUTPUT_DIR_MODEL_VARIANTS = OUTPUT_DIR_SIMULATION.joinpath("model_variants")
if not OUTPUT_DIR_MODEL_VARIANTS.is_dir():
    OUTPUT_DIR_MODEL_VARIANTS.mkdir()

OUTPUT_DIR_OPTIMISATION = OUTPUT_DIR.joinpath("optimisation")
if not OUTPUT_DIR_OPTIMISATION.is_dir():
    OUTPUT_DIR_OPTIMISATION.mkdir()

EXAMPLE_FILEPATH_PREPROCESS = OUTPUT_DIR_DATA.joinpath(
    "preprocess_data_specs.json")

EXAMPLE_FILEPATH_ANALYSE = OUTPUT_DIR_DATA.joinpath("analyse_data_specs.json")

EXAMPLE_FILEPATH_BUILD = OUTPUT_DIR_DATA.joinpath("build_model_specs.json")

EXAMPLE_FILEPATH_VARIANTS = OUTPUT_DIR_SIMULATION.joinpath(
    "analyse_model_variants_specs.json")

EXAMPLE_FILEPATH_ASSESS = OUTPUT_DIR_SIMULATION.joinpath(
    "assess_capacities_specs.json")

EXAMPLE_FILEPATH_OPTIMISATION_SIMULATION = OUTPUT_DIR_OPTIMISATION.joinpath(
    "optimisation_specs_simulation.json")

EXAMPLE_FILEPATH_OPTIMISATION_COMPUTATION = OUTPUT_DIR_OPTIMISATION.joinpath(
    "optimisation_specs_computation.json")

EXAMPLE_MODEL_NO_CART = Path(__file__).parent.parent.parent.resolve().joinpath(
    "example_models", "HospitalModel - "
    "cart[False].json")

EXAMPLE_MODEL_CART = Path(__file__).parent.parent.parent.resolve().joinpath(
    "example_models", "HospitalModel - "
    "cart[True].json")


def copy_json() -> None:
    """Copy the json specifications for cli into OUTPUT_DIR if not already
    existent."""
    example_paths = [
        EXAMPLE_FILEPATH_PREPROCESS, EXAMPLE_FILEPATH_ANALYSE,
        EXAMPLE_FILEPATH_BUILD, EXAMPLE_FILEPATH_VARIANTS,
        EXAMPLE_FILEPATH_ASSESS, EXAMPLE_FILEPATH_OPTIMISATION_SIMULATION,
        EXAMPLE_FILEPATH_OPTIMISATION_COMPUTATION
    ]
    for file_path in example_paths:
        if not file_path.is_file():
            with open(
                    Path(__file__).parent.parent.parent.resolve().joinpath(
                        "example_specs", file_path.name), "r") as f:
                data_specs = json.load(f)
            with open(file_path, "w") as f:
                json.dump(data_specs, f)


def split_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, datetime, datetime]:
    """The first row of the data could contain the start- and enddate.

    Therefore, the data should be splitted.

    :return: the data without the first row,
    the dates extracted from the first row
    """
    startdate = pd.to_datetime(data.iloc[0][GLOB_BEGIN])
    enddate = pd.to_datetime(data.iloc[0][GLOB_END])

    startdate = datetime(startdate.year, startdate.month, startdate.day)
    enddate = datetime(enddate.year, enddate.month, enddate.day)
    data_ = data.iloc[1:, :]
    data_.loc[:, DATECOLUMNS] = data_[DATECOLUMNS].astype("float")

    return data_, startdate, enddate


def get_data(filepath: Path,
             sep: str = ";",
             **kwargs: Any) -> Tuple[pd.DataFrame, datetime, datetime]:
    """read the data from a given excel file.

    :param filepath: the excel file, where the data is contained
    :param sep: the seperator with which the entries of the data file are seperated
    """

    data = pd.read_csv(filepath, sep=sep, **kwargs)
    return split_data(data)


@validate
def drop_week_arrival(data: Dataset[WEEKDAY, ...],
                      week: bool = True) -> Dataset:
    """Drop arrivals on week/weekends.

    :param data: Data to consider.
    :param week: If week should be kept (True) or dropped (False).

    :return: Adjusted data.
    """

    week_data = data.iloc[:-1][WEEKDAY]
    week_data.index = data.index[1:]
    data["Pre_Weekday"] = week_data
    if week:
        drop_qry = or_query(column_query(WEEKDAY, 5, ">="),
                            WEEKDAY + "<" + "Pre_Weekday")
    else:
        drop_qry = or_query(
            and_query(column_query(WEEKDAY, 4, "<="),
                      column_query(WEEKDAY, 0, ">=")),
            column_query("Pre_Weekday", 4, "<="),
            column_query("Arrival", 3, ">"))

    data = data.drop(index=data.query(drop_qry).index)

    return data


def column_query(column: Union[str], value: Any, operator: str = "=="):
    """generate a query aginsta agiven column.

    :param column: the column to query against
    :param value: the value to search for
    :param operator: smaller, equal,...
    """
    if isinstance(value, str):
        qry = column + operator + "'" + value + "'"
    else:
        qry = column + operator + str(value)

    return qry


def and_query(*qrys):
    """create a and query from a given set of querys.

    :param qrys: the respective queries
    """
    return "(" + "&".join([qry for qry in qrys if qry]) + ")"


def or_query(*qrys):
    """create a and query from a given set of querys.

    :param qrys: the respective queries
    """
    return "(" + "|".join([qry for qry in qrys if qry]) + ")"


def make_week(data: pd.DataFrame, begin: int, column: Any = BEGIN):
    """Make a boolean week and a numbered weekday column in the dataframe based
    on the given column.

    :param data: the dataframe
    :param begin: the day at 0 time (0 Mo, 1 Tue,...)
    :param column: the column to take
    """
    data.loc[:, WEEK] = ((data.loc[:, column] + begin) % 7) // 5
    data.loc[:, WEEKDAY] = (data.loc[:, column] + begin) % 7 // 1

    return data
