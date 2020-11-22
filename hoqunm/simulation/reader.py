"""Read all important model information from an excel file.

The logic therefore is implemented here.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
import scipy.stats

from hoqunm.simulation.hospital import HospitalSpecs
from hoqunm.utils.distributions import HyperDistribution

ARRIVAL_SHEET = "Arrivals"
NURSING_SHEET = "Nursings"
WARD_SHEET = "Wards"
ROUTING_SHEET = "Routings"
WAITING_MAP_SHEET = "Waiting Maps"

WARD_CAPACITY_ROW = "Ward Capacity"
WARD_HOLDING_ROW = "Ward Holding"


def get_arrival(xls: pd.ExcelFile) -> np.ndarray:
    """Retrieve and convert the arrival data from excel file.

    :param xls: The excel file.

    :return: The arrival distributions.
    """

    df_arrival = pd.read_excel(xls, ARRIVAL_SHEET, index_col=0, dtype="str")

    # get the arrival distributions from ARRIVAL_SHEET
    arrival = distribution(df_arrival)

    return arrival


def get_service(xls: pd.ExcelFile) -> np.ndarray:
    """Retrieve and convert the service data from excel file.

    :param xls: The excel file.

    :return: The service distributions.
    """

    df_service = pd.read_excel(xls, NURSING_SHEET, index_col=0, dtype="str")

    # get the service distributions from SERVICE_SHEET
    service = distribution(df_service)

    return service


def get_routing(xls: pd.ExcelFile, st: int, cl: int) -> np.ndarray:
    """Retrieve the routing data from excel file.

    :param xls: The excel file.
    :param st: The number of wards.
    :param cl: The number of classes.

    :return: The routing matrix.
    """

    df_routing = pd.read_excel(xls, ROUTING_SHEET, index_col=0).astype("float")

    # build the routing matrix from ROUTING_SHEET
    routing = np.concatenate((np.array(df_routing), np.zeros(
        (st * cl, cl - 1))),
                             axis=1).reshape((st, cl, st + 1, cl))

    return routing


def get_capacities(xls: pd.ExcelFile) -> np.ndarray:
    """Retireve the capacity data from excel file.

    :param xls: The excel file.

    :return: The capacities.
    """
    df_ward_info = pd.read_excel(xls, WARD_SHEET, index_col=0).astype("str")

    # get the different information from STATTION_SHEET
    capacities = np.array(
        df_ward_info.loc[WARD_CAPACITY_ROW]).astype("float").astype("int")

    return capacities


def get_holdings(xls: pd.ExcelFile) -> np.ndarray:
    """Retireve the holdings data from excel file.

    :param xls: The excel file.

    :return: The capacities.
    """
    df_ward_info = pd.read_excel(xls, WARD_SHEET, index_col=0).astype("str")

    holdings = np.array(df_ward_info.loc[WARD_HOLDING_ROW]).astype(
        "float").astype("int").astype("bool")

    return holdings


def get_waitings(xls: pd.ExcelFile, ward_map) -> np.ndarray:
    """Retireve the waiting data from excel file.

    :param xls: The excel file.
    :param ward_map: The ward to index map.

    :return: The waiitngs.
    """

    ward_map_inv = {ind: i for i, ind in ward_map.items()}

    df_waitings = pd.read_excel(xls, WAITING_MAP_SHEET, index_col=0)

    # get the waitings from WAITING_MAP_SHEET
    waitings: Dict[int, List[int]] = {i: [] for i in ward_map}

    for _, row in df_waitings.iterrows():
        for ward in df_waitings.columns:
            if row[ward] != "None":
                waitings[ward_map_inv[ward]].append(ward_map_inv[row[ward]])

    return waitings


def get_mappings(xls: pd.ExcelFile) -> Tuple[Dict[int, str], Dict[int, str]]:
    """Retrieve the mapping information from the excel file. This is given
    implicit and can be generated through the arrival data in the corresponding
    excel sheet.

    :param xls: The excel file.

    :return: A ward and a class map.
    """

    df_arrival = pd.read_excel(xls, ARRIVAL_SHEET, index_col=0, dtype="str")

    # get the mapping from index to ward name and its inverse
    ward_map = dict(enumerate(df_arrival.index))

    # get the mapping from index to class name, inverse not needed
    class_map = dict(enumerate(df_arrival.columns))

    return ward_map, class_map


def get_data(excel_path: Path) -> HospitalSpecs:
    """read the hospital specifications and all related information from given
    excel file. The sheets must have explicit names!

    :param excel_path: The excel path of the file.

    :return: Created HospitalSpecs class.
    """

    xls = pd.ExcelFile(excel_path)

    capacities = get_capacities(xls)
    arrival = get_arrival(xls)
    service = get_service(xls)
    ward_map, class_map = get_mappings(xls)
    routing = get_routing(xls, len(ward_map), len(class_map))
    waitings = get_waitings(xls, ward_map)
    holdings = get_holdings(xls)

    # finished. return everything packed in Hospital_specs class
    return HospitalSpecs(capacities=capacities,
                         arrival=arrival,
                         service=service,
                         routing=routing,
                         waitings=waitings,
                         holdings=holdings,
                         ward_map=ward_map,
                         class_map=class_map)


# pylint: disable=eval-used
def distribution(df: Union[pd.DataFrame, np.ndarray]) -> np.ndarray:
    """Translate string information for distributions to better readable
    information.

    :param df: The dataframe containing the string information.
    """
    out = np.zeros_like(df, dtype="O")
    for index, val in np.ndenumerate(df):
        if val != "0":
            # TODO: fix eval!
            val = eval(val)
            distribution_ = val.pop("distributions")
            out[index] = HyperDistribution(getattr(scipy.stats, distribution_),
                                           **val)

    return out
