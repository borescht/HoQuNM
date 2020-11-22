"""Provide methodologies for preprocessing the data."""

import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import pandas as pd

from hoqunm.data_tools.base import (
    AGE, BEGIN, BIRTH, CURRENT_CLASS, CURRENT_WARD, DATECOLUMNS, DIAGNR, END,
    EXTERNAL, FA_BEGIN, FLOW, GLOB_BEGIN, GLOB_END, INTERNAL, PATIENT,
    POST_CLASS, POST_WARD, PRE_CLASS, PRE_WARD, SEX, URGENCY, and_query,
    column_query, or_query, split_data)
from hoqunm.utils.utils import get_logger

pd.set_option('mode.chained_assignment', None)


class Preprocessor:
    """A class holding different functionalities for preprocessing the data
    from the hospital.

    The aim is to preprocess the data in so far as to give it to class
    Data. To do it clean i would imagine sth like a validation function,
    which validates that the data is in the correct format.

    :param filepath: The file path to the data (as csv/xls/...).
    :param sep: The separator with which columns are separated.
    :parma **kwargs: Additional arguments used by pandas.read_csv.
    """
    def __init__(self,
                 filepath: Path = Path(),
                 sep: str = ";",
                 startdate: datetime = datetime(2018, 1, 1),
                 enddate: datetime = datetime(2020, 1, 1),
                 datescale: timedelta = timedelta(1),
                 logger: Optional[logging.Logger] = None,
                 **kwargs: Any) -> None:
        self.data = pd.read_csv(filepath, sep=sep, **kwargs)
        self.startdate = startdate
        self.enddate = enddate
        self.datescale = datescale
        self.data_backup = self.data.copy()
        self.logger = logger if logger is not None else get_logger(
            "Preprocessor", filepath.parent.joinpath("preprocessor.log"))

    def split_data(self) -> None:
        """The first row of the data could contain the start- and enddate.

        Therefore, the data should be splitted.
        """
        self.backup()
        self.data, self.startdate, self.enddate = split_data(self.data)

    def backup(self) -> None:
        """Save the current state of self.data in self.data_backup."""
        self.data_backup = self.data.copy()

    def reset(self) -> None:
        """Replace self.data with self.data_backup (last data state)."""
        self.data = self.data_backup.copy()

    def add(self,
            filepath: Path = Path(),
            sep: str = ";",
            **kwargs: Any) -> None:
        """Add new data to existing data.

        :param filepath: The file path to the data (as csv/xls/...).
        :param sep: The separator with which columns are separated.
        :parma **kwargs: Additional arguments used by pandas.read_csv.
        """
        data = pd.read_csv(filepath, sep=sep, **kwargs)

        assert all(data.columns == self.data.columns)

        self.data = self.data.append(data, ignore_index=True)

    def write(self, filepath: Path, sep: str = ";", **kwargs: Any) -> None:
        """Save the current state in a csv file.

        :param filepath: The file path to the data (as csv/xls/...).
        :param sep: The separator with which columns are separated.
        :parma **kwargs: Additional arguments used by pandas.to_csv.
        """
        data = pd.DataFrame(columns=self.data.columns)
        time_ser = pd.Series(index=self.data.columns)
        time_ser[GLOB_BEGIN] = self.startdate
        time_ser[GLOB_END] = self.enddate
        data = data.append(time_ser, ignore_index=True)
        data = data.append(self.data, ignore_index=True)
        data.to_csv(filepath, sep=sep, **kwargs)

    # pylint: disable=redefined-builtin
    def datetimes_to_float(self,
                           format: str = "%d.%m.%Y %H:%M",
                           startdate: Optional[datetime] = None,
                           scale: Optional[timedelta] = None,
                           columns: Optional[List[str]] = None) -> None:
        """Make date string to dates and the to floats. Set errors to NaN. Set
        all datetimes before start of analysis to negative values.

        :param format: The format in which the datetimes are currently given.
        :param startdate: A date indicating, when analysis should start.
        :param scale: T timedelta indicating at which level one should scale. Days seem reasonable.
        :param columns: The columns under consideration.
        """

        if startdate is None:
            startdate = self.startdate
        if scale is None:
            scale = self.datescale
        if columns is None:
            columns = DATECOLUMNS

        datecolumns = pd.DataFrame({
            datecolumn: pd.to_datetime(self.data[datecolumn],
                                       format=format,
                                       errors="coerce")
            for datecolumn in columns
        })
        naindex = [
            datecolumns.loc[:, datecolumn].isnull() for datecolumn in columns
        ]
        datecolumns = datecolumns.fillna(
            datetime.max.replace(year=2100, second=0, microsecond=0))
        self.data[columns] = (datecolumns - startdate) / scale
        self.data[columns] = self.data[columns].astype("float")
        for i, column in enumerate(columns):
            self.data.loc[naindex[i], column] = float("inf")

    def make_flow(self, split_str: str = " \\f:9919\\Þ\\f:-\\ ") -> None:
        """The csv file is not well formatted. Split the column FLOW into
        PRE_WARD and CURRENT_WARD.

        :param split_str: The string separating PRE_WARD from CURRENT_WARD.
        """

        self.backup()

        self.data.loc[:, PRE_WARD] = float("NaN")
        self.data.loc[:, CURRENT_WARD] = float("NaN")

        for i, row in self.data.iterrows():
            split = row[FLOW].split(split_str)
            self.data.loc[i, PRE_WARD] = split[0]
            self.data.loc[i, CURRENT_WARD] = split[-1]

        self.data = self.data.drop(columns=FLOW)

    def replace_ward_keys(self,
                          ward_map: Optional[Dict[str, List[str]]] = None,
                          internal_prefix: Optional[List[str]] = None) -> None:
        """The csv file has more information than needed.

        Convert unimportant wards (not given in ward_map) to EXTERNAL ward
        and map the remaining wards as given via ward_map.

        :param ward_map: Key is the ward name with its values. All names for it in the csv.
        :param internal_prefix: List of internal prefixs to consider.
        """

        if ward_map is not None:

            self.backup()

            for ward, keys in ward_map.items():
                self.data = self.data.replace(keys, ward)

            all_wards = list(self.data.loc[:, CURRENT_WARD].unique()) + list(
                self.data.loc[:, PRE_WARD].unique())

            internal_wards = [
                ward for ward in all_wards
                if ward not in ward_map and ((internal_prefix is None) or any(
                    ward.startswith(internal_prefix_)
                    for internal_prefix_ in internal_prefix))
            ]

            external_wards = [
                ward for ward in all_wards
                if ward not in list(ward_map) and ward not in internal_wards
            ]

            self.data = self.data.replace(internal_wards, INTERNAL)

            self.data = self.data.replace(external_wards, EXTERNAL)

        else:
            self.logger.warning(
                "No ward map given. No ward keys have been replace.")

    def make_urgency(self, split_str: str = "\\f:9919\\Þ\\f:-\\ ") -> None:
        """Get the main information from URGENCY (N0,...,N5).

        :split_st: String to split at.
        """
        self.backup()

        if URGENCY in self.data.columns:
            for i, row in self.data.iterrows():
                split = row[URGENCY].split(split_str)
                if len(split[1]) > 1:
                    try:
                        self.data.loc[i, URGENCY] = int(split[1][1])
                    except ValueError:
                        self.data.loc[i, URGENCY] = float("NaN")
                else:
                    self.data.loc[i, URGENCY] = float("NaN")

    # pylint: disable=redefined-builtin
    def make_age(self,
                 reference_date: datetime = datetime.today(),
                 format: str = "%d.%m.%Y") -> None:
        """Make an age value from the BIRTH column given a reference date.

        This could be helfpul for CART and some other analysis.

        :param reference_date: The data at which the age will be interpreted.
        :param format: The date format of BIRTH.
        """
        self.backup()
        if BIRTH in self.data.columns:
            scale = timedelta(365)

            self.data.loc[:, AGE] = self.data.loc[:, BIRTH].copy()
            self.datetimes_to_float(format=format,
                                    startdate=reference_date,
                                    scale=scale,
                                    columns=[AGE])
            self.data.loc[:, AGE] *= -1
            self.data.loc[:, AGE] = self.data.loc[:, AGE].astype("int")

    def clean_empty(self):
        """The data has empty END entries which can be dropped."""
        self.backup()
        qry = column_query(END, " ")
        empty_index = self.data.query(qry).index
        self.data = self.data.drop(index=empty_index)

    def _clean_patient_data(self, patient_data: pd.DataFrame) -> pd.DataFrame:
        """Clean all the data for a given patient.

        This should refelct the special needs for the data sheets
        obtained from the hospital.

        :param patient_data: The sorted entries for one patient.

        :return: Cleaned data.
        """
        df = pd.DataFrame(columns=patient_data.columns)

        rowi = patient_data.iloc[0, :].copy()
        if rowi.loc[PRE_WARD] == rowi.loc[CURRENT_WARD]:
            rowi.loc[PRE_WARD] = EXTERNAL

        for _, rowj in patient_data.iloc[1:, :].iterrows():
            if rowi.loc[CURRENT_WARD] in [EXTERNAL, INTERNAL]:
                # entry is not interesting, go on
                rowi = rowj.copy()
            elif rowi.loc[END] == rowj.loc[BEGIN] and rowi.loc[
                    CURRENT_WARD] == rowj.loc[CURRENT_WARD]:
                # enough evidence for a multiple entry, change the END of the last row
                rowi.loc[END] = rowj.loc[END]
            elif rowi.loc[END] == rowj.loc[BEGIN] and rowi.loc[
                    CURRENT_WARD] == rowj.loc[PRE_WARD]:
                # found a new row, so save rowi and make new
                rowi.loc[POST_WARD] = rowj.loc[CURRENT_WARD]
                df = df.append(rowi)
                rowi = rowj.copy()
            elif rowj.loc[PRE_WARD] in [EXTERNAL, INTERNAL]:
                # maybe the patient visited again, so drop the current row and start sth new
                rowi.loc[POST_WARD] = rowj.loc[PRE_WARD]
                df = df.append(rowi)
                rowi = rowj.copy()
            else:
                # there are some errors in the data set concerning multiple
                # entries for different ICPM.
                # just go to the next row, hoping not to mess things up
                self.logger.warning("Warning. Something else happened.")

        if rowi.loc[CURRENT_WARD] not in [EXTERNAL, INTERNAL]:
            if not rowi.isna().loc[END]:
                rowi.loc[POST_WARD] = EXTERNAL
            df = df.append(rowi)

        return df

    def clean_data(self) -> None:
        """Clean the data from multiple entries regarding the same stay.

        This should refelct the special needs for the data sheets
        obtained from the hospital.
        """

        self.backup()
        data = self.data.copy()

        data.loc[:, PATIENT] = float("NaN")
        data.loc[:, POST_WARD] = float("NaN")

        df = pd.DataFrame(columns=data.columns)

        i = 0

        while i < data.shape[0]:
            rowi = data.iloc[i, :]

            # query parameters should reflect exactly one patient
            qry = and_query(column_query(BIRTH, rowi.loc[BIRTH]),
                            column_query(SEX, rowi.loc[SEX]),
                            column_query(GLOB_BEGIN, rowi.loc[GLOB_BEGIN]),
                            column_query(GLOB_END, rowi.loc[GLOB_END]),
                            column_query(FA_BEGIN, rowi.loc[FA_BEGIN]),
                            column_query(DIAGNR, rowi.loc[DIAGNR]))

            patient_data = data.query(qry)

            patient_data.loc[:, PATIENT] = i + 1

            data = data.drop(index=patient_data.iloc[1:, :].index)

            patient_data = self._clean_patient_data(patient_data)

            df = df.append(patient_data)

            i += 1

        # now data is clean, work with it, self.data is still saved in csv!
        # it would be cleaner to sort by date, but this can be done later too
        # -> since time is not yet formatted, sorting by date is no good idea!
        self.data = df.sort_index()

    def clean_data_gen(
        self
    ) -> Generator[Tuple[int, pd.DataFrame, pd.DataFrame], None, None]:
        """A generator which helps understand the cleaning process of
        clean_data.

        :yields: one after another all entries associated with a
        specific patient and its DataFrame obtained through cleaning
        those with clean_patient_data.
        """

        data = self.data.copy()

        data.loc[:, PATIENT] = float("NaN")
        data.loc[:, POST_WARD] = float("NaN")

        i = 0

        while i < data.shape[0]:
            rowi = data.iloc[i]

            # query parameters should reflect exactly one patient
            qry = and_query(column_query(BIRTH, rowi[BIRTH]),
                            column_query(SEX, rowi[SEX]),
                            column_query(GLOB_BEGIN, rowi[GLOB_BEGIN]),
                            column_query(GLOB_END, rowi[GLOB_END]),
                            column_query(FA_BEGIN, rowi[FA_BEGIN]),
                            column_query(DIAGNR, rowi[DIAGNR]))

            patient_data = data.query(qry)
            patient_data_ = self._clean_patient_data(patient_data)
            yield i, patient_data, patient_data_

            data = data.drop(index=patient_data.iloc[1:].index)

            i += 1

    def prune_begin(self) -> None:
        """Drop all data entries for patiens who left before our observation
        begins.

        This are the entries with END <= 0.
        """
        self.backup()

        qry = column_query(END, 0, "<=")
        junk_data = self.data.query(qry)
        self.data = self.data.drop(index=junk_data.index)

    def prune_end(self, enddate: Optional[datetime] = None) -> None:
        """Drop all data entries for patiens who came after our observation
        end.

        This are the entries with BEGIN > end.

        :param enddate: The time when our observation ends.
        """

        self.backup()

        if not enddate:
            end = self.data[GLOB_BEGIN].max()
            self.enddate = self.startdate + self.datescale * end
        else:
            end = (enddate - self.startdate) / self.datescale
            self.enddate = enddate

        qry = column_query(BEGIN, end, "<=")
        self.data = self.data.query(qry)

        qry = column_query(END, end, ">")
        self.data.loc[self.data.query(qry).index,
                      [END, POST_WARD]] = float("NaN")

    def alter_begin(self, startdate: datetime) -> None:
        """Change the beginning date to the new startdate.

        Only possible if self.startdate < startdate < elf.enddate.

        :param startdate: The new startdate.
        """
        if startdate < self.startdate:
            raise ValueError(
                "The startdate should not be smaller then the current startdate."
            )

        if startdate > self.enddate:
            raise ValueError(
                "The startdate should not be greater then the current enddate."
            )

        change = (startdate - self.startdate) / self.datescale

        self.data.loc[:, DATECOLUMNS] -= change
        qry = column_query(END, 0, "<")
        drop_index = self.data.query(qry).index
        self.data = self.data.drop(index=drop_index)
        self.startdate = startdate

    def alter_end(self, enddate: datetime) -> None:
        """Change the end to a new end given in enddate.

        Only possible if self.startdate < enddate < self.enddate.

        :param enddate: The new enddate.
        """
        if enddate > self.enddate:
            raise ValueError(
                "The enddate should not be greater then the current enddate.")

        if enddate < self.startdate:
            raise ValueError(
                "The enddate should not be smaller then the current startdate."
            )

        endfloat = (enddate - self.startdate) / self.datescale
        qry = column_query(BEGIN, endfloat, ">")
        drop_index = self.data.query(qry).index
        self.data = self.data.drop(index=drop_index)

        qry = column_query(END, endfloat, ">")
        nan_index = self.data.query(qry).index
        self.data.loc[nan_index, END] = float("NaN")
        self.enddate = enddate

    def drop_ward(self, *ward: str) -> None:
        """Drop wards if not needed anymore.

        :param ward: The ward(s) to drop.
        """
        qry = or_query(*[column_query(CURRENT_WARD, ward_) for ward_ in ward])
        drop_index = self.data.query(qry).index
        self.data = self.data.drop(index=drop_index)

        self.data = self.data.replace(ward, INTERNAL)

    def restrict_ward(self, wards: List[str]) -> None:
        """Restrict to the given wards, dropping the rest.

        :param wards: The wards to keep only.
        """
        qry = and_query(
            *[column_query(CURRENT_WARD, ward_, "!=") for ward_ in wards])
        drop_index = self.data.query(qry).index
        self.data = self.data.drop(index=drop_index)

        qry = and_query(
            column_query(POST_WARD, EXTERNAL, "!="),
            *[column_query(POST_WARD, ward_, "!=") for ward_ in wards])
        replace_index = self.data.query(qry).index
        self.data.loc[replace_index, POST_WARD] = INTERNAL

        qry = and_query(
            column_query(PRE_WARD, EXTERNAL, "!="),
            *[column_query(PRE_WARD, ward_, "!=") for ward_ in wards])
        replace_index = self.data.query(qry).index
        self.data.loc[replace_index, PRE_WARD] = INTERNAL

    def drop_internal(self) -> None:
        """Drop internal wards."""
        self.data = self.data.replace(INTERNAL, EXTERNAL)


def preprocess(filepath: List[Path],
               startdate: datetime,
               enddate: datetime,
               ward_key_map: Dict[str, List[str]],
               internal_prefix: List[str],
               urgency_split_string: str,
               birth_format: str,
               flow_split_string: str,
               timestamp_format: str,
               logger: Optional[logging.Logger] = None) -> Preprocessor:
    """Preprocess given excel file.

    :param filepath: Path to file.
    :param startdate: Startdate to take.
    :param enddate: Enddate to take.
    :param ward_key_map: Map for hospital wards to their respective keys.
    Exp.: Sometimes, multiple keys are referring to one ward. Map them.
    :param internal_prefix: Prefix associated with internal wards.
    Exp.: All wards in one hospital could be found by a common prefix.
    Alternatively, a list of all wards to be considered as internal can be provided.
    :param urgency_split_string: Where urgency should be splitted.
    :param birth_format: Format of birthday column.
    :param flow_split_string: Where flow should be splitted.
    :param timestamp_format: Format for timestamp to use.
    :param logger: Logger to use.

    :return: Preprocessor instance with preprocessed data.
    """

    if logger is None:
        logger = get_logger(__file__,
                            file_path=filepath[0].parent.joinpath(
                                f"{Path(__file__).resolve().stem}.log"))

    preprocessor = Preprocessor(filepath[0],
                                startdate=startdate,
                                enddate=enddate,
                                logger=logger)

    for file in filepath[1:]:
        preprocessor.add(file)

    preprocessor.logger.info(
        f"Succesfully imported data with shape {preprocessor.data.shape} "
        f"and columns\n{list(preprocessor.data.columns)}.")

    preprocessor.logger.info("Make patient flow.")
    preprocessor.make_flow(split_str=flow_split_string)

    preprocessor.logger.info("Clean from empty rows.")
    preprocessor.clean_empty()

    preprocessor.logger.info("Replace ward keys by mapping.")
    preprocessor.replace_ward_keys(ward_map=ward_key_map,
                                   internal_prefix=internal_prefix)

    preprocessor.logger.info("Make urgency (N0,...,N5) if existent.")
    preprocessor.make_urgency(split_str=urgency_split_string)

    preprocessor.logger.info("Clean data. This may take a while.")
    preprocessor.clean_data()
    preprocessor.logger.info("Finished cleaning.")

    preprocessor.logger.info("Make age from birth if existent.")
    preprocessor.make_age(format=birth_format)

    preprocessor.logger.info("Make timestamps to float.")
    preprocessor.datetimes_to_float(format=timestamp_format)

    preprocessor.data.loc[:, PRE_CLASS] = 0
    preprocessor.data.loc[:, CURRENT_CLASS] = 0
    preprocessor.data.loc[:, POST_CLASS] = 0

    preprocessor.logger.info("Prune data at beginning.")
    preprocessor.prune_begin()

    preprocessor.logger.info("Prune data at end.")
    preprocessor.prune_end()

    return preprocessor


def adjust_data(filepath: Path,
                wards: List[str],
                startdate: datetime,
                enddate: datetime,
                keep_internal: bool = False,
                logger: Optional[logging.Logger] = None) -> Preprocessor:
    """Adjust already preprocessed data.

    :param filepath: Path to file.
    :param wards: Wards to consider in analysis.
    :param startdate: Startdate to take.
    :param enddate: Enddate to take.
    :param keep_internal: Wether information on internal wards should be kept.
    :param logger: Logger to use.

    :return: Preprocessor instance with preprocessed data.
    """

    if logger is None:
        logger = get_logger(__file__,
                            file_path=filepath.parent.joinpath(
                                f"{Path(__file__).resolve().stem}.log"))

    preprocessor = Preprocessor(filepath=filepath, logger=logger)
    preprocessor.split_data()

    preprocessor.logger.info("Adjust start- and enddate.")
    preprocessor.alter_begin(startdate)
    preprocessor.alter_end(enddate)

    preprocessor.logger.info("Restrict wards if necessary.")
    preprocessor.restrict_ward(wards)

    if not keep_internal:
        preprocessor.logger.info(
            "Drop information on internal wards not under consideration.")
        preprocessor.drop_internal()

    return preprocessor
