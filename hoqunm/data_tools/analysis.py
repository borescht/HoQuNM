"""Provide methodologies for analysing the provided data."""

import json
import logging
from collections import Counter
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import graphviz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dataenforce import Dataset, validate
from sklearn import tree

from hoqunm.data_tools.base import (
    BEGIN, CART_COLUMNS, CART_COLUMNS_TRANSLATION, CAT_COLUMNS, CURRENT_CLASS,
    CURRENT_WARD, END, EXTERNAL, INTER_ARRIVAL, INTERNAL, MAX, MEAN, MIN,
    MIN_HEAD, OCCUPANCY, OUTPUT_DIR, PATIENT, POST_CLASS, POST_WARD, PRE_CLASS,
    PRE_WARD, SERVICE, SIGN, STD, TIME, WEEK, Min_Head_Name, and_query,
    column_query, drop_week_arrival, get_data, make_week, or_query)
from hoqunm.data_tools.preprocessing import Preprocessor
from hoqunm.simulation.evaluators import Evaluator
from hoqunm.simulation.hospital import HospitalSpecs
from hoqunm.utils.utils import annotate_heatmap, get_logger, heatmap

pd.set_option('mode.chained_assignment', None)

# pylint: disable=too-many-lines


class CartSpecs:
    """Holding specifications for CART analysis.

    :param wards: Wards to consider for CART.
    :param feature_columns: Columns to consider for analysis.
    :param max_depth: Maximal tree depth.
    :param min_samples_leaf: Minimal samples per leaf.
    """
    def __init__(self,
                 wards: List[str],
                 feature_columns: Optional[List[str]] = None,
                 cat_columns: Optional[List[str]] = None,
                 max_depth: int = 4,
                 min_samples_leaf: int = 200):
        self.wards = wards
        self.feature_columns = feature_columns if feature_columns is not None else CART_COLUMNS
        self.cat_columns = cat_columns if cat_columns is not None else CAT_COLUMNS
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

        if not all(c in self.feature_columns for c in self.cat_columns):
            raise ValueError("Not all catgerocial columns in feature columns.")

    @staticmethod
    def load_dict(arguments: Dict[str, Any]) -> "CartSpecs":
        """Create class from Dict with arguments and values in it.

        :param arguments: The dict containing the parameter-argument pairs.
        :return: Class instance.
        """
        return CartSpecs(**arguments)


class Analyser:
    """A class for analysing the data of the hospital. The data should come in
    the right format (possibly preprocessed with Preprocessor).

    :param filepath: The excel file, where the data is contained.
    :param sep: The separator with which the entries of the data file are seperated.
    """
    @validate
    def __init__(self,
                 data: Min_Head_Name = pd.DataFrame(columns=MIN_HEAD),
                 filepath: Path = Path(),
                 sep: str = ";",
                 startdate: datetime = datetime(2019, 1, 1),
                 enddate: datetime = datetime(2020, 1, 1),
                 datescale: timedelta = timedelta(1),
                 logger: Optional[logging.Logger] = None,
                 output_dir: Optional[Path] = None,
                 **kwargs: Any) -> None:
        self.datescale = datescale
        if data.empty and filepath != Path():
            self.data, self.startdate, self.enddate = get_data(
                filepath, sep, **kwargs)
        elif data.empty and filepath == Path():
            raise ValueError("Empty DataFrame and empty filepath given")
        else:
            self.data = data.copy()
            self.startdate = startdate
            self.enddate = enddate

        self.output_dir = output_dir if output_dir is not None else OUTPUT_DIR
        self.logger = logger if logger is not None else get_logger(
            "modeller", self.output_dir.joinpath("modeller.log"))

        assert all(column in self.data.columns for column in MIN_HEAD)

        self.wards, self.wards_map, self.wards_map_inv = self._make_wards()
        self.ward_occupancy = self._make_ward_occupancy()
        self.occupancy, self.occupancy_week, self.occupancy_weekend = self._make_occupancy(
        )
        self.capacities = self._read_capacities()
        self.hospital_specs = HospitalSpecs(
            capacities=np.array(self.capacities))
        self.hospital_specs.ward_map = self.wards_map_inv
        self.classes = self._make_classes()
        self.hospital_specs.set_U(len(self.classes))
        self.class_tree = tree.DecisionTreeRegressor()
        self.cart_code_map: Dict[Any, Dict[int, Any]] = dict()
        self.cart_graphs: List[Tuple[Any, graphviz.Source]] = []

        self.datescale = datescale

        self.make_week_column()

    def copy(self) -> "Analyser":
        """Copy self into a fresh object.

        :return: New instance of Analyser.
        """
        other = Analyser(data=self.data.copy())
        for key, value in self.__dict__.items():
            if not callable(getattr(self, key)):
                try:
                    setattr(other, key, value.copy())
                except AttributeError:
                    setattr(other, key, value)

        return other

    @staticmethod
    def from_preprocessor(
            preprocessor: Preprocessor,
            output_dir: Optional[Path] = None,
            logger: Optional[logging.Logger] = None) -> "Analyser":
        """Make an instance of self from Preprocessor.

        :param preprocessor: The preprocessor to use.
        :param output_dir: Output dir for plot saving.
        :param logger: Logger to use.

        :return: Analyser instance.
        """
        return Analyser(preprocessor.data,
                        startdate=preprocessor.startdate,
                        enddate=preprocessor.enddate,
                        datescale=preprocessor.datescale,
                        output_dir=output_dir,
                        logger=logger)

    def set_capacities(self, capacities: Dict[str, int]) -> None:
        """Set new capacities.

        This is mainly thought for reducing capacities. It is the
        general case, that the occupancies show much more beds than are
        practically available. To have a good distributions comparison,
        we want to adjust the occupancy distributions. Note that we do
        not adjust the ward-wise flows from which the occupancy
        distributions result! We adress this by filtering with >= instead of
        ==.

        :param capacities: Mapping for capacities to use.
        """
        for ward, capacity in capacities.items():
            self.capacities.loc[ward] = capacity
        self.hospital_specs.capacities = np.array(self.capacities)
        # squash occupancies
        self.occupancy = self._squash_occupancy(self.occupancy)
        self.occupancy_week = self._squash_occupancy(self.occupancy_week)
        self.occupancy_weekend = self._squash_occupancy(self.occupancy_weekend)

    def _squash_occupancy(self, occupancy: Dataset) -> Dataset:
        for ward, ser in occupancy.items():
            occupancy.loc[self.capacities[ward],
                          ward] = ser.loc[self.capacities[ward]:].sum()
            occupancy.loc[self.capacities[ward] + 1:, ward] = 0

        return occupancy

    @property
    def occupancies(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Return all different occupancies.

        :return: Occupancy, week occupancy and weekend occupancy.
        """
        return self.occupancy, self.occupancy_week, self.occupancy_weekend

    def make_occupancy(self) -> None:
        """Make the occupancies from the given data.

        Assume datetimes to be float!
        """

        self.ward_occupancy = self._make_ward_occupancy()

        self.occupancy, self.occupancy_week, self.occupancy_weekend = self._make_occupancy(
        )

        self.capacities = self._read_capacities()

        self.hospital_specs.capacities = np.array(self.capacities)

        self.hospital_specs.ward_map = self.wards_map_inv

        self.set_preclass()
        self.set_postclass()
        self.make_classes()

    def _make_ward_occupancy(self) -> Dict[Any, Dataset]:
        """Make the occupancy ward-wise."""
        ward_occupancy = {ward: pd.DataFrame() for ward in self.wards}

        end = (self.enddate - self.startdate) / self.datescale

        for ward in ward_occupancy:
            qry = column_query(CURRENT_WARD, ward)
            ward_data = self.data.query(qry)

            end_qry = column_query(END, end, ">=")
            end_index = ward_data.query(end_qry).index
            ward_data.loc[end_index, END] = float("NaN")

            ward_flow = pd.DataFrame({
                BEGIN:
                list(ward_data[BEGIN]) + list(ward_data[END]),
                SIGN: [1] * len(ward_data) + [-1] * len(ward_data)
            })
            ward_flow = ward_flow.dropna()
            ward_flow = ward_flow.sort_values(by=BEGIN,
                                              axis=0).reset_index(drop=True)
            ward_flow.loc[:, OCCUPANCY] = ward_flow[SIGN].cumsum()
            time = np.array(ward_flow[BEGIN])[1:] - np.array(
                ward_flow[BEGIN])[:-1]
            ward_flow = ward_flow.iloc[:-1]
            ward_flow.loc[:, TIME] = time
            ward_flow.loc[:, END] = ward_flow[BEGIN] + ward_flow[TIME]
            timeqry = column_query(BEGIN, 0, ">=")
            ward_flow = ward_flow.query(timeqry)
            begin = self.startdate.weekday()
            ward_flow = make_week(ward_flow, begin, BEGIN)

            ward_occupancy[ward] = ward_flow

        return ward_occupancy

    def _make_occupancy(self) -> Tuple[Dataset, Dataset, Dataset]:
        """Make occupancy."""
        occupancy = pd.DataFrame(
            0,
            columns=self.wards,
            index=range(
                max([
                    ward_flow[OCCUPANCY].max()
                    for ward, ward_flow in self.ward_occupancy.items()
                ]) + 1),
            dtype="float")
        occupancy_week = occupancy.copy()
        occupancy_weekend = occupancy.copy()
        for ward, ser in occupancy.items():
            for index in ser.index:
                all_qry = column_query(OCCUPANCY, index)
                occupancy.loc[index, ward] = self.ward_occupancy[ward].query(
                    all_qry)[TIME].sum()

                week_qry = and_query(column_query(OCCUPANCY, index),
                                     column_query(WEEK, 0))
                occupancy_week.loc[index,
                                   ward] = self.ward_occupancy[ward].query(
                                       week_qry)[TIME].sum()

                weekend_qry = and_query(column_query(OCCUPANCY, index),
                                        column_query(WEEK, 1))
                occupancy_weekend.loc[index,
                                      ward] = self.ward_occupancy[ward].query(
                                          weekend_qry)[TIME].sum()

        # finished
        occupancy = occupancy.divide(occupancy.sum(axis=0), axis=1)
        occupancy_week = occupancy_week.divide(occupancy_week.sum(axis=0),
                                               axis=1)
        occupancy_weekend = occupancy_weekend.divide(
            occupancy_weekend.sum(axis=0), axis=1)

        return occupancy, occupancy_week, occupancy_weekend

    def adjust_occupancy_pacu(self) -> None:
        """Adjust the occupancy for PACU."""
        if "PACU" in self.occupancy.columns:
            self.occupancy.loc[0,
                               "PACU"] = self.occupancy.loc[0, "PACU"] - 9 / 28
            self.occupancy.loc[:,
                               "PACU"] = self.occupancy.loc[:,
                                                            "PACU"] / (19 / 28)

    def regain_occupancy_pacu(self) -> None:
        """Revert the change made by adjust_occupancy_PACU."""
        if "PACU" in self.occupancy.columns:
            self.occupancy.loc[:,
                               "PACU"] = self.occupancy.loc[:,
                                                            "PACU"] * (19 / 28)
            self.occupancy.loc[0,
                               "PACU"] = self.occupancy.loc[0, "PACU"] + 9 / 28

    def _read_capacities(self) -> pd.Series:
        """Read capacities from self.occupancy."""

        capacities = pd.Series(0, index=self.occupancy.columns)
        for column, item in self.occupancy.iteritems():
            capacities.loc[column] = item[item != 0].index[-1]
        return capacities

    def make_week_column(self) -> None:
        """Make a boolean column, if the arriving day is a weekday or a weekend
        day.

        Also make a column, indicating which day of the week the arrival
        column is.
        """
        begin = self.startdate.weekday()
        self.data = make_week(self.data, begin, BEGIN)

    def plot_flow(self, capacities: pd.Series, squash: bool = False) -> None:
        """Plot the observed occupancy.

        :param capacities: Capacities of individual wards as should be.
        :param squash: If true, squash the flow to given capacities.
        """

        plot_begin = self.startdate

        for ward, ward_flow in self.ward_occupancy.items():
            max_time = ward_flow[END].max(skipna=True)
            plot_num = int(np.ceil(max_time / 365))
            plot_height = 0.3 * plot_num * (ward_flow[OCCUPANCY].max() -
                                            ward_flow[OCCUPANCY].min())
            fig = plt.figure(figsize=(12, plot_height))

            for j in range(int(np.ceil(max_time / 365))):
                ax = fig.add_subplot(plot_num, 1, j + 1)
                timeqry = and_query(column_query(BEGIN, j * 365, ">="),
                                    column_query(BEGIN, (j + 1) * 365, "<"))
                data = ward_flow.query(timeqry)
                if squash:
                    capacity = capacities[ward]
                    qry_squash = column_query(OCCUPANCY, capacity, ">")
                    index_squash = data.query(qry_squash).index
                    data.loc[index_squash, OCCUPANCY] = capacity
                ax.bar(data[BEGIN],
                       0.1,
                       align="edge",
                       bottom=data[OCCUPANCY] - 0.05,
                       width=data[TIME],
                       label=OCCUPANCY,
                       color="b")
                Analyser.plot_min_max_mean_flow(data, j * 365, (j + 1) * 365,
                                                1, ax)
                ax.plot([
                    max(j * 365, data[BEGIN].min()),
                    min((j + 1) * 365, data[BEGIN].max())
                ], [capacities[ward]] * 2,
                        linewidth=5,
                        color="black",
                        label="Ward capacity")
                ax.set_title(
                    "Occupancy for ward: {}, year: {}, squashed: {}.".format(
                        ward, j + 1, squash))
                ax.set_yticks(list(set(data[OCCUPANCY])))
                x_ticklabels = [
                    date(plot_begin.year + j, m, 1)
                    for m in range(1, 13) if m >= plot_begin.month
                ] + [
                    date(plot_begin.year + j + 1, m, 1)
                    for m in range(1, 13) if m < plot_begin.month
                ]
                x_tick_datetimes = [
                    datetime(plot_begin.year + j, m, 1)
                    for m in range(1, 13) if m >= plot_begin.month
                ] + [
                    datetime(plot_begin.year + j + 1, m, 1)
                    for m in range(1, 13) if m < plot_begin.month
                ]
                x_ticks = [(label - plot_begin) / timedelta(1)
                           for label in x_tick_datetimes]
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_ticklabels, rotation=45)
                x_ticks_minor = list(range(j * 365, (j + 1) * 365))
                ax.set_xticks(x_ticks_minor, minor=True)
                ax.set_xlabel("Months")
                ax.set_ylabel("Patients")
                ax.legend()
                ax.grid()

            fig.tight_layout()

            if not squash:
                filename = f"occupancy - ward[{ward}].pdf"
                plt.savefig(self.output_dir.joinpath(filename))
                plt.close()
            else:
                filename = f"occupancy_squashed - ward[{ward}].pdf"
                plt.savefig(self.output_dir.joinpath(filename))
                plt.close()

    @staticmethod
    def plot_min_max_mean_flow(data: Dataset, begin: int, end: int, step: int,
                               ax: plt.axis) -> None:
        """Plot the minimal, maximal and mean occupancies for the given steps
        (intervals).

        :param data: The data to use.
        :param begin: The begin time.
        :param end: The last time to use.
        :param step: The windows/step size to use.
        :param ax: The axis to plot to.
        """
        last_occupancy = data.iloc[0][OCCUPANCY]
        occupancy_df = pd.DataFrame([],
                                    columns=[MIN, MAX, MEAN, "Last"],
                                    index=range(begin, end, step))
        begin = int(max(data[BEGIN].min(), begin))
        end = int(min(data[BEGIN].max(), end))
        for i in range(begin, end, step):
            rangeqry = and_query(column_query(BEGIN, i, ">="),
                                 column_query(BEGIN, i + step, "<="))
            range_data = data.query(rangeqry)[[BEGIN, TIME, OCCUPANCY]]
            if not range_data.empty:
                range_data.loc[range_data.index[-1],
                               TIME] = i + step - range_data.iloc[-1][BEGIN]
                start_time = range_data.iloc[0][BEGIN] - i
                begin_ser = pd.Series([i, start_time, last_occupancy],
                                      index=[BEGIN, TIME, OCCUPANCY])
                last_occupancy = range_data.loc[range_data.index[-1],
                                                OCCUPANCY]
                range_data = range_data.append(begin_ser, ignore_index=True)
            else:
                range_data = pd.DataFrame([[i, step, last_occupancy]],
                                          columns=[BEGIN, TIME, OCCUPANCY],
                                          index=[0])
            min_occupancy = range_data[OCCUPANCY].min()
            max_occupancy = range_data[OCCUPANCY].max()
            mean_occupancy = max(min_occupancy,
                                 (range_data[OCCUPANCY] * range_data[TIME] /
                                  step).sum())
            occupancy_df.loc[i] = [
                min_occupancy, max_occupancy, mean_occupancy, last_occupancy
            ]

        ax.plot(occupancy_df[MIN], linewidth=1, label=MIN, color="y")
        ax.plot(occupancy_df[MAX], linewidth=1, label=MAX, color="g")
        ax.plot(occupancy_df[MEAN], linewidth=1, label=MEAN, color="r")

    def plot_occupancy(self) -> None:
        """Plot the observed occupancy distributions."""
        ob_ev_all = Evaluator(self.hospital_specs)
        ob_ev_all.distributions.occupancy = np.array(self.occupancy).T
        ob_ev_all.name = "Whole observation"

        ob_ev_week = Evaluator(self.hospital_specs)
        ob_ev_week.distributions.occupancy = np.array(self.occupancy_week).T
        ob_ev_week.name = "Week observation"

        ob_ev_weekend = Evaluator(self.hospital_specs)
        ob_ev_weekend.distributions.occupancy = np.array(
            self.occupancy_weekend).T
        ob_ev_weekend.name = "Weekend observation"

        ob_ev_all.plot_against([ob_ev_week, ob_ev_weekend])

    def feature_correlation(self,
                            cart_specs: Optional[CartSpecs] = None) -> None:
        """Compute the correlation of given features.

        :param cart_specs: Specifications for cart analysis to use.
        """

        if SERVICE not in self.data.columns:
            self.make_service()

        data = self.data.copy()

        if cart_specs is None:
            cart_specs = CartSpecs(wards=self.wards)
        hist_columns = cart_specs.feature_columns + [SERVICE]
        data.loc[:, cart_specs.cat_columns] = data.loc[:, cart_specs.
                                                       cat_columns].astype(
                                                           "category")

        for column in cart_specs.cat_columns:
            data.loc[:, column] = data.loc[:, column].cat.codes
        num_wards = len(self.wards)
        fig = plt.figure(figsize=(12, 4 * int(np.ceil(num_wards / 2))))
        for i, ward in enumerate(self.wards):
            qry = column_query(CURRENT_WARD, ward)
            data_ = data.query(qry)[hist_columns]
            ax = fig.add_subplot(int(np.ceil(num_wards / 2)), 2, i + 1)
            ax.set_title("Ward: {}, number of patients: {}".format(
                ward, data_.shape[0]))
            im, _ = heatmap(data_.corr(),
                            data_.columns,
                            data_.columns,
                            ax=ax,
                            cmap="YlGn",
                            cbarlabel="correlation")
            annotate_heatmap(im, valfmt="{x:.3f}", threshold=0)

        fig.tight_layout()

    def day_arrival(self) -> None:
        """Compute the daily culmulated arrivals."""
        self.make_week_column()
        fig = plt.figure(figsize=(12, 3 * len(self.wards)))
        for i, ward in enumerate(self.wards):
            qry = column_query(CURRENT_WARD, ward)
            data = self.data.query(qry)

            ax = fig.add_subplot(len(self.wards), 1, i + 1)
            ax.set_title("Ward: {}, number of patients: {}".format(
                ward, data.shape[0]))

            pairs: List[Tuple[Any, Any]] = list(
                Counter(data["Weekday"].astype("int")).items())
            pairs.sort(key=lambda x: x[0])
            pairs_ = list(zip(*pairs))
            x_ax = pairs_[0]
            y_ax = pairs_[1]
            ax.bar(x_ax, y_ax)

            qry_out = column_query(PRE_WARD, EXTERNAL)
            pairs_out: List[Tuple[Any, Any]] = list(
                Counter(data.query(qry_out)["Weekday"].astype("int")).items())
            pairs_out.sort(key=lambda x: x[0])
            pairs_out_ = list(zip(*pairs_out))
            x_ax_o = pairs_out_[0]
            y_ax_o = pairs_out_[1]
            ax.bar(x_ax_o, y_ax_o)

            ax.set_xticks(x_ax)
            ax.set_xticklabels(["Mo", "Tu", "We", "Thu", "Fr", "Sa", "Su"])
        fig.tight_layout()

    @validate
    def rolling_arrival(
        self,
        window: float = 365.,
        pre_ward: Optional[List[str]] = None
    ) -> Dict[Tuple[str, Any], Dataset[MEAN, STD]]:
        """Compute the rolling arrival mean and variance.

        :param window: The window size to use.
        :param pre_ward: The pre_wards for which the rolling arrival.
        shall be computed.

        :return: The rolling_arrival for the respective wards.
        """
        if pre_ward is None:
            pre_ward = [EXTERNAL]

        rolling_arrival_dict = {
            (ward, class_): self.rolling_arrival_ward_class(ward=ward,
                                                            class_=class_,
                                                            pre_ward=pre_ward,
                                                            window=window)
            for class_ in self.classes for ward in self.wards
        }

        plot_num = len(self.wards) * len(self.classes)
        fig = plt.figure(figsize=(12, 3 * plot_num))

        for i, key_value in enumerate(rolling_arrival_dict.items()):
            ward_class, rolling_arrival_ = key_value
            ward, class_ = ward_class
            ax = fig.add_subplot(plot_num, 1, i + 1)
            ax.plot(rolling_arrival_.loc[:, MEAN], label=MEAN)
            ax.plot(rolling_arrival_.loc[:, STD], label=STD)
            ax.legend()
            ax.set_title(f"Rolling inter-arrival for ward {ward}\n"
                         f"with window size {window}")
            ax.set_xlabel("Starting time")
            ax.set_ylabel("Days")
            ax.grid(axis="y")

        fig.tight_layout()

        return rolling_arrival_dict

    @validate
    def rolling_arrival_ward_class(self,
                                   ward: Any,
                                   class_: Any,
                                   pre_ward: Optional[List[Any]] = None,
                                   window: float = 365.) -> Dataset[MEAN, STD]:
        """Compute the rolling arrival mean and variance for the given ward and
        class.

        :param ward: The ward under consideration.
        :param class_: The class under consideration.
        :param pre_ward: The pre_wards for which the rolling arrival
        shall be computed.
        :param window: The window size to use.

        :return: The rolling_arrival for the respective ward and class.
        """
        if pre_ward is None:
            pre_ward = [EXTERNAL]

        ward_class_qry = and_query(column_query(CURRENT_WARD, ward),
                                   column_query(CURRENT_CLASS, class_))
        ward_class_data = self.data.query(ward_class_qry).dropna(
            subset=[BEGIN, END])
        ward_class_data.loc[:, "Arrival"] = self.make_inter_arrival(
            ward_class_data, pre_ward=pre_ward)
        ward_class_data = ward_class_data.dropna(subset=["Arrival"])
        last_time = int(max(ward_class_data[BEGIN].dropna().max() - window, 1))
        df = pd.DataFrame(columns=[MEAN, STD], index=range(last_time))
        if ward == "PACU":
            ward_class_data = drop_week_arrival(ward_class_data, week=True)
        for i in range(last_time):
            qry = and_query(column_query(BEGIN, i, ">="),
                            column_query(BEGIN, i + window, "<="))
            data = ward_class_data.query(qry)
            df.loc[i, MEAN] = data["Arrival"].mean()
            df.loc[i, STD] = data["Arrival"].std()
        return df

    @validate
    def rolling_arrival_ratio(
            self,
            window: float = 365.) -> Dict[Tuple[str, Any], Dataset[MEAN, STD]]:
        """Compute the rolling arrival ratio for the given ward and class.

        :param window: The window size to use.

        :return: The rolling_arrival for the respective ward and class.
        """
        rolling_arrival_ratio_dict = {
            (ward, class_):
            self.rolling_arrival_ward_class(ward=ward,
                                            class_=class_,
                                            pre_ward=[INTERNAL, EXTERNAL],
                                            window=window) /
            self.rolling_arrival_ward_class(
                ward=ward, class_=class_, pre_ward=[INTERNAL], window=window)
            for class_ in self.classes for ward in self.wards
        }

        plot_num = len(self.wards) * len(self.classes)
        fig = plt.figure(figsize=(12, 4 * plot_num))

        for i, key_value in enumerate(rolling_arrival_ratio_dict.items()):
            ward_class, rolling_arrival_ratio_ = key_value
            ward, class_ = ward_class
            ax = fig.add_subplot(plot_num, 1, i + 1)
            ax.plot(rolling_arrival_ratio_[MEAN],
                    label="INTERNAL/(INTERNAL+EXTERNAL)")
            ax.legend()
            rec_qry = and_query(column_query(CURRENT_WARD, ward),
                                column_query(CURRENT_CLASS, class_))
            records = len(self.data.query(rec_qry))
            ax.set_title(
                "Rolling arrival for ward {}, class {} and as "
                "INTERNAL/(INTERNAL+EXTERNAL),\nfor #records: {}, with window size {}"
                .format(ward, class_, records, window))
            ax.set_xlabel("Starting time")

        fig.tight_layout()
        return rolling_arrival_ratio_dict

    @validate
    def rolling_service(
            self,
            window: float = 365.) -> Dict[Tuple[str, Any], Dataset[MEAN, STD]]:
        """Compute the rolling service mean and variance.

        :param window: The window size to use.

        :return: The rolling_service for the respective wards.
        """
        rolling_service_dict = {(ward, clas):
                                self.rolling_service_ward_class(ward=ward,
                                                                class_=clas,
                                                                window=window)
                                for clas in self.classes
                                for ward in self.wards}

        plot_num = len(self.wards) * len(self.classes)
        fig = plt.figure(figsize=(12, 4 * plot_num))

        for i, key_value in enumerate(rolling_service_dict.items()):
            ward_class, rolling_service_ = key_value
            ward, clas = ward_class
            ax = fig.add_subplot(plot_num, 1, i + 1)
            ax.plot(rolling_service_[MEAN], label=MEAN)
            ax.plot(rolling_service_[STD], label=STD)
            ax.legend()
            ax.set_title("Rolling service for ward {}\n"
                         "with window size {}".format(ward, window))

            ax.set_xlabel("Starting time")
            ax.set_ylabel("Days")
            ax.grid(axis="y")

        fig.tight_layout()

        return rolling_service_dict

    @validate
    def rolling_service_ward_class(self,
                                   ward: Any,
                                   class_: Any,
                                   window: float = 365.) -> Dataset[MEAN, STD]:
        """Compute the rolling service mean and variance for the given ward and
        class.

        :param ward: The ward under consideration.
        :param class_: The class under consideration.
        :param window: The window size to use.

        :return: The rolling_service for the respective ward and class
        """

        self.make_service()
        ward_class_qry = and_query(column_query(CURRENT_WARD, ward),
                                   column_query(CURRENT_CLASS, class_))
        ward_class_data = self.data.query(ward_class_qry)
        ward_class_data = ward_class_data.dropna(subset=[BEGIN, END])
        last_time = int(max(ward_class_data[BEGIN].max() - window, 1))
        df = pd.DataFrame(columns=[MEAN, STD], index=range(last_time))
        for i in range(last_time):
            qry = and_query(column_query(BEGIN, i, ">="),
                            column_query(BEGIN, i + window, "<="))
            data = ward_class_data.query(qry)
            df.loc[i, MEAN] = data[SERVICE].mean()
            df.loc[i, STD] = data[SERVICE].std()

        return df

    def rolling_routing(self,
                        window: float = 365.,
                        incoming: bool = False) -> Dict[str, Dataset]:
        """Compute the rolling routings for a given ward assuming just one
        class.

        :param window: The window size to use.
        :param incoming: Wether to consider incoming or outgoing patients.

        :return: The rolling_routing for the respective wards.
        """

        rolling_routing_dict = {
            ward: self.rolling_routing_ward(ward=ward,
                                            window=window,
                                            incoming=incoming)
            for ward in self.wards
        }

        plot_num = len(self.wards)
        fig = plt.figure(figsize=(12, 4 * plot_num))

        for i, key_value in enumerate(rolling_routing_dict.items()):
            ward, rolling_routing_ = key_value
            ax = fig.add_subplot(plot_num, 1, i + 1)
            self.plot_rolling_routing(ward, rolling_routing_, window, ax)

        fig.tight_layout()

        return rolling_routing_dict

    @staticmethod
    def plot_rolling_routing(ward: Any, rolling_routing_: Dataset,
                             window: float, ax: plt.axis) -> None:
        """Plot the computed rolling routing.

        :param ward: The ward to plot it for.
        :param rolling_routing_: The computed rolling routing DataFrame.
        :param window: The window size.
        :param ax: The axis to plot to.
        """
        for ward2, rolling_routing_ward in rolling_routing_.iteritems():
            ax.plot(rolling_routing_ward, label=ward2)
        ax.legend()
        ax.set_title("Rolling routing for ward {} \n"
                     "with window size {}".format(ward, window))
        ax.set_xlabel("Starting time")
        ax.set_ylabel("Routing probability")
        ax.grid(axis="y")

    def rolling_routing_ward(self,
                             ward: Any,
                             window: float = 365.,
                             incoming: bool = False) -> Dataset:
        """Compute the rolling routing for a given ward.

        :param ward: The ward under consideration.
        :param window: The window size to use.
        :param incoming: Analyse incoming or outgoing.

        :return: The rolling_routing for the respective ward.
        """

        data = self.data.copy()
        if POST_CLASS not in data.columns:
            data[POST_CLASS] = 0
        data.loc[:, [CURRENT_CLASS, POST_CLASS]] = 0
        last_time = int(max(data[BEGIN].max() - window, 1))
        post_wards = list(data[POST_WARD].dropna().unique())
        post_wards.remove(INTERNAL)
        if incoming:
            post_wards.append(INTERNAL)
        df = pd.DataFrame(columns=post_wards, index=range(last_time))
        post_wards.remove(EXTERNAL)
        for i in range(last_time):
            qry = and_query(column_query(BEGIN, i, ">="),
                            column_query(BEGIN, i + window, "<="))
            data_ = data.query(qry)

            for ward2 in post_wards:
                r = self.compute_routing(data_,
                                         ward,
                                         ward2,
                                         0,
                                         0,
                                         incoming=incoming)
                df.loc[i, ward2] = r
            df.loc[i, EXTERNAL] = 1 - df.loc[i].sum()

        return df

    def rolling_occupancy(self,
                          window: float = 365.,
                          step: float = 183.) -> Dict[str, Dataset]:
        """Compute the rolling occupancy distributions.

        :param window: The window size to use.
        :param step: The stepsize. We do not want to have it rolling over all days,
                     so only step_wise.

        :return: The rolling_occupancy for the respective wards
        """

        rolling_occupancy_dict = {
            ward: self.rolling_occupancy_ward(ward, window=window, step=step)
            for ward in self.wards
        }

        plot_num = len(self.wards)
        fig = plt.figure(figsize=(12, 4 * plot_num))

        for i, key_value in enumerate(rolling_occupancy_dict.items()):
            ward, rolling_occupancy_ = key_value
            ax = fig.add_subplot(plot_num, 1, i + 1)
            capacities = np.array([rolling_occupancy_.index[-1]] *
                                  len(rolling_occupancy_.columns))
            evaluator = Evaluator(HospitalSpecs(capacities))
            # squash occupancy to max capacity
            evaluator.distributions.occupancy = np.array(rolling_occupancy_).T
            evaluator.plot_occupancy(ax=ax)
            ax.legend(rolling_occupancy_.columns)
            rec_qry = column_query(CURRENT_WARD, ward)
            records = len(self.data.query(rec_qry))
            ax.set_title(
                "Rolling occupancy for ward {}\n"
                "for #records: {}, with window size {} and step size {}".
                format(ward, records, window, step))

        fig.tight_layout()

        return rolling_occupancy_dict

    def rolling_occupancy_ward(self,
                               ward: Any,
                               window: float = 365.,
                               step: float = 183.) -> Dataset:
        """Compute the rolling occupancy distributions for a given ward.

        :param ward: The ward under consideration.
        :param window: The window size to use.
        :param step: The stepsize. We do not want to have it rolling over all days,
                     so only step_wise.

        :return: The rolling_occupancy for the respective ward.
        """

        ward_flow = self.ward_occupancy[ward]
        steps = max(int(np.ceil(
            (ward_flow[BEGIN].max() - window) / step)), 1) + 1
        occupancy = pd.DataFrame(0,
                                 columns=[step * i for i in range(steps)],
                                 index=range(ward_flow[OCCUPANCY].max() + 1),
                                 dtype="float")
        for i in range(steps):
            for index in occupancy.index:
                qry = and_query(column_query(OCCUPANCY, index),
                                column_query(BEGIN, i * step, ">"),
                                column_query(BEGIN, i * step + window, "<"))
                occupancy.loc[index,
                              i * step] = ward_flow.query(qry)[TIME].sum()

        # finished
        occupancy = occupancy.divide(occupancy.sum(axis=0), axis=1)

        return occupancy

    def cart_classes(
        self,
        cart_specs: Optional[CartSpecs] = None
    ) -> Tuple[Dict[Any, Dict[int, Any]], List[Tuple[Any, graphviz.Source]]]:
        """Cecide on classes using CART assume values in features to be
        sortable and already numerical.

        :param cart_specs: Specifications to use.

        :return: The graphvis source files holding the decision trees.
        """

        if cart_specs is None:
            cart_specs = CartSpecs(wards=self.wards)

        # convert values to categories first
        data = self.data.copy()

        data[cart_specs.feature_columns] = data[
            cart_specs.feature_columns].astype("category")
        code_map = {
            feature: dict(enumerate(data[feature].cat.categories))
            for feature in cart_specs.cat_columns
        }
        for feature in cart_specs.cat_columns:
            data[feature] = data[feature].cat.codes

        graphs = []

        for ward in cart_specs.wards:
            qry = column_query(CURRENT_WARD, ward)
            data_ = data.query(qry).dropna(subset=[BEGIN, END])
            X = np.array(data_.loc[:, cart_specs.feature_columns])
            y = np.array(data_.loc[:, END] - data_.loc[:, BEGIN])
            clf = tree.DecisionTreeRegressor(
                max_depth=cart_specs.max_depth,
                min_samples_leaf=cart_specs.min_samples_leaf)
            clf_fit = clf.fit(X, y)
            self.class_tree = clf_fit

            data_apply = data.query(qry)
            X_apply = np.array(data_apply.loc[:, cart_specs.feature_columns])
            leaf_id = clf.apply(X_apply)
            self.data.loc[data_apply.index, CURRENT_CLASS] = leaf_id
            self.set_preclass()
            self.set_postclass()
            self.make_classes()

            feature_names = cart_specs.feature_columns.copy(
            ) if cart_specs.feature_columns is not None else []
            for i, name in enumerate(feature_names):
                if CART_COLUMNS_TRANSLATION.get(name, None) is not None:
                    feature_names[i] = CART_COLUMNS_TRANSLATION[name]
            dot_data = tree.export_graphviz(clf,
                                            out_file=None,
                                            feature_names=feature_names,
                                            filled=True,
                                            rounded=True,
                                            special_characters=True)
            graph = graphviz.Source(dot_data)
            graphs.append((ward, graph))
        self.cart_code_map = code_map
        self.cart_graphs = graphs

        return code_map, graphs

    def set_preclass(self) -> None:
        """Analyze the data and set the preclasses from given current
        classes."""

        self.data[PRE_CLASS] = float("NaN")
        for patient in set(self.data[PATIENT]):
            patient_data = self.data.query(column_query(
                PATIENT, patient)).sort_values(by=BEGIN)
            class_data = patient_data.iloc[:-1][CURRENT_CLASS]
            class_data.index = patient_data.index[1:]
            patient_data[PRE_CLASS] = class_data
            patient_data.loc[patient_data.index[0],
                             PRE_CLASS] = patient_data.loc[
                                 patient_data.index[0], CURRENT_CLASS]
            self.data.loc[patient_data.index] = patient_data

    def set_postclass(self) -> None:
        """Analyze the data and set the postclasses from given current
        classes."""

        self.data[POST_CLASS] = float("NaN")
        for patient in set(self.data[PATIENT]):
            patient_data = self.data.query(column_query(
                PATIENT, patient)).sort_values(by=BEGIN)
            class_data = patient_data.iloc[1:][CURRENT_CLASS]
            class_data.index = patient_data.index[:-1]
            patient_data[POST_CLASS] = class_data
            patient_data.loc[patient_data.index[-1],
                             POST_CLASS] = patient_data.loc[
                                 patient_data.index[-1], CURRENT_CLASS]
            self.data.loc[patient_data.index] = patient_data

    def make_classes(self) -> None:
        """Read classes from data."""

        self.classes = self._make_classes()
        self.hospital_specs.set_U(len(self.classes))

    def _make_classes(self) -> List[Any]:
        """Read classes form data."""
        classes = list(self.data[CURRENT_CLASS].unique())
        classes.sort()
        return classes

    def make_wards(self) -> None:
        """Read wards from data."""

        self.wards, self.wards_map, self.wards_map_inv = self._make_wards()

    def _make_wards(self) -> Tuple[List[Any], Dict[Any, int], Dict[int, Any]]:
        """Make the ward list with index mapping from the data."""

        wards: List[Any] = list(self.data[CURRENT_WARD].unique())
        wards.sort()
        wards_map = {ward: i for i, ward in enumerate(wards)}
        wards_map_inv = dict(enumerate(wards))

        return wards, wards_map, wards_map_inv

    @staticmethod
    def compute_routing(data: Dataset,
                        ward1: Any,
                        ward2: Any,
                        class1: Any,
                        class2: Any,
                        incoming: bool = False) -> float:
        """Compute the routing probability for 2 given wards and 2 given
        classes.

        :param data: The data under consideration.
        :param ward1: The preceding ward.
        :param ward2: The current ward.
        :param class1: The preceding class.
        :param class2: The current class.
        :param incoming: Wether to consider incoming or outgoing patients.

        :return: The probability.
        """
        qry1 = and_query(column_query(CURRENT_WARD, ward1),
                         column_query(CURRENT_CLASS, class1),
                         column_query(BEGIN, 0, ">="))
        if incoming:
            flow_ward = PRE_WARD
            flow_class = PRE_CLASS
        else:
            flow_ward = POST_WARD
            flow_class = POST_CLASS
        qry2 = and_query(column_query(flow_ward, ward2),
                         column_query(CURRENT_WARD, ward1),
                         column_query(flow_class, class2),
                         column_query(CURRENT_CLASS, class1),
                         column_query(BEGIN, 0, ">="))

        class1_patients = data.query(qry1).dropna(subset=[BEGIN, END])
        class2_patients = data.query(qry2).dropna(subset=[BEGIN, END])

        if len(class1_patients) > 0:
            r = len(class2_patients) / len(class1_patients)
        else:
            r = 0

        return r

    @staticmethod
    def make_inter_arrival(data: Dataset,
                           pre_ward: Optional[List[Any]] = None) -> pd.Series:
        """Iterate over arrival and substract last arrival from current
        arrival. Assume data for a given ward and class.

        :param data: The data under investigation (for a special ward/class).
        :param pre_ward: The pre_wards to consider.

        :return: The inter-arrival times for the given data.
        """
        if pre_ward is None:
            pre_ward = [EXTERNAL, INTERNAL]
        qry = and_query(
            or_query(
                *[column_query(PRE_WARD, pre_ward_)
                  for pre_ward_ in pre_ward]), column_query(BEGIN, 0, ">="))
        data = data.query(qry)[BEGIN]
        data = data.sort_values()
        if not data.empty:
            arrival_data = (data.iloc[1:].reset_index(drop=True) -
                            data.iloc[:-1].reset_index(drop=True))
        else:
            arrival_data = data
        arrival_data.index = data.index[1:]
        arrival_data.name = INTER_ARRIVAL
        arrival_data = arrival_data.dropna()
        arrival_data = arrival_data[arrival_data != 0]

        return arrival_data

    def make_service(self, classes: Optional[List[int]] = None) -> None:
        """Make the service data for the given set of classes.

        :param classes: The list of class indices under consideration.
        """

        if classes is None:
            if hasattr(self, "classes"):
                classes = self.classes
            else:
                classes = [0]

        self.data.loc[:, SERVICE] = float("NaN")
        data = self.data.copy()

        for ward in self.wards:
            for class_ in classes:
                qry = and_query(column_query(CURRENT_WARD, ward),
                                column_query(CURRENT_CLASS, class_),
                                column_query(BEGIN, 0, ">="))
                class_data = data.query(qry).dropna(subset=[BEGIN, END])
                if not class_data.empty:
                    service_data = (class_data[END] - class_data[BEGIN])
                    self.data.loc[service_data.index, SERVICE] = service_data

        self.data.loc[:, SERVICE] = self.data[SERVICE].astype("float")


def analyse(preprocessor: Preprocessor,
            wards: List[str],
            capacities: List[int],
            adjust_pacu_occupancy: bool = True,
            output_dir: Path = OUTPUT_DIR,
            logger: Optional[logging.Logger] = None) -> Analyser:
    """Analyse data from preprocessor.

    :param preprocessor: Preprocessor to use.
    :param wards: Wards to consider.
    :param capacities: Capacity per ward to consider.
    :param adjust_pacu_occupancy: Adjust pacu occupancy because of weekends.
    :param output_dir: Directory for plot saving.
    :param logger: Logger to use for logging.

    :return: Anaylser instance with analysed data.
    """

    if not output_dir.is_dir():
        output_dir.mkdir()

    if logger is None:
        logger = get_logger(__file__,
                            file_path=output_dir.joinpath(
                                f"{Path(__file__).resolve().stem}.log"))

    analyser = Analyser.from_preprocessor(preprocessor=preprocessor,
                                          output_dir=output_dir,
                                          logger=logger)

    analyser.logger.info("Number of entries in data: {}.".format(
        len(analyser.data)))
    analyser.logger.info("Entries per ward: {}".format(
        Counter(analyser.data[CURRENT_WARD])))

    analyser.logger.info("Make occupancy.")
    analyser.make_occupancy()

    analyser.logger.info("Compute flow and make plots.")
    ward_capacity = pd.Series(dict(zip(wards, capacities)))
    analyser.plot_flow(capacities=ward_capacity, squash=False)

    analyser.plot_flow(capacities=ward_capacity, squash=True)

    analyser.plot_occupancy()
    s = ", ".join(wards)
    filename = f"distributions - wards[{s}].pdf"
    plt.savefig(output_dir.joinpath(filename))
    plt.close()

    if "PACU" in wards and adjust_pacu_occupancy:
        analyser.logger.info("Adjust PACU occupancy.")
        analyser.adjust_occupancy_pacu()

    analyser_ = analyser.copy()
    analyser_.set_capacities(capacities=ward_capacity)

    ob_ev_all = Evaluator(analyser_.hospital_specs)
    ob_ev_all.distributions.occupancy = np.array(analyser_.occupancy).T

    Evaluator.plot_many(evaluation_results=[ob_ev_all],
                        colors=["r"],
                        markers=["*"],
                        labels=["Real observation"])
    filename = f"real_observation.pdf"
    plt.savefig(output_dir.joinpath(filename))
    plt.close()

    return analyser


def advanced_analysis(preprocessor: Preprocessor,
                      wards: List[str],
                      capacities: List[int],
                      cart_specs: CartSpecs,
                      output_dir: Path,
                      window_size: float = 90.,
                      adjust_pacu_occupancy: bool = True,
                      logger: Optional[logging.Logger] = None) -> Analyser:
    """Undertake some advanced analysis on existing Analyser class.

    :param preprocessor: Preprocessor to use.
    :param wards: Wards to consider.
    :param capacities: Capacity per ward to consider.
    :param cart_specs: Specifications for cart to use.
    :param output_dir: Output dir for plot saving.
    :param window_size: Window_size for rolling plots.
    :param adjust_pacu_occupancy: Adjust pacu occupancy because of weekends.
    :param logger: Logger to use for logging.

    :return: Anaylser instance with anaylsed data.
    """

    if logger is None:
        logger = get_logger(__file__,
                            file_path=output_dir.joinpath(
                                f"{Path(__file__).resolve().stem}.log"))

    analyser = analyse(preprocessor=preprocessor,
                       wards=wards,
                       capacities=capacities,
                       adjust_pacu_occupancy=adjust_pacu_occupancy,
                       output_dir=output_dir,
                       logger=logger)

    analyser.feature_correlation(cart_specs=cart_specs)
    plt.savefig(output_dir.joinpath(f"Feature correlation.pdf"))
    plt.close()

    for pre_ward in [[EXTERNAL], [INTERNAL], [EXTERNAL, INTERNAL]]:
        analyser.rolling_arrival(pre_ward=pre_ward, window=window_size)
        key = ",".join(pre_ward)
        plt.savefig(output_dir.joinpath(f"Rolling arrival - {key}.pdf"))
        plt.close()

    analyser.rolling_service(window=window_size)
    plt.savefig(output_dir.joinpath(f"Rolling service.pdf"))
    plt.close()

    for incoming in [True, False]:
        analyser.rolling_routing(incoming=incoming, window=window_size)
        key = "incoming" if incoming else "outgoing"
        plt.savefig(
            output_dir.joinpath(
                f"Rolling routing - {key}pre_ward=[external].pdf"))
        plt.close()

    code_map, graphs = analyser.cart_classes(cart_specs=cart_specs)
    with open(output_dir.joinpath("Decision tree - code map.json"), "w") as f:
        json.dump(code_map, f)

    # pylint: disable=broad-except
    for ward, graph in graphs:
        try:
            graph.render(
                output_dir.joinpath(f"Decision tree for ward {ward}.gv"))
        except BaseException as e:
            logger.warning(
                f"Could not save decision tree data. Received error {e}.")

    return analyser
