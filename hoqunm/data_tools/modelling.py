"""Provide methodologies for creating the model from the data."""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats

from hoqunm.data_tools.analysis import Analyser, CartSpecs, analyse
from hoqunm.data_tools.base import (BEGIN, CART_COLUMNS, CURRENT_CLASS,
                                    CURRENT_WARD, END, EXTERNAL, INTERNAL,
                                    OUTPUT_DIR, SERVICE, and_query,
                                    column_query, drop_week_arrival)
from hoqunm.data_tools.preprocessing import adjust_data
from hoqunm.simulation.evaluators import EvaluationResults
from hoqunm.simulation.hospital import HospitalSpecs
from hoqunm.utils.distributions import (Hypererlang, HypererlangSpecs,
                                        distribution_to_rate, fit_expon,
                                        fit_hypererlang, plot_distribution_fit,
                                        rate_to_distribution)
from hoqunm.utils.utils import LOGGING_DIR, get_logger

pd.set_option('mode.chained_assignment', None)


class Modeller:
    """Class for creating the model with the help of Analyser class.

    A possible workflow would be:
        1. __init__ with analyser
        3. analyser.make_occupancy
        4. analyser.make_classes or analyser.set_classes
        5. self.inter_arrival_fit
        6. self.service_fit
        7. self.routing
        (8. self.adjust_fits)

    :param analyser: The analyser (already finished) to use.
    :param logger: Logger for logging.
    :param output_dir: Output directory for plots.
    """
    def __init__(self,
                 analyser: Analyser,
                 logger: Optional[logging.Logger] = None,
                 output_dir: Path = OUTPUT_DIR):
        self.analyser = analyser
        self.hospital_specs = self.analyser.hospital_specs
        self.output_dir = output_dir
        self.logger = logger if logger is not None else get_logger(
            "modeller", self.output_dir.joinpath("modeller.log"))

    def copy(self) -> "Modeller":
        """Copy self.

        :return: New instance with same attributes.
        """
        out = Modeller(analyser=self.analyser.copy(),
                       logger=self.logger,
                       output_dir=self.output_dir)
        return out

    def make_cart(
        self,
        cart_specs: Optional[CartSpecs] = None
    ) -> Tuple[Dict[Any, Any], Any]:
        """Make cart analysis.

        :param cart_specs: Specifications for CART analysis.

        :return: Code map for columns converted to categorical, created CART graphs.
        """

        if cart_specs is None:
            cart_specs = CartSpecs(wards=self.analyser.wards)

        code_map, graphs = self.analyser.cart_classes(cart_specs=cart_specs)

        self.logger.info(f"CART mapping: \n{code_map}")
        return code_map, graphs

    # pylint: disable=too-many-nested-blocks
    def inter_arrival_fit(self,
                          classes: Optional[List[int]] = None,
                          distributions: Optional[List[Callable[
                              [Union[List[float], np.ndarray, pd.Series]],
                              Union[Hypererlang, scipy.stats.expon]]]] = None,
                          filename="inter_arrival_fit") -> List[HospitalSpecs]:
        """compute inter arrival fit distributions from data.

        :param classes: The classes to include, if empty include all.
        :param distributions: Callables which return fitted distributions to data.
        :param filename: Filename for plot.

        :return: A numpy array holding the distributions for each ward and class.
        If multiple distributions are given, a numpy.zero array will be returned.
        """

        if classes is None:
            if hasattr(self.analyser, "classes"):
                classes = self.analyser.classes
            else:
                classes = [0]

        if distributions is None:
            distributions = [fit_expon]

        arrivals = [
            np.zeros((len(self.analyser.wards), len(classes), 2), dtype="O")
            for _ in range(len(distributions))
        ]

        for j, origin in enumerate([EXTERNAL, INTERNAL, [INTERNAL, EXTERNAL]]):
            for ward in self.analyser.wards:
                for i, class_ in enumerate(classes):
                    qry = and_query(column_query(CURRENT_WARD, ward),
                                    column_query(CURRENT_CLASS, class_))
                    class_data = self.analyser.data.query(qry).dropna(
                        subset=[BEGIN, END])
                    class_data["Arrival"] = self.analyser.make_inter_arrival(
                        class_data, pre_ward=[origin])
                    if ward == "PACU":
                        class_data = drop_week_arrival(class_data, week=True)
                    arrival_data = class_data["Arrival"].dropna()
                    distribution_fits: List[Union[Hypererlang,
                                                  scipy.stats.expon]] = []
                    if not arrival_data.empty:
                        for k, distribution_ in enumerate(distributions):
                            distribution_fits.append(
                                distribution_(arrival_data))
                            if j in [0, 1]:
                                arrivals[k][self.analyser.wards_map[ward], i,
                                            j] = distribution_fits[0]
                        title = f"ward: {ward}, class: {int(class_)}, origin: {origin}"
                        plot_distribution_fit(arrival_data,
                                              distribution_fits,
                                              title=title)
                        d = ", ".join([d.name for d in distribution_fits])
                        filename_ = filename + f" - distributions[{d}] - ward[{ward}] - " \
                                               f"class[{int(class_)}] - origin[{origin}].pdf"
                        plt.savefig(self.output_dir.joinpath(filename_))
                        plt.close()

        self.hospital_specs.set_arrival(arrivals[0])
        hospital_specs = [
            self.hospital_specs.copy() for _ in range(len(distributions))
        ]
        for specs, arrival in zip(hospital_specs, arrivals):
            specs.set_arrival(arrival)

        return hospital_specs

    def service_fit(
            self,
            classes: Optional[List[int]] = None,
            distributions: Optional[List[
                Callable[[Union[List[float], np.ndarray, pd.Series]],
                         Union[Hypererlang, scipy.stats.expon]]]] = None,
            filename="service_fit",
    ) -> List[HospitalSpecs]:
        """Compute service fit distributions from data.

        :param classes: The classes to include, if empty include all.
        :param distributions: Callables which return fitted distributions to data.
        :param filename: The filename for plot saving.

        :return: A numpy array holding the distributions for each ward and class.
        If multiple distributions are given, a numpy.zero array will be returned.
        """

        if classes is None:
            if hasattr(self.analyser, "classes"):
                classes = self.analyser.classes
            else:
                classes = [0]

        if distributions is None:
            distributions = [fit_expon]

        services = [
            np.zeros((len(self.analyser.wards), len(classes)), dtype="O")
            for _ in range(len(distributions))
        ]

        self.analyser.make_service()

        self.logger.info(f"Modell for service.")

        for ward in self.analyser.wards:
            for i, class_ in enumerate(classes):
                qry = and_query(column_query(CURRENT_WARD, ward),
                                column_query(CURRENT_CLASS, class_))
                class_data = self.analyser.data.query(qry)
                service_data = class_data[SERVICE].dropna()
                distribution_fits: List[Union[Hypererlang,
                                              scipy.stats.expon]] = []
                if not service_data.empty:
                    for j, distribution_ in enumerate(distributions):
                        distribution_fit = distribution_(service_data)
                        distribution_fits.append(distribution_fit)

                        title = f"Ward: {ward}, Class: {int(class_)}"
                        plot_distribution_fit(service_data, [distribution_fit],
                                              title=title)
                        filename_ = filename.format(distribution_fit.name,
                                                    ward, int(class_))
                        plt.savefig(
                            self.output_dir.joinpath(f"{filename_}.pdf"))
                        plt.close()

                        services[j][self.analyser.wards_map[ward],
                                    i] = distribution_fit

                    title = f"ward: {ward}, class: {int(class_)}"
                    plot_distribution_fit(service_data,
                                          distribution_fits,
                                          title=title)
                    d = ", ".join([d.name for d in distribution_fits])
                    filename_ = filename + f" - distributions[{d}] - ward[{ward}] - " \
                                           f"class[{int(class_)}].pdf"
                    plt.savefig(self.output_dir.joinpath(filename_))
                    plt.close()

        self.hospital_specs.set_service(services[0])
        hospital_specs = [
            self.hospital_specs.copy() for _ in range(len(distributions))
        ]
        for specs, service in zip(hospital_specs, services):
            specs.set_service(service)

        return hospital_specs

    def routing(self) -> np.ndarray:
        """Computing routing from data.

        :return: Routing matrix.
        """

        assert hasattr(self.analyser, "classes")

        routing_ = np.zeros(
            (len(self.analyser.wards), len(self.analyser.classes),
             len(self.analyser.wards) + 1, len(self.analyser.classes)),
            dtype="float")

        data = self.analyser.data.copy()

        for ward1 in self.analyser.wards:
            for ward2 in self.analyser.wards:
                for c1, class1 in enumerate(self.analyser.classes):
                    for c2, class2 in enumerate(self.analyser.classes):
                        routing_[self.analyser.wards_map[ward1], c1,
                                 self.analyser.wards_map[ward2],
                                 c2] = self.analyser.compute_routing(
                                     data, ward1, ward2, class1, class2)

        routing_[:, :, len(self.analyser.wards),
                 0] = 1 - routing_.sum(axis=(2, 3))

        self.hospital_specs.routing = routing_

        return routing_

    def adjust_fits(self) -> HospitalSpecs:
        """Adjust arrival and routing.

        Assume that arrival, service and routing exist all in
        hospital_specs.
        """

        occupancy = np.array(self.analyser.occupancy).T

        self.hospital_specs.arrival = self.adjust_arrival(
            arrival=self.hospital_specs.arrival,
            capacities=self.hospital_specs.capacities,
            occupancy=occupancy)
        self.hospital_specs.routing = self.adjust_routing(
            routing=self.hospital_specs.routing,
            capacities=self.hospital_specs.capacities,
            holdings=self.hospital_specs.holdings,
            occupancy=occupancy)

        return self.hospital_specs.copy()

    @staticmethod
    def adjust_routing(routing: np.ndarray,
                       capacities: np.ndarray,
                       holdings: List[bool],
                       occupancy: np.ndarray,
                       logger: Optional[logging.Logger] = None) -> np.ndarray:
        """Adjust routing.

        :param routing: Routing to consider.
        :param capacities: Capacities to consider.
        :param holdings: If a ward holds (then do not adjust routing.
        :param occupancy: Occupancy to consider.
        :param logger: Logger to use.

        :return: Adjusted routing matrix.
        """

        if logger is None:
            logger = get_logger("adjust_routing",
                                LOGGING_DIR.joinpath("adjust_routing.log"))
        I = len(capacities)

        # adjust routing!
        routing = routing.copy()
        routing_ = routing.copy()
        for index, val in np.ndenumerate(routing):
            if (not holdings[index[0]]) and (index[2] != routing.shape[2] - 1):
                routing_[index] = val / (
                    1 - occupancy[index[2], capacities[index[2]]])

        routing_[:, :, I, 0] = 1 - routing_[:, :, :I, :].sum(axis=(2, 3))

        if np.any(routing_ < 0) or np.any(routing_ > 1):
            for index in np.ndindex(routing_.shape[:2]):
                if np.any(routing_[index] < 0) or np.any(routing_[index] > 1):
                    logger.warning(
                        f"Routing issue on index: {index}\n"
                        f"Re-adjusting the matrix:\n{routing_[index]}")
                    routing_[index] = np.maximum(routing_[index],
                                                 0) / np.maximum(
                                                     routing_[index], 0).sum()

        return routing_

    @staticmethod
    def adjust_arrival(arrival: np.ndarray, capacities: np.ndarray,
                       occupancy: np.ndarray) -> np.ndarray:
        """Adjust arrival distributions.

        :param arrival: Arrival distributions to consider.
        :param capacities: Capacities to consider.
        :param occupancy: Occupancy to consider.

        :return: Adjusted arrival.
        """

        arrival = arrival.copy()
        arrival_ = arrival.copy()
        for index, val in np.ndenumerate(arrival):
            if val != 0:
                arrival_[index] = scipy.stats.expon(
                    scale=val.mean() *
                    (1 - occupancy[index[0], capacities[index[0]]]))

        return arrival_


class Service:
    """Holds possible service information regarding different distributions.

    :param expon: Service exonential distributions.
    :param hypererlang: Service hypererlang distributions.
    """
    def __init__(self,
                 expon: Optional[np.ndarray] = None,
                 hypererlang: Optional[np.ndarray] = None):
        self.expon = expon
        self.hypererlang = hypererlang

    def save_dict(self) -> Dict[str, List[Any]]:
        """Make class details into dict for saving.

        :return: Dict for saving.
        """
        return {
            "expon": HospitalSpecs.service_to_list(service=self.expon),
            "hypererlang":
            HospitalSpecs.service_to_list(service=self.hypererlang)
        }

    @staticmethod
    def load_dict(arguments: Dict[str, Any]) -> "Service":
        """Create class from Dict with arguments and values in it.

        :param arguments: The dict containing the parameter-argument pairs.

        :return: Class instance.
        """
        arguments_ = {
            key: HospitalSpecs.service_from_list(service=args)
            for key, args in arguments.items()
        }
        return Service(**arguments_)


class WardModel:
    """Holds the ward model for different variants. This is basically intended
    to generate different models out of before made analysis.

    :param arrival: Computed arrival distributions.
    :param service: Computed service distributions.
    :param routing: Computed routings.
    :param capacity: Criginal capacity.
    :param logger: Logger to use.
    """
    def __init__(self,
                 name: str,
                 arrival: np.ndarray,
                 service: Service,
                 routing: np.ndarray,
                 occupancy: np.ndarray,
                 capacity: int = 1,
                 logger: Optional[logging.Logger] = None):
        self.name = name
        self.arrival = arrival
        self.service = service
        self.routing = routing
        self._occupancy = occupancy
        self.capacity = capacity
        self.logger = logger if logger is not None else get_logger(
            "WardModel", LOGGING_DIR.joinpath("ward_model.log"))

    def specs(
        self,
        capacity: Optional[int] = None,
        service_name: str = "expon",
        adjust_arrival: bool = True,
        adjust_internal_rate: float = 0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Create the specifications for the ward for the given parameters.

        :param capacity: Ward capacity to consider.
        :param service_name: Service distributions to use.
        :param adjust_arrival: If arrival should be adjusted via observed occupancy.
        :param adjust_internal_rate: Adjust internal arrival rate by given parameter.

        :return: Ward specifications.
        """
        if capacity is None:
            capacity = self.capacity
        arrival = self.arrival.copy()
        for i, class_arrival in enumerate(arrival[0]):
            class_rates = distribution_to_rate(class_arrival)
            class_rates[0] += (1 - adjust_internal_rate) * class_rates[1]
            class_rates[1] = adjust_internal_rate * class_rates[1]
            arrival[0, i] = rate_to_distribution(class_rates)
        if adjust_arrival:
            arrival[:, :, 0] = Modeller.adjust_arrival(
                arrival=arrival[:, :, 0],
                capacities=np.array([capacity]),
                occupancy=self.occupancy(capacity=capacity).reshape((1, -1)))

        service_ = getattr(self.service, service_name)
        if service_ is not None:
            service = service_.copy()
        else:
            raise NotImplementedError("Service does not exist.")

        return arrival, service, self.routing.copy(), self.occupancy(
            capacity=capacity)

    def occupancy(self, capacity: Optional[int] = None) -> np.ndarray:
        """Compute occupancy distributions for given capacity.

        :param capacity: Capacity to use.

        :return: Adjusted occupancy.
        """
        if capacity is None:
            capacity = self.capacity
        occ = self._occupancy[:capacity + 1].copy()
        occ[capacity] = self._occupancy[capacity:].sum()
        return occ

    def save_dict(self) -> Dict[str, Any]:
        """Save the parameters and values of the class as a dict.

        :return: Dict for saving purposes.
        """
        arguments = {
            "name": self.name,
            "arrival": HospitalSpecs.arrival_to_list(arrival=self.arrival),
            "service": self.service.save_dict(),
            "routing": self.routing.tolist(),
            "occupancy": self._occupancy.tolist(),
            "capacity": self.capacity
        }

        return arguments

    @staticmethod
    def load_dict(arguments: Dict[str, Any]) -> "WardModel":
        """Load a class instance from saved dict.

        :param arguments: Arguments-value mapping for class instance.

        :return: Class instance for given parameters.
        """
        if not all(arg in arguments
                   for arg in ["name", "arrival", "service", "routing"]):
            raise KeyError
        return WardModel(name=arguments["name"],
                         arrival=HospitalSpecs.arrival_from_list(
                             arrival=arguments["arrival"]),
                         service=Service.load_dict(arguments["service"]),
                         routing=np.array(arguments["routing"]),
                         occupancy=np.array(arguments["occupancy"]),
                         capacity=arguments.get("capacity", 1))


class HospitalModel:
    """Model of te hospital. Basically holds all wards to consider and logi to
    compute different models from it.

    :param wards: Statons to consider.
    :param logger: Logger to use.
    """
    def __init__(self,
                 wards: List[WardModel],
                 logger: Optional[logging.Logger] = None):
        self.logger = logger if logger is not None else get_logger(
            "HospitalModel", LOGGING_DIR.joinpath("hospital_model.log"))
        self.wards = {ward.name: ward for ward in wards}

        self.ward_map_inv: Dict[int, str] = {
            i: ward.name
            for i, ward in enumerate(wards)
        }
        self.ward_map = {
            ward: index
            for index, ward in self.ward_map_inv.items()
        }

    @property
    def ward_names(self) -> List[str]:
        """Return used ward names (sorted).

        :return: Sorted ward names.
        """
        ward_names_ = list(self.wards)
        ward_names_.sort()
        return ward_names_

    def occupancy(self, **capacities) -> np.ndarray:
        """Compute occupancy distributions for given capacities.

        :param capacities: Capacities for each ward.
        :return: Adjusted occuppancy distributions.
        """
        ward_occupancies = []
        max_index = max(list(capacities.values())) + 1
        for ward_name in self.ward_names:
            if capacities.get(ward_name, None):
                ward_occupancies.append(
                    np.pad(
                        self.wards[ward_name].occupancy(capacities[ward_name]),
                        (0, max_index - capacities[ward_name])).reshape(1, -1))
        return np.concatenate(ward_occupancies)

    def occupancy_as_evaluator(self, **capacities) -> EvaluationResults:
        """Compute occupancy distributions and return them as
        EvaluationResults.

        :param capacities: Capacities for each ward.
        :return: Adjusted occuppancy distributions as EvaluationResults.
        """
        occupancy = self.occupancy(**capacities)
        hospital_specs = self.hospital_specs(capacities=capacities,
                                             adjust_arrival=False,
                                             adjust_routing=False)
        evaluation_results = EvaluationResults(hospital_specs=hospital_specs)
        evaluation_results.distributions.occupancy = occupancy
        evaluation_results.name = "Real observation"

        return evaluation_results

    def hospital_specs(self,
                       capacities: Optional[Dict[str, int]] = None,
                       service_name: str = "expon",
                       adjust_arrival: bool = True,
                       adjust_int_rates: Optional[Dict[str, float]] = None,
                       adjust_routing: bool = True) -> HospitalSpecs:
        """For the given parameters, compute a HospitalSpecs instance.

        :param capacities: Capacities to consider.
        :param service_name: Service to use.
        :param adjust_arrival: Adjust outside arrival from occupancy distributions.
        :param adjust_int_rates: Adjust internal arrival rate with given parameter.
        :param adjust_routing: Adjust internal routing by occupancy distributions.

        :return: Computed HospitalSpecs.
        """

        if capacities is None:
            capacities = {
                ward_name: ward.capacity
                for ward_name, ward in self.wards.items()
            }

        if len([ward for ward in capacities if ward in self.wards]) == 0:
            raise ValueError("No valid wards given.")

        if adjust_int_rates is None:
            adjust_int_rates = {ward: 0.0 for ward in capacities}

        specs = []
        for ward_name in self.ward_names:
            if capacities.get(ward_name, None):
                specs.append(self.wards[ward_name].specs(
                    capacity=capacities[ward_name],
                    service_name=service_name,
                    adjust_arrival=adjust_arrival,
                    adjust_internal_rate=adjust_int_rates.get(ward_name, 0)))
        # clean routing
        ward_indices = [
            self.ward_map[ward] for ward in capacities if ward in self.ward_map
        ]
        ward_indices.sort()
        arrival = np.concatenate([spec[0] for spec in specs])
        service = np.concatenate([spec[1] for spec in specs])
        routing = np.concatenate(
            [np.array([spec[2][:, ward_indices, :]]) for spec in specs])
        max_capacity = max(list(capacities.values())) + 1
        occupancy = np.concatenate([
            np.array([np.pad(spec[3], (0, max_capacity - spec[3].shape[0]))])
            for spec in specs
        ])

        # pylint: disable=unsubscriptable-object
        out_routing = 1 - routing.sum(axis=(2, 3), keepdims=True)
        out_routing = np.pad(out_routing,
                             pad_width=[
                                 (0, 0), (0, 0), (0, 0),
                                 (0, routing.shape[3] - out_routing.shape[3])
                             ])
        routing = np.concatenate([routing, out_routing], axis=2)

        if adjust_routing:
            routing = Modeller.adjust_routing(
                routing=routing,
                capacities=np.array(list(capacities.values())),
                holdings=[False] * len(capacities),
                occupancy=occupancy,
                logger=self.logger)

        ward_map = {
            index: ward
            for index, ward in self.ward_map_inv.items() if ward in capacities
        }
        capacities_ = np.zeros((len(ward_map)), dtype=int)
        for idx, ward in ward_map.items():
            capacities_[idx] = capacities[ward]

        return HospitalSpecs(capacities=capacities_,
                             arrival=arrival,
                             service=service,
                             routing=routing,
                             ward_map=ward_map)

    def get_model(self,
                  model: int = 1,
                  capacities: Optional[Dict[str, int]] = None,
                  service_name: str = "expon",
                  adjust_int_rates: Optional[Dict[str, float]] = None,
                  waitings: Optional[Dict[str, List[str]]] = None):
        """Create the model (models 1,2,3 are possible).

        :param model: Model to use.
        :param capacities: Capacities to use.
        :param service_name: Service distributions to use.
        :param adjust_int_rates: Internal arrrival adjust rates to use.
        :param waitings: Waiting map to use.

        :return: Specific model HospitalSpecs.
        """
        if model not in [1, 2, 3]:
            raise ValueError(
                f"Model should be in [1,2,3]. Provided model: {model}.")

        adjust_arrival = model in [1, 2, 3]

        adjust_int_rates = None if model in [1, 2] else adjust_int_rates

        adjust_routing = model == 1

        hospital_specs = self.hospital_specs(capacities=capacities,
                                             service_name=service_name,
                                             adjust_arrival=adjust_arrival,
                                             adjust_int_rates=adjust_int_rates,
                                             adjust_routing=adjust_routing)

        if model in [2, 3]:
            hospital_specs.set_holdings(
                **{ward: True
                   for ward in hospital_specs.ward_map_inv})

        if waitings is not None:
            hospital_specs.set_waitings(**waitings)
        return hospital_specs

    def save_dict(self) -> Dict[str, List[Any]]:
        """Save class instance to dict.

        :return: Argument value mapping.
        """
        arguments = {
            "wards": [ward.save_dict() for ward in self.wards.values()]
        }
        return arguments

    def save(self, filepath: Path = Path()) -> None:
        """Save the HospitalModel to json via save_dict.

        :param filepath: Path to save to.
        """

        arguments_dict = self.save_dict()
        with open(filepath, "w") as f:
            json.dump(arguments_dict, f)

    @staticmethod
    def load_dict(arguments: Dict[str, Any]) -> "HospitalModel":
        """Create class instance from dict.

        :param arguments: Arguments value mapping for class call.

        :return: Class instance.
        """
        wards = [
            WardModel.load_dict(ward_arguments)
            for ward_arguments in arguments["wards"]
        ]

        return HospitalModel(wards=wards)

    @staticmethod
    def load(filepath: Path = Path(),
             logger: Optional[logging.Logger] = None) -> "HospitalModel":
        """Load the HospitalModel from json via load_dict.

        :param filepath: Filepath to load from.
        :param logger: Logger to use.

        :return: Loaded instance of self.
        """
        if logger is None:
            logger = get_logger("hospital_model",
                                LOGGING_DIR.joinpath("hospital_model.log"))
        with open(filepath, "r") as f:
            arguments = json.load(f)
        arguments["logger"] = logger
        return HospitalModel.load_dict(arguments)


def make_hospital_model(
        filepath: Path,
        wards: List[str],
        capacities: List[int],
        startdate: datetime = datetime(2019, 1, 1),
        enddate: datetime = datetime(2019, 12, 1),
        cart_specs: Optional[CartSpecs] = None,
        hypererlang_specs: Optional[HypererlangSpecs] = None,
        adjust_pacu_occupancy: bool = True,
        output_dir: Path = OUTPUT_DIR,
        logger: Optional[logging.Logger] = None) -> List[HospitalModel]:
    """Make hospital model.

    :param filepath: Path to excel file to analyse.
    :param wards: Wards to consider and their respective capacities.
    :param capacities: Capacities for wards.
    :param startdate: Startdate to use.
    :param enddate: Enddate to use.
    :param cart_specs: Specifications for CART analysis.
    :param hypererlang_specs: Specifications for hypererlang fit.
    :param adjust_pacu_occupancy: Adjust pacu occupancy because of weekends.
    :param output_dir: Output_dir to use for plots.
    :param logger: Logger to use for logging.

    :return: Created HospitalModels.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = output_dir.joinpath("Modelling - " + timestamp)
    output_dir.mkdir()

    if logger is None:
        logger = get_logger(__file__,
                            file_path=output_dir.joinpath(
                                f"{Path(__file__).resolve().stem}.log"))

    if hypererlang_specs is None:
        hypererlang_specs = HypererlangSpecs()

    if cart_specs is None:
        cart_specs = CartSpecs(wards=wards)

    preprocessor = adjust_data(filepath=filepath,
                               wards=wards,
                               startdate=startdate,
                               enddate=enddate,
                               keep_internal=True,
                               logger=logger)

    analyser = analyse(preprocessor=preprocessor,
                       wards=wards,
                       capacities=capacities,
                       output_dir=output_dir,
                       adjust_pacu_occupancy=adjust_pacu_occupancy,
                       logger=logger)
    plt.close()

    hospital_models: List[HospitalModel] = []
    for cart in [False, True]:
        modeller = Modeller(analyser=analyser,
                            logger=logger,
                            output_dir=output_dir)

        c = "multiple classes" if cart else "one class"
        modeller.logger.info(f"Build model for {c}.")

        modeller = model_class_arrival_routing(modeller=modeller,
                                               make_cart=cart,
                                               cart_specs=cart_specs)
        plt.close()

        hospital_specs_service = modeller.service_fit(
            distributions=[
                fit_expon,
                lambda x: fit_hypererlang(x, specs=hypererlang_specs)
            ],
            filename=f"service_fit - cart[{cart}]")
        plt.close()

        ward_models = []
        for ward in wards:
            ward_index = hospital_specs_service[0].ward_map_inv[ward]
            arrival = hospital_specs_service[0].arrival[ward_index:ward_index +
                                                        1]
            service = Service(
                expon=hospital_specs_service[0].service[ward_index:ward_index +
                                                        1],
                hypererlang=hospital_specs_service[1].
                service[ward_index:ward_index + 1])
            occupancy = np.array(modeller.analyser.occupancies[0][ward])
            routing = hospital_specs_service[0].routing[ward_index, :, :-1, :]
            ward_models.append(
                WardModel(name=ward,
                          arrival=arrival,
                          service=service,
                          routing=routing,
                          occupancy=occupancy))
            plt.close()

        hospital_model = HospitalModel(wards=ward_models)
        filename = f"HospitalModel - cart[{cart}].json"
        hospital_model.save(filepath=output_dir.joinpath(filename))

        modeller.logger.info(
            f"Model for {c} saved in {output_dir.joinpath(filename)}.")

        hospital_models.append(hospital_model)
    return hospital_models


def model_class_arrival_routing(
        modeller: Modeller,
        make_cart: bool = False,
        cart_specs: Optional[CartSpecs] = None,
        output_dir: Path = OUTPUT_DIR,
) -> Modeller:
    """Modell cart classes, arrival and routing.

    :param modeller: Modeller instance to use.
    :param make_cart: Make cart classes.
    :param cart_specs: Specifications for cart.
    :param output_dir: Output_dir to use for plots.

    :return: Modeller instance with analysed information.
    """
    if make_cart:
        if cart_specs is None:
            cart_specs = CartSpecs(wards=modeller.analyser.wards)
        modeller.logger.info("Make cart based on {}.".format(CART_COLUMNS))
        _, graphs = modeller.make_cart(cart_specs)

        # pylint: disable=broad-except
        try:
            for ward, graph in graphs:
                filename = f"decision_tree - ward[{ward}].gv"
                graph.render(output_dir.joinpath(filename))
        except Exception:
            pass

    modeller.logger.info("Classes: {}.".format(modeller.analyser.classes))

    modeller.logger.info("Make routing")
    modeller.routing()

    modeller.logger.info("Make inter-arrival")
    modeller.logger.info("Inter-arrival fits")
    filename = f"inter_arrival_fit - cart[{make_cart}]"
    modeller.inter_arrival_fit(distributions=[fit_expon], filename=filename)

    return modeller
