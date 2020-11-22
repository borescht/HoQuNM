"""Classes for a hospital-like queueing model. Classes are build like their
real counterparts.

Associate wards 0,...,N-1 as the N wards. Ward N is the outside.

Use SimPy: https://simpy.readthedocs.io/en/latest/.
This is a simulation oriented module for implementing
also queueing networks. The advantage is that it works with generators and
gives at the end a faster solutions then a direct approach.
"""

from inspect import signature
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import scipy.stats
import simpy as sm

from hoqunm.utils.distributions import (HyperDistribution, Hypererlang,
                                        distribution_to_rate,
                                        rate_to_distribution)


class Patient:
    """The patient with current ward and current class.

    :param ward: The ward, where the patient currently resides.
    :param pre_ward: The ward, where the patient has been before.
    :param clas: the Current class of the patient.
    :param pre_clas: The pre class of the patient.
    :param arrival: 0 for external arrival, 1 for internal arrival.
    """
    def __init__(self, ward: "Ward", pre_ward: "Ward", clas: int,
                 pre_clas: int, arrival: int):
        self.ward = ward
        self.pre_ward = pre_ward
        self.clas = clas
        self.pre_clas = pre_clas
        self.arrival = arrival
        self.waited_ward: Optional[Ward] = None


class Ward:
    """The ward holds patients and treates them.

    It has a limited amount of beds. It has some connected
    waiting wards It has to report statistics to the hospital.

    :param env: The SimPy environment.
    :param hospital: The corresponding Hospital class.
    :param capacity: The capacity (number of beds) of the ward.
    :param index: The index assigned to the ward by the hospital.
    :param name: Name of the ward.
    :param waiting_wards: The waiting wards assigned to the ward.
    :param holding: Indicates if the ward holds patients or not.
    """
    def __init__(self, env: sm.Environment, hospital: "Hospital",
                 capacity: int, index: int, name: str,
                 waiting_wards: List[int], holding: bool):
        self.env = env
        self.hospital = hospital
        self.beds = sm.Resource(env, capacity)
        self.capacity = capacity
        self.index = index
        self.name = name
        self.waiting_wards = waiting_wards
        self.holding = holding

    def _nurse_intern(
        self, patient: Patient
    ) -> Generator[Union[sm.resources.base.Put, sm.events.Timeout], None,
                   None]:
        """Nurse the patient directly in the ward.

        The ward took the patient and since it is full,
        s/he can be nursed in the ward directly. Do also the
        reporting.

        :param patient: The patient to be nursed.

        :yield: Outputs for simpy environment.
        """

        # we accept a new patient, so report the old state
        self.hospital.reporter.occ_report(self)

        t = get_random_time(self.hospital.specs.service[self.index,
                                                        patient.clas])

        # start a request
        with self.beds.request() as request:
            yield request
            yield self.env.timeout(t)

            # patient leaves now, so report the current state
            self.hospital.reporter.occ_report(self)

    def _nurse_with_holding(
        self, patient: Patient
    ) -> Generator[Union[sm.resources.base.Put, sm.events.Timeout], None,
                   None]:
        """The patient has holding behaviour.

        Since the ward is full s/he will stay in the preceding ward
        until capacity is available. The s/he can be nursed in the ward.

        :param patient: The patient to be nursed.

        :yield: Outputs for simpy environment.
        """

        t = get_random_time(self.hospital.specs.service[self.index,
                                                        patient.clas])

        if patient.waited_ward is not None:
            pre_ward = patient.waited_ward
        else:
            pre_ward = patient.pre_ward

        # start request
        # pylint: disable=unused-variable
        with self.beds.request() as request:
            with pre_ward.beds.request() as pre_ward_request:
                # report the state at the old ward
                self.hospital.reporter.occ_report(pre_ward)

                # first pre-ward request, because bed is available
                yield pre_ward_request
                yield request

                # report the state at the old ward
                self.hospital.reporter.occ_report(pre_ward)

            # now real service starts
            # we therefore accept the new patient now, so report the old state
            self.hospital.reporter.occ_report(self)
            yield self.env.timeout(t)

            # patient leaves now, so report the current state
            self.hospital.reporter.occ_report(self)

    def _nurse_with_waiting(
        self, patient: Patient, waiting_ward: int
    ) -> Generator[Union[sm.resources.base.Put, sm.events.Timeout,
                         sm.events.AnyOf], None, None]:
        """The patient has waiting behaviour.

        Since the ward is full and the patient has no holding behaviour
        but a waiting ward was found, the patient will stay there first
        until capacity is available.

        :param patient: The patient to be nursed.
        :param waiting_ward: The ward where the patient will wait.

        :yield: Outputs for simpy environment.
        """

        # we accept a new patient, so report the old state
        self.hospital.reporter.occ_report(self)

        # start a request
        with self.beds.request() as request:
            # safe current time for later
            now = self.env.now

            t = get_random_time(self.hospital.specs.service[self.index,
                                                            patient.clas])

            # patient waiting time as a request
            waiting_request = self.env.timeout(t)

            # the waiting ward will accept a new patient,
            # so report the old state
            self.hospital.reporter.wait_report(self, waiting_ward, 1)

            # start a request at the waiting room
            # (we know that there is available capcity)
            with self.hospital.wards[waiting_ward].beds.request(
            ) as waiting_ward_request:
                yield waiting_ward_request
                # either the patient is accepted at our ward or
                # his/her waiting time exceeds
                yield self.env.any_of([request, waiting_request])

                # the patient is leaving the waiting ward,
                # so we report the current state
                self.hospital.reporter.wait_report(self, waiting_ward, -1)

            # patient waiting time not exceeded, so s/he is admitted in the ward
            # we still have to nurse the patient
            if self.env.now < now + t:
                yield self.env.timeout(now + t - self.env.now)

            # else nursing time is finished and patient
            # gets routed to next ward

            # patient leaves now, so report the current state
            self.hospital.reporter.occ_report(self)

    def _make_waiting(
            self,
            patient: Patient) -> Generator[sm.events.Process, None, None]:
        """Look for an available waiting ward and initiate nursing process.

        The patient has no holding behaviour and the capacity is full.
        Therefore look in the available waiting wards if a free ward can
        be found. If yes, nurse the patient there first, until capacity
        becomes available.

        :param patient: The patient to be nursed.

        :yield: The nursing process of the patient.
        """

        # we need to know if we waited at the end
        waited = False

        # check all waiting rooms, keeping the right order
        for waiting_ward in self.waiting_wards:
            if self.hospital.wards[waiting_ward].has_capacity(patient=patient):
                # we found a free waiting room
                # we will have been waiting at the end
                waited = True

                yield self.env.process(
                    self._nurse_with_waiting(patient, waiting_ward))
                patient.waited_ward = self.hospital.wards[waiting_ward]
                # we found a ward, so we break
                break

        if not waited:
            # we did not found a waiting ward

            # patient leaves now, so report the current state
            self.hospital.reporter.rej_report(patient)

            # so s/he leaves
            patient.ward = self.hospital.outside

    def nurse(self,
              patient: Patient) -> Generator[sm.events.Process, None, None]:
        """The nursing process. Choose how the patient will be nursed.

        Take a patient and nurses him in the ward.
        Either, capacity is immediately available. Otherwise check for
        holding or waiting behaviour.
        Reports states to Hospital_Reporter!!

        :param patient: The patient to be nursed.

        :yield: The nursing process of the patient.
        """

        # a new patient arrived
        self.hospital.reporter.arr_count(patient)

        if self.has_capacity(patient):
            # we have available capacities, so nurse internally
            yield self.env.process(self._nurse_intern(patient))
            patient.waited_ward = None

        elif patient.pre_ward.index != self.hospital.specs.I and patient.pre_ward.holding:
            # the pre_ward holds, the patient therefore stays in the pre_ward
            # till service is available

            yield self.env.process(self._nurse_with_holding(patient))
            patient.waited_ward = None

        else:
            # check for a waiting room

            yield self.env.process(self._make_waiting(patient))

        patient.arrival = 0

    def has_capacity(self, patient) -> bool:
        """Check if the hospital has available capacity.

        :return: Bool indicating if capacity is available.
        """
        if patient.pre_ward.index == self.hospital.specs.I and patient.arrival == 1:
            return True
        return self.beds.count + len(self.beds.queue) < self.capacity


class ReporterMetrics:
    """A class containing all metrics for reporter class for the corresponding
    distributions.

    :param I: The number of wards.
    :param U: The number of classes.
    :param capacities: The capacities for each respective ward.
    """
    def __init__(self, I: int, U: int, capacities: np.ndarray):
        assert capacities.shape == (I, )
        self.I = I
        self.U = U
        self.capacities = capacities
        self.occupancy = np.zeros((I, capacities.max() + 1), dtype="float")
        self.occupancy_poisson = np.zeros((I, capacities.max() + 1),
                                          dtype="float")
        self.waiting_occupancy = np.zeros((I, I, capacities.max() + 1),
                                          dtype="float")
        self.rejection_full = np.zeros((I, U, I + 1, U, 2), dtype="int")
        self.arrival = np.zeros((I, U, I + 1, U, 2), dtype="int")


class ReporterDistributions(ReporterMetrics):
    """A class containing all distributions which for reporter creates.

    :param I: The number of wards.
    :param U: The number of classes.
    :param capacities: The capacities for each respective ward.
    """
    def __init__(self, I: int, U: int, capacities: np.ndarray):
        super().__init__(I, U, capacities)

        self.rejection = np.zeros((I, ))
        self.rejection_out = np.zeros((I, ))
        for key, value in self.__dict__.items():
            if isinstance(value, np.ndarray) and key != "capacities":
                setattr(self, key, value.astype("float"))

    def mean_occupancy(self) -> np.ndarray:
        """Compute mean occupancy per ward.

        :return: Mean occupancy per ward.
        """
        return np.sum(self.occupancy * np.arange(self.occupancy.shape[1]),
                      axis=1)

    def utilisation(self) -> np.ndarray:
        """Compute utilisation.

        :return: Computed utilisation.
        """
        return self.mean_occupancy() / self.capacities

    def full_occupancy(self) -> np.ndarray:
        """Compute the full occupancy per ward.

        :return: Full occupancy per ward.
        """
        out = np.zeros_like(self.capacities, dtype="float")
        for i, c in enumerate(self.capacities):
            out[i] = self.occupancy[i, int(c)]
        return out

    def profit(self, costs: np.ndarray = np.array(0.8)) -> np.ndarray:
        """Compute profit.

        :param costs: Costs to consider per ward.

        :return: Profit.
        """
        return self.profit_bed(costs=costs) * self.capacities

    def profit_bed(self, costs: np.ndarray = np.array(0.8)) -> np.ndarray:
        """Compute profit per bed.

        :param costs: Costs to consider per ward.

        :return: Profit per bed.
        """
        return self.utilisation() - costs

    def save_dict(self) -> Dict[str, Any]:
        """Create argument value mapping for instance saving.

        :return: Argument value mapping.
        """
        arguments: Dict[str, Any] = dict()
        for arg, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                arguments[arg] = val.tolist()
            else:
                arguments[arg] = val
        return arguments

    @staticmethod
    def load_dict(arguments: Dict[str, Any]) -> "ReporterDistributions":
        """Create instance from given dict.

        :param arguments: Argument value mapping.

        :return: Class instance.
        """
        arguments_: Dict[str, Any] = dict()
        for arg, obj in signature(ReporterDistributions).parameters.items():
            if arg in arguments:
                if obj.annotation == np.ndarray:
                    arguments_[arg] = np.array(arguments[arg])
                else:
                    arguments_[arg] = arguments[arg]

        out_distributions = ReporterDistributions(**arguments_)

        for key, val in out_distributions.__dict__.items():
            if key in arguments and key not in arguments_:
                if isinstance(val, np.ndarray):
                    setattr(out_distributions, key, np.array(arguments[key]))
                else:
                    setattr(out_distributions, key, arguments[key])
        return out_distributions

    def __truediv__(self, other: Union[int, float]) -> "ReporterDistributions":
        out = self.copy()
        out /= other

        return out

    def __itruediv__(self, other: Union[int,
                                        float]) -> "ReporterDistributions":
        if isinstance(other, (int, float)):
            capacities = self.capacities.copy()
            for key, value in self.__dict__.items():
                if isinstance(value, np.ndarray):
                    setattr(self, key, value.copy() / other)
            self.capacities = capacities
        else:
            raise ValueError

        return self

    def __mul__(self, other: Union[int, float]) -> "ReporterDistributions":
        out = self.copy()
        out *= other

        return out

    def __imul__(self, other: Union[int, float]) -> "ReporterDistributions":
        if isinstance(other, (int, float)):
            capacities = self.capacities.copy()
            for key, value in self.__dict__.items():
                if isinstance(value, np.ndarray):
                    setattr(self, key, value.copy() * other)
            self.capacities = capacities
        else:
            raise ValueError

        return self

    def __add__(self,
                other: "ReporterDistributions") -> "ReporterDistributions":
        out = self.copy()
        out += other

        return out

    def __iadd__(
            self,
            other: Union["ReporterDistributions"]) -> "ReporterDistributions":
        if isinstance(other, type(self)):
            capacities = self.capacities.copy()
            for key, value in self.__dict__.items():
                if isinstance(value, np.ndarray):
                    other_value = getattr(other, key)
                    value_pad = np.pad(
                        value,
                        [(0, max(other_value.shape[i] - value.shape[i], 0))
                         for i in range(len(value.shape))])
                    other_value_pad = np.pad(
                        other_value,
                        [(0, max(value.shape[i] - other_value.shape[i], 0))
                         for i in range(len(value.shape))])
                    setattr(self, key, value_pad + other_value_pad)
            self.capacities = capacities
        else:
            raise ValueError

        return self

    def copy(self):
        """Copy instance into new instance.

        :return: Copied instance.
        """
        out = ReporterDistributions(I=self.I,
                                    U=self.U,
                                    capacities=self.capacities.copy())
        for key, val in self.__dict__.items():
            if isinstance(val, np.ndarray):
                setattr(out, key, val.copy())
            else:
                setattr(out, key, val)
        return out


class Reporter(ReporterMetrics):
    """A class responsible for reporting all metrics.

    :param hospital: A hospital instance, where the reprter belongs.
    :param start: The starting value of reporting, important for returing the distributions.
    :param end: The end value of reporting, important for returing the distributions.
    """
    def __init__(self,
                 hospital: "Hospital",
                 start: float = 0,
                 end: float = float("inf")):
        self.hospital = hospital
        self.start = start
        self.end = end
        super().__init__(self.hospital.specs.I, self.hospital.specs.U,
                         self.hospital.specs.capacities)
        self.distributions = ReporterDistributions(
            self.hospital.specs.I, self.hospital.specs.U,
            self.hospital.specs.capacities)
        self.waiting_requests = np.zeros(
            (self.hospital.specs.I, self.hospital.specs.I), dtype="int")
        self.waiting_times = np.zeros(
            (self.hospital.specs.I, self.hospital.specs.I),
            dtype="float") + self.start
        self.last_changes = np.zeros_like(self.hospital.specs.capacities,
                                          dtype="float") + self.start

    def reset_utilities(self, new_start: float) -> None:
        """Create all utilities we need for reporting.

        This is to be thought to do in-between a simulation note that waiting_requests should not be
        reseted!

        :param new_start: The new starting value.
        """

        self.start = new_start

        super().__init__(self.hospital.specs.I, self.hospital.specs.U,
                         self.hospital.specs.capacities)
        self.distributions = ReporterDistributions(
            self.hospital.specs.I, self.hospital.specs.U,
            self.hospital.specs.capacities)

        # self.waiting_requests stays unchanged!!
        self.waiting_times = np.zeros(
            (self.hospital.specs.I, self.hospital.specs.I),
            dtype="float") + self.start
        self.last_changes = np.zeros_like(self.hospital.specs.capacities,
                                          dtype="float") + self.start

    def copy(self) -> "Reporter":
        """Create a new reporter with the same information.

        :return: Copied instance of Reporter.
        """

        reporter = Reporter(self.hospital, self.start)
        for key, value in self.__dict__.items():
            if not callable(value):
                setattr(reporter, key, value)

        return reporter

    def occ_report(self, ward: Ward) -> None:
        """Change the occupancy report for the given ward.

        :param ward: The ward under consideration.
        """
        self.occupancy[
            ward.index, ward.beds.
            count] += self.hospital.env.now - self.last_changes[ward.index]
        self.last_changes[ward.index] = self.hospital.env.now

    def occ_report_poisson(self) -> None:
        """Change the occupancy report with regard to poisson arrivals.

        It should suffice to report the occupancy only, when poisson
        arrivals arrive at a ward.
        """

        for ward in self.hospital.wards:
            self.occupancy_poisson[ward.index, ward.beds.count] += 1

    def wait_report(self, ward: Ward, waiting_ward: int, change: int) -> None:
        """Change the wait report.

        :param ward: The ward under consideration.
        :param waiting_ward: The waiting ward under consideration.
        :param change: Wether the patient arrives or leaves at the waiting ward.
        """

        self.waiting_occupancy[waiting_ward, ward.index, self.waiting_requests[
            waiting_ward,
            ward.index]] += self.hospital.env.now - self.waiting_times[
                waiting_ward, ward.index]
        self.waiting_times[waiting_ward, ward.index] = self.hospital.env.now

        self.waiting_requests[
            waiting_ward, ward.
            index] += change  # save the changings also in the initialisation phase!!

        self.occ_report(ward=self.hospital.wards[waiting_ward])

    def rej_report(self, patient: Patient) -> None:
        """Change the rejection report.

        :param patient: The patient which got rejected.
        """
        self.rejection_full[patient.ward.index, patient.clas,
                            patient.pre_ward.index, patient.pre_clas,
                            patient.arrival] += 1

    def arr_count(self, patient: Patient) -> None:
        """Count an arriving patient.

        :param patient: The patient which arrived.
        """
        self.arrival[patient.ward.index, patient.clas, patient.pre_ward.index,
                     patient.pre_clas, patient.arrival] += 1

        # check if patient is outside arrival (poisson) and act accordingly
        if patient.pre_ward.index == self.hospital.specs.I:
            self.occ_report_poisson()

    def end_reporting(self) -> None:
        """End the reporting.

        Occupancy and waiting have to be changed to the last state.
        Create the distributions.
        """
        if self.hospital.env.now > self.start:
            self.end = self.hospital.env.now
            for ward in self.hospital.wards:
                self.occ_report(ward)
                for waiting_ward in ward.waiting_wards:
                    self.wait_report(ward, waiting_ward, 0)

        # get all distributions
        self.distributions.occupancy = self.occupancy / (self.end - self.start)
        self.distributions.occupancy_poisson = self.occupancy_poisson / self.arrival[:, :,
                                                                                     -1, :, :].sum(
                                                                                     )
        self.distributions.waiting_occupancy = self.waiting_occupancy / (
            self.end - self.start)

        arrival = self.arrival.copy()
        arrival[arrival == 0] = 1
        self.distributions.rejection_full = self.rejection_full / arrival
        self.distributions.rejection = self.rejection_full.sum(
            axis=(1, 2, 3))[:, 0] / self.arrival.sum(axis=(1, 2, 3))[:, 0]

        arrival_ = self.arrival.sum(axis=(1, 3))
        arrival_[arrival_ == 0] = 1
        self.distributions.rejection_out = self.rejection_full.sum(
            axis=(1, 3))[:, -1, 0] / arrival_[:, -1, 0]

        self.distributions.arrival = self.arrival.copy()


class HospitalSpecs:
    """A class which holds all information corresponding to a hospital.

    :param capacities: The capacities for each ward.
    :param arrival: The inter arrival distributions for each ward and each class.
    :param service: The service distributions for each ward and each class.
    :param routing: The routing logic (i,u)->(j,v).
    :param waitings: The waiting wards for each wards.
    :param holdings: A list of bools, indicating if a ward holds patients or not.
    :param ward_map: A map, which maps ward names to their indices.
    :param class_map: A map, which maps class names to their indices.
    """
    def __init__(self,
                 capacities: np.ndarray,
                 arrival: np.ndarray = np.zeros((1, 1, 2)),
                 service: np.ndarray = np.zeros((1, 1)),
                 routing: np.ndarray = np.zeros((1, 1)),
                 waitings: Optional[Dict[int, List[int]]] = None,
                 holdings: Optional[List[bool]] = None,
                 ward_map: Optional[Dict[int, str]] = None,
                 class_map: Optional[Dict[int, str]] = None):

        self.capacities = capacities
        self.I = self.capacities.shape[0]
        self.U = 1

        if arrival.any():
            if arrival.shape[0] != self.capacities.shape[0]:
                raise ValueError(
                    f"Shapes do not match for arrival with shape {arrival.shape} "
                    f"and capacities with shape {self.capacities.shape}.")
            self.U = arrival.shape[1]
            self.arrival = arrival  # shape (I,U)
        # we want out arrival (0) and other arrival (1)
        if not arrival.shape[2] == 2:
            raise ValueError(
                f"Arrival shoud have shape (.,.,2). It has shape {arrival.shape}."
            )
        if service.any():
            if service.shape[0] != self.capacities.shape[0]:
                raise ValueError(
                    f"Shapes do not match for service with shape {service.shape} "
                    f"and capacities with shape {self.capacities.shape}.")
            self.U = service.shape[1]
            self.service = service  # shape (I,U)
        if routing.any():
            if (routing.shape[0], routing.shape[2]) != (
                    self.capacities.shape[0], self.capacities.shape[0] + 1):
                raise ValueError(
                    f"Shapes do not match for routing with shape {routing.shape} "
                    f"and capacities with shape {self.capacities.shape}.")
            self.U = routing.shape[1]
            self.routing = routing  # shape (I,U,I+1,U)
        if not self.U:
            self.U = 1
        if not arrival.any():
            self.arrival = np.zeros((self.I, self.U, 2), dtype="O")
        if not service.any():
            self.service = np.zeros((self.I, self.U), dtype="O")
        if not routing.any():
            self.routing = np.zeros((self.I, self.U, self.I + 1, self.U),
                                    dtype="float")
            self.routing[:, :, -1, :] = 1
        if waitings is not None:
            if len(waitings) != self.capacities.shape[0]:
                raise ValueError(
                    f"Shapes do not match for waitings with len {len(waitings)} "
                    f"and capacities with shape {self.capacities.shape}.")
            self.waitings = waitings  # the waiting room logic
        elif ward_map is not None:
            self.waitings = {i: [] for i in ward_map}
        else:
            self.waitings = {i: [] for i in range(self.capacities.shape[0])}
        if holdings is not None and len(holdings) > 0:
            if len(holdings) != self.capacities.shape[0]:
                raise ValueError(
                    f"Shapes do not match for holdings with len {len(holdings)} "
                    f"and capacities with shape {self.capacities.shape}.")
            self.holdings = holdings
        else:
            self.holdings = [False] * self.capacities.shape[0]
        if ward_map is not None:
            self.ward_map = ward_map
        else:
            self.ward_map = {
                i: str(i)
                for i in range(self.capacities.shape[0])
            }
        if class_map is not None:
            self.class_map = class_map
        else:
            self.class_map = {i: str(i) for i in range(self.arrival.shape[1])}
        self.arrival_rate = distribution_to_rate(self.arrival)
        self.service_rate = distribution_to_rate(self.service)

    @property
    def ward_map_inv(self) -> Dict[str, int]:
        """Make inverse ward map.

        :return: Inversed ward_map.
        """
        return {value: key for key, value in self.ward_map.items()}

    def set_holdings(self, **holdings: bool) -> None:
        """Set holdings on/off for given wards.

        :param holdings: Wether holdings is on (True) or off (False).
        """
        for ward, hold in holdings.items():
            if ward in self.ward_map_inv:
                self.holdings[self.ward_map_inv[ward]] = hold

    def copy(self) -> "HospitalSpecs":
        """Copy all information and create a new class instance.

        :return: New instance of self.
        """

        arguments = dict()

        for key in signature(HospitalSpecs).parameters:
            value = getattr(self, key)
            if hasattr(value, "copy"):
                arguments[key] = value.copy()
            else:
                arguments[key] = value

        hospital_specs = HospitalSpecs(**arguments)

        return hospital_specs

    @staticmethod
    def arrival_to_list(arrival: np.ndarray) -> List[Any]:
        """Create a list from arrival array.

        :param arrival: Arrival distributions.

        :return: Save_dict for distributions in list.
        """
        parameter_list = np.zeros_like(arrival).tolist()
        for index, arrival_ in np.ndenumerate(arrival):
            i, j, k = index
            if arrival_ != 0 and arrival_.dist.name == "expon":
                parameter_list[i][j][k] = {
                    "expon": {
                        "args": list(arrival_.args),
                        "kwds": arrival_.kwds
                    }
                }
            elif arrival_ != 0 and arrival_.dist.name == "hypererlang":
                parameter_list[i][j][k] = {"hypererlang": arrival_.save_dict()}
        return parameter_list

    @staticmethod
    def arrival_from_list(arrival: List[Any]) -> np.ndarray:
        """Create arrival array from given list with dicts for each
        distributions.

        :param arrival: The list with arguments for each distributions.

        :return: Arrival array.
        """
        out_array = np.zeros(
            (len(arrival), len(arrival[0]), len(arrival[0][0])), dtype="O")

        for index in np.ndindex(out_array.shape):
            i, j, k = index
            params = arrival[i][j][k]
            if params != 0 and "expon" in params:
                out_array[index] = scipy.stats.expon(*params["expon"]["args"],
                                                     **params["expon"]["kwds"])

            elif params != 0 and "hypererlang" in params:
                out_array[index] = Hypererlang.load_dict(arguments=params)

        return out_array

    @staticmethod
    def service_to_list(service: np.ndarray) -> List[Any]:
        """Create a list from service array.

        :param service: Arrival distributions.

        :return: Save_dict for distributions in list.
        """
        parameter_list = np.zeros_like(service).tolist()
        for index, service_ in np.ndenumerate(service):
            i, j = index
            if service_ != 0 and service_.dist.name == "expon":
                parameter_list[i][j] = {
                    "expon": {
                        "args": list(service_.args),
                        "kwds": service_.kwds
                    }
                }
            elif service_ != 0 and service_.dist.name == "hypererlang":
                parameter_list[i][j] = {"hypererlang": service_.save_dict()}
        return parameter_list

    @staticmethod
    def service_from_list(service: List[Any]) -> np.ndarray:
        """Create service array from given list with dicts for each
        distributions.

        :param service: The list with arguments for each distributions.

        :return: Service array.
        """
        out_array = np.zeros((len(service), len(service[0])), dtype="O")

        for index in np.ndindex(out_array.shape):
            i, j = index
            params = service[i][j]
            if params != 0 and "expon" in params:
                out_array[index] = scipy.stats.expon(*params["expon"]["args"],
                                                     **params["expon"]["kwds"])

            elif params != 0 and "hypererlang" in params:
                out_array[index] = Hypererlang.load_dict(
                    arguments=params["hypererlang"])

        return out_array

    def save_dict(self) -> Dict[str, Any]:
        """Create dictionary with argument value mapping.

        :return: Argument value mapping for class creation.
        """
        arguments: Dict[str, Any] = dict()
        for arg in signature(HospitalSpecs).parameters:
            val = getattr(self, arg)
            if isinstance(val, np.ndarray):
                arguments[arg] = val.tolist()
            else:
                arguments[arg] = val

        arguments["arrival"] = self.arrival_to_list(arrival=self.arrival)
        arguments["service"] = self.service_to_list(service=self.service)

        return arguments

    @staticmethod
    def load_dict(arguments) -> "HospitalSpecs":
        """Create class instance from given dict.

        :param arguments: Arguments value mapping for class instance.

        :return: Class instance.
        """
        arguments_ = dict()
        arguments_["arrival"] = HospitalSpecs.arrival_from_list(
            arrival=arguments["arrival"])
        arguments_["service"] = HospitalSpecs.service_from_list(
            service=arguments["service"])

        for arg, obj in signature(HospitalSpecs).parameters.items():
            if arg in arguments and arg not in arguments_:
                if obj.annotation == np.ndarray:
                    arguments_[arg] = np.array(arguments[arg])
                else:
                    arguments_[arg] = arguments[arg]
        arguments_["ward_map"] = {
            int(key): val
            for key, val in arguments_["ward_map"].items()
        }
        arguments_["waitings"] = {
            int(key): val
            for key, val in arguments_["waitings"].items()
        }

        return HospitalSpecs(**arguments_)

    def set_U(self, U: int) -> None:
        """Set new class number. This will delete all service, arrival and
        routing data.

        :param U: The new number of classes.
        """

        self.U = U
        self.arrival = np.zeros((self.I, self.U, 2), dtype="O")
        self.service = np.zeros((self.I, self.U), dtype="O")
        self.routing = np.zeros((self.I, self.U, self.I + 1, self.U),
                                dtype="float")
        self.routing[:, :, -1, :] = 1

    def set_service(self, service: np.ndarray) -> None:
        """Set new service.

        :param service: The service distributions for each ward and class.
        """

        if self.service.shape != service.shape:
            raise ValueError(
                f"Shapes do not match with old shpae {self.service.shape} "
                f"and new shape {service.shape}.")

        self.service = service
        self.service_rate = distribution_to_rate(self.service)

    def set_arrival(self, arrival: np.ndarray) -> None:
        """Set new arrival.

        :param arrival: The inter arrival distributions for each ward and class.
        """

        if self.arrival.shape != arrival.shape:
            raise ValueError(
                f"Shapes do not match with old shpae {self.arrival.shape} "
                f"and new shape {arrival.shape}.")

        self.arrival = arrival
        self.arrival_rate = distribution_to_rate(self.arrival)

    def set_waitings(self, **waiting_wards: List[str]) -> None:
        """Set the waiting map.

        :param waiting_wards: Associated waiting wards for ward.
        """
        if any(
                self.ward_map_inv.get(ward, None) is None
                for ward in waiting_wards):
            return
        for ward, waiting_wards_ in waiting_wards.items():
            ward_index = self.ward_map_inv[ward]
            waiting_wards_idx = [
                self.ward_map_inv[waiting_ward]
                for waiting_ward in waiting_wards_
            ]
            self.waitings[ward_index] = waiting_wards_idx

    def set_capacities(self, **capacities: int) -> None:
        """Set capacities.

        :param capacities: Associated capacities for wards.
        """
        for ward, capacity in capacities.items():
            if ward in self.ward_map_inv:
                self.capacities[int(self.ward_map_inv[ward])] = capacity

    def perturbate(self, scale=0.2) -> None:
        """Perturbate the arrival, service and routing. This assumes
        exponential service!

        :param scale: Scale to use for normal distributions.
        """

        self.set_arrival(
            rate_to_distribution(self.arrival_rate * scipy.stats.norm.rvs(
                loc=1, scale=scale, size=self.arrival_rate.shape)))
        self.set_service(
            rate_to_distribution(self.service_rate * scipy.stats.norm.rvs(
                loc=1, scale=scale, size=self.service_rate.shape)))
        self.routing[:, :, :self.I, :] *= scipy.stats.norm.rvs(
            loc=1, scale=scale, size=self.routing[:, :, :self.I, :].shape)
        for index, s in np.ndenumerate(
                self.routing[:, :, :self.I, :].sum(axis=(2, 3))):
            if s > 1:
                self.routing[index] /= s
        self.routing[:, :, self.I,
                     0] = 1 - self.routing[:, :, :self.I, :].sum(axis=(2, 3))

    @staticmethod
    def random(wards: Tuple[int, int] = (3, 10),
               classes: Tuple[int, int] = (1, 5),
               capacities: Tuple[int, int] = (5, 40),
               arrival: Tuple[float, float] = (1, 4),
               service: Tuple[float, float] = (0.1, 1),
               routing_zeros: float = 0.9) -> "HospitalSpecs":
        """Create a random instance of self, where arrival and service
        distributions are all exponential.

        :param wards: The interval for the number of wards
        :param classes: The interval for the number of classes.
        :param capacities: The interval for the respective capacities.
        :param arrival: The interval for the lambda parameter of the inter arrival distributions.
        :param service: The interval for the lambda parameter for the service distributions.
        :param routing_zeros: Randomly put zeros into routing matrix with this probability.

        :return: Hospital_specs.
        """

        ward_num = np.random.randint(wards[0], wards[1] + 1, dtype=int)

        class_num = np.random.randint(classes[0], classes[1] + 1, dtype=int)

        capacities = np.random.randint(capacities[0],
                                       capacities[1] + 1,
                                       size=(ward_num, ))

        arrival_ = np.array([[[
            scipy.stats.expon(scale=1 / (arrival[0] + np.random.random() *
                                         (arrival[1] - arrival[0]))), 0
        ] for _ in range(class_num)] for _ in range(ward_num)])

        service_ = np.array([[
            scipy.stats.expon(scale=1 / (service[0] + np.random.random() *
                                         (service[1] - service[0])))
            for _ in range(class_num)
        ] for _ in range(ward_num)])

        routing = np.random.random(size=(ward_num, class_num, ward_num + 1,
                                         class_num))
        for i in range(ward_num):
            routing[i, :, i, :] = 0

        # create random zeros
        mult = np.random.choice(a=[0, 1],
                                p=[1 - routing_zeros, routing_zeros],
                                size=(ward_num, class_num, ward_num + 1,
                                      class_num))
        routing = routing * mult
        routing = (routing.transpose(
            (2, 3, 0, 1)) / routing.sum(axis=(2, 3))).transpose((2, 3, 0, 1))

        return HospitalSpecs(capacities=capacities,
                             arrival=arrival_,
                             service=service_,
                             routing=routing)


class Hospital:
    """The hospital with different wards and patients flowing through it.

    The hospital holds all wards The hospital holds the arriving logic.
    The hospital holds the nursing logic The hospital holds the routing logic.
    The hospital saves reports to report statistics from each ward in Reporter
    class.

    :param env: The SimPy Environment.
    :param hospital_specs: The specifications for the hospital.
    """
    def __init__(self, env: sm.Environment, hospital_specs: HospitalSpecs):
        self.env = env
        self.specs = hospital_specs
        self.wards = [
            Ward(env=self.env,
                 hospital=self,
                 capacity=self.specs.capacities[i],
                 index=i,
                 name=self.specs.ward_map[i],
                 waiting_wards=self.specs.waitings.get(i, []),
                 holding=self.specs.holdings[i]) for i in range(self.specs.I)
        ]
        self.outside = Ward(env=self.env,
                            hospital=self,
                            capacity=1,
                            index=self.specs.I,
                            name="EXTERNAL",
                            waiting_wards=[],
                            holding=False)  # outside ward
        self.wards_ = self.wards + [self.outside]
        self.reporter = Reporter(self)

    def patient_flow(
            self,
            patient: Patient) -> Generator[sm.events.Process, None, None]:
        """Logic of the patient flowing through the hospital.

        :param patient: The patient which goes through the hospital.

        :yield: The flow of the patient through the process.
        """

        # as long as patient does not leave, self.I is outside!
        while patient.ward.index != self.specs.I:
            ward = patient.ward
            clas = patient.clas

            # the patient gets nursed in his ward
            yield self.env.process(ward.nurse(patient))

            if patient.ward.index != self.specs.I:
                # patient might have been rejected
                patient.pre_ward = ward
                patient.pre_clas = clas
                patient.ward = self.wards_[np.random.choice(
                    list(range(self.specs.I + 1)),
                    p=self.specs.routing[ward.index, clas].sum(axis=1))]

            if patient.ward.index != self.specs.I:
                # patient might have been routed outside
                patient.clas = np.random.choice(
                    list(range(self.specs.U)),
                    p=self.specs.routing[ward.index, clas, patient.ward.index]
                    / self.specs.routing[ward.index, clas,
                                         patient.ward.index].sum())

    def create_sources(self) -> None:
        """Creates a source for every ward and every class.

        Puts them in Simpy process.
        """

        for index, arr in np.ndenumerate(self.specs.arrival):
            if arr != 0:
                self.env.process(
                    self.source(ward=self.wards[index[0]],
                                clas=index[1],
                                pre_clas=index[1],
                                arrival_index=index[2],
                                arrival=arr))

    def source(
        self, ward: Ward, clas: int, pre_clas: int, arrival_index: int,
        arrival: Union[HyperDistribution, scipy.stats.rv_continuous]
    ) -> Generator[sm.events.Timeout, None, None]:
        """Creates patients with the logic from arrival (shape(I, U)).

        :param ward: The ward where the patients will be routed to.
        :param clas: The class of the patients when arriving at that ward.
        :param pre_clas: The pre class of the patient.
        :param arrival_index: 0 for external arrival, 1 for internal arrival.
        :param arrival: The inter arrival distributions.

        :yield: The arriving process of patients.
        """

        while True:
            yield self.env.timeout(get_random_time(distribution=arrival))

            patient = self.patient_flow(
                Patient(ward=ward,
                        pre_ward=self.outside,
                        clas=clas,
                        pre_clas=pre_clas,
                        arrival=arrival_index))
            self.env.process(patient)

    def setup(self, mode: int = 1) -> None:
        """Setup the initial hospital, therefore: create a initial state
        (random or empty). Create the ouside arriving logic.

        :param mode: Mode=0 is empty state (all to 0), mode=1 is random state.
        """

        self.create_state(mode=mode)
        self.create_sources()

    def create_state(self, mode: int = 1) -> None:
        """We create a state of the hospital.

        :param mode: Mode=0 is empty state (all to 0), mode=1 is random state.
        """

        if mode:
            # we create a random state
            # be careful which patients are allowed
            random_patients = self.rand_N_sum(
                self.specs.capacities, self.specs.arrival,
                self.specs.routing.transpose((2, 3, 0, 1)))

            for ward_clas, val in np.ndenumerate(random_patients):
                for _ in range(val):
                    patient = self.patient_flow(
                        Patient(self.wards[ward_clas[0]],
                                self.outside,
                                ward_clas[1],
                                ward_clas[1],
                                arrival=0))
                    self.env.process(patient)

    def run(self, end: float) -> None:
        """Run the hospital till end.

        :param end: The end of the run time.
        """

        if self.env.now < end:
            self.env.run(until=end)
            self.reporter.end_reporting()

    @staticmethod
    def rand_N_sum(capacities: np.ndarray, arrival: np.ndarray,
                   routing_to: np.ndarray) -> np.ndarray:
        """Give a random vector over types with sum not greater then N. Only
        take those patient classes which really can arrive at a ward.

        :param capacities: The ward capactities.
        :param arrival: The inter arrival distributions.
        :param routing_to: Where the patient should get routed to.

        :return: The occupancies at the wards.
        """

        random_patients = np.zeros_like(arrival)

        for ward, capacity in enumerate(capacities):
            # determine allowed classes first
            classes = []

            for index, val in enumerate(arrival[ward]):
                if val[0] != 0 or val[1] != 0 or np.any(routing_to[ward,
                                                                   index]):
                    # this class is arriving at the ward
                    classes.append(index)

            # create a random distributions of patients
            if len(classes) >= 1:
                for _ in range(np.random.randint(0, capacity + 2)):
                    random_patients[ward, np.random.choice(classes)] += 1

        return random_patients


def get_random_time(
    distribution: Union[HyperDistribution, scipy.stats.rv_continuous]
) -> float:
    """Get the random time for a given distributions.

    :param distribution: A distributions with rvs() method.

    :return: The random time.
    """
    if distribution != 0:
        return distribution.rvs()
    else:
        return float("inf")
