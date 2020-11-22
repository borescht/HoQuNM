"""Create important distributions classes.

Especially provide the logic for a hypererlang distributions with data
fitting.
"""

import logging
import multiprocessing as mp
from itertools import combinations_with_replacement
from typing import Any, Dict, Generator, Iterable, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
from pynverse import inversefunc

from hoqunm.utils.utils import LOGGING_DIR, Heap, get_logger


class HypererlangSpecs:
    """Specifiactions for hyper-erlang fit.

    :param processors: Processors to use for multiprocessing.
    :param N: The sum of the length of the erlang distributions for state limitation.
    :param convergence_criteria: The convergence cirteria in each round.
    :param maximal_distributions: The maximla distributions in each step.
    """
    def __init__(self,
                 processors: int = mp.cpu_count() - 1,
                 N: int = 10,
                 convergence_criteria: Optional[List[float]] = None,
                 maximal_distributions: Optional[List[int]] = None):
        self.processors = processors
        self.N = N
        self.convergence_criteria = convergence_criteria if convergence_criteria is not None else [
            1e-4, 1e-6, 1e-8
        ]
        self.maximal_distributions = maximal_distributions if maximal_distributions is not None \
            else [50, 25, 1]

        if not len(self.convergence_criteria) == len(
                self.maximal_distributions):
            raise AttributeError(
                "Length of convergence criteria and maximal distributions do not match."
            )
        if self.N <= 0:
            raise ValueError(f"N has to be larger then 10. N is {N}")

    @staticmethod
    def load_dict(arguments: Dict[str, Any]) -> "HypererlangSpecs":
        """Create class from Dict with arguments and values in it.

        :param arguments: The dict containing the parameter-argument pairs.
        :return: Class instance.
        """
        return HypererlangSpecs(**arguments)


class HyperDistribution:
    """A class representing a hyper distributions of a current distributions
    type. for compatibility, the methods and attributes are similar to those of
    scipy.stats.rv_continuous.

    :param distribution: The distributions type.
    :param hyper: The hyper parameters.
    :param kwargs: The arguments needed for the distributions.
    Each will be in list style having the same shape as hyper.
    """
    def __init__(self, distribution: scipy.stats.rv_continuous,
                 hyper: Union[np.ndarray, List[float]],
                 **kwargs: Union[np.ndarray, List[float]]):
        self.dist = self  # for compatibility with scipy.stats
        self.distribution = distribution
        self.name = "hyper" + self.distribution.name
        self.hyper = np.asarray(hyper).reshape(-1)
        self.hyper = self.hyper / self.hyper.sum()
        kwargs = {
            key: np.asarray(arg).reshape(-1)
            for key, arg in kwargs.items()
        }
        self.kwargs = [{key: arg[i]
                        for key, arg in kwargs.items()}
                       for i in range(self.hyper.shape[0])]
        self.paramno = self.hyper.shape[0] * (1 + len(kwargs))

    def mean(self) -> float:
        """Return the mean of the distributions.

        :return: Mean of the distributions.
        """
        return float(
            np.sum([
                p * self.distribution.mean(**self.kwargs[i])
                for i, p in enumerate(self.hyper)
            ]))

    def var(self) -> np.float:
        """Return the variance of the distributions.

        :return: Variance of the distributions.
        """
        return float(
            np.sum([
                p * self.distribution.var(**self.kwargs[i])
                for i, p in enumerate(self.hyper)
            ]))

    def pdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """The pdf (probability density function) evaluated at x.

        :param x: x values, where the pdf should be evaluated.

        :return: Corresponding value of pdf at x.
        """
        return np.sum([
            p * self.distribution.pdf(x=x, **self.kwargs[i])
            for i, p in enumerate(self.hyper)
        ],
                      axis=0)

    def cdf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """The cdf (culmulative density function) evaluated at x.

        :param x: x values, where the pdf should be evaluated.

        :return: Corresponding value of cdf at x.
        """
        return np.sum([
            p * self.distribution.cdf(x=x, **self.kwargs[i])
            for i, p in enumerate(self.hyper)
        ],
                      axis=0)

    def ppf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """The ppf (percent point function - the inverse of the cdf) evaluated at x.
        Since this is not analytically available, compute it with an inversefunc module.

        :param x: x values, where the ppf should be evaluated.

        :return: Corresponding value of ppf at x.
        """
        return inversefunc(self.cdf)(x)

    def rvs(self, size: np.shape = None) -> Union[np.ndarray, float]:
        """A random value of the random variable.

        :param size: The size of the np.array with random values.
        :return: Random value(s).
        """

        index = np.random.choice(a=self.hyper.shape[0],
                                 p=self.hyper,
                                 size=size)
        out = np.zeros(size, dtype="float64")
        if size:
            for i, _ in enumerate(self.hyper):
                out[index == i] = self.distribution.rvs(**self.kwargs[i],
                                                        size=size)[index == i]
        else:
            out = self.distribution.rvs(**self.kwargs[index], size=size)
        return out

    def log_likelihood(self, x: Union[float, np.ndarray]) -> float:
        """Compute the log likelihood of the hyper_distribution w.r.t to
        observed data x.

        :param x: The observed data.

        :return: The log likelihood.
        """
        return np.sum(np.log(self.pdf(x)))

    def __str__(self) -> str:
        """A representation of the class very basic.

        :return: String of all attributes with respective values.
        """
        return str([(key, val) for key, val in self.__dict__.items()
                    if not callable(getattr(self, key))])


class Hypererlang(HyperDistribution):
    """A class representing a hyper erlang distributions this is in so far
    special, that we know an algorithm to fit a hypererlang distributions to
    data.

    :param hyper: The hyper parameters.
    :param kwargs: The arguments needed for the distributions.
    Each will be in list style having the same shape as hyper.
    """

    name = "hypererlang"

    def __init__(self,
                 hyper: List[float],
                 paramno: Optional[int] = None,
                 logger: Optional[logging.Logger] = None,
                 **kwargs: Union[np.ndarray, List[float]]):
        if kwargs.get("lambd"):
            lambd = np.asarray(kwargs.pop("lambd"))
            kwargs["scale"] = 1 / lambd

        super().__init__(scipy.stats.erlang, hyper, **kwargs)

        if paramno is not None:
            self.paramno = paramno

        self.lambd = 1 / np.asarray(kwargs["scale"]).reshape(-1)

        self.a = np.asarray(kwargs["a"]).reshape(-1)

        self.convergence_error = np.inf
        self.log_likelihood_fit = -np.inf

        self.logger = logger if logger is not None else get_logger(
            "hypererlang_distribution",
            LOGGING_DIR.joinpath("hypererlang_distribution.log"))

    def save_dict(self) -> Dict[str, Any]:
        """Create dictionary with argument value mapping.

        :return: Argument value mapping for class creation.
        """
        arguments = {"hyper": self.hyper.tolist(), "paramno": self.paramno}
        arguments.update({
            key: [arg[key] for arg in self.kwargs]
            for key in self.kwargs[-1]
        })
        arguments["a"] = self.a.tolist()
        return arguments

    @staticmethod
    def load_dict(arguments: Dict[str, Any]) -> "Hypererlang":
        """Create class instance from given dict.

        :param arguments: Arguments value mapping for class instance.

        :return: Class instance.
        """
        return Hypererlang(**arguments)

    def ppf(self, x: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """The ppf (percent point function - the inverse of the cdf) evaluated at x.
        Since this is not analytically available, compute it with an inversefunc module.

        It is known, that the domain are only positive floats, so provide domain!

        :param x: x values, where the ppf should be evaluated.

        :return: Corresponding value of ppf at x.
        """
        return inversefunc(self.cdf, domain=0)(x)

    def fit_lambd_hyper(self,
                        x: Union[List[float], np.ndarray],
                        convergence_criterion: float = 1e-6) -> "Hypererlang":
        """fit lambda and hyper parameters for given data until
        convergence_criterion is met.

        :param x: The data to fit the distributions to.
        :param convergence_criterion: The criterion which has to be met in order to quit fitting.

        :return: An instane of self.
        """

        x = np.asarray(x)

        log_a = np.array([
            np.sum(np.log(np.arange(1, a_, dtype="float64")))
            for i, a_ in enumerate(self.a)
        ])  # shape(m)

        x_ = x.reshape(-1, 1)  # shape(k, 1)

        runs = 0

        self.log_likelihood_fit = self.log_likelihood(x)

        while convergence_criterion <= self.convergence_error:
            p_ = self.lambd * np.exp((self.a - 1) * np.log(self.lambd * x_) -
                                     log_a - self.lambd * x_)  # shape(k, m)

            q_ = self.hyper * p_  # shape(k, m)

            q_ = q_ / q_.sum(axis=1).reshape(-1, 1)  # shape(k, m)

            self.hyper = (1 / x_.shape[0]) * q_.sum(axis=0)  # shape(m)

            self.lambd = self.a * q_.sum(axis=0) / np.sum(q_ * x_,
                                                          axis=0)  # shape(m)

            log_likelihood_fit = self.log_likelihood(x)

            self.convergence_error = abs(
                (log_likelihood_fit - self.log_likelihood_fit) /
                self.log_likelihood_fit)

            self.log_likelihood_fit = log_likelihood_fit

            runs += 1

        for i, kwarg_i in enumerate(self.kwargs):
            kwarg_i["scale"] = 1 / self.lambd[i]
        return self

    def fit(self,
            x: Union[List[float], np.ndarray],
            specs: Optional[HypererlangSpecs] = None) -> None:
        """Compute a hypererlang distributions which fits the data with EM
        algorithm according to "A novel approach for phasetype fitting", where
        the length of all erlang distributions is equal to N. The fitting is
        done in 3 respective rounds, each reducing the number of configurations
        under consideration while increasing the convergence_criterium floc is
        appears only for compatibility reasons with
        scipy.stats.rv_continuous.fit.

        Change the parameters on self!

        :param x: The data to fit to.
        :param specs: The specifications.
        """

        if specs is None:
            specs = HypererlangSpecs()
        convergence_criteria = np.asarray(specs.convergence_criteria)
        maximal_distributions = np.asarray(specs.maximal_distributions)

        hypererlangs: Iterable[Hypererlang] = self.iterate_hyp_erl(specs.N)

        heap = Heap()

        for i, convergence_criterion in enumerate(convergence_criteria):
            heap.change_length(maximal_distributions[i])

            if specs.processors > 1:
                pool = mp.Pool(processes=specs.processors)
                for hypererlang_ in hypererlangs:
                    # this gives all allowed values for r_m
                    pool.apply_async(hypererlang_.fit_lambd_hyper,
                                     args=(x, convergence_criterion),
                                     callback=heap.push,
                                     error_callback=self.error_callback)
                pool.close()
                pool.join()

            else:
                for hypererlang_ in hypererlangs:
                    # this gives all allowed values for r_m
                    heap.push(
                        hypererlang_.fit_lambd_hyper(x, convergence_criterion))

            hypererlangs = heap.copy_to_list()

        # heap[0] has the paramters we want, so copy them
        candidate = heap.nlargest(1)[0]
        for key, val in candidate.__dict__.items():
            if hasattr(candidate, key):
                setattr(self, key, val)

    @staticmethod
    def iterate_hyp_erl(N: int = 10) -> Generator["Hypererlang", None, None]:
        """Generate all combinations of hypererlang a parameters and yield the
        hypererlang such that the sum of the length of all erlang distributions
        is equal N.

        :param N: The sum of the length of the erlang distributions.

        :yield: Hypererlang distrbutions.
        """

        assert N > 0

        for i in combinations_with_replacement(list(range(N + 1)), N):
            if np.sum(i) == N:
                a = np.array(i).astype("int")
                a = a[a != 0]
                yield Hypererlang(hyper=[1 / a.shape[0]] * a.shape[0],
                                  a=a,
                                  lambd=[1] * a.shape[0],
                                  paramno=N * 3)

    def __lt__(self, other: "Hypererlang") -> bool:
        if hasattr(self, "log_likelihood_fit") and hasattr(
                other, "log_likelihood_fit"):
            return self.log_likelihood_fit < other.log_likelihood_fit
        else:
            raise ValueError

    def error_callback(self, error: BaseException) -> None:
        """Log error during multiprocessing.

        :param error: The error received.
        """
        self.logger.warning(error)


def fit_hypererlang(x: Union[List[float], np.ndarray, pd.Series],
                    specs: Optional[HypererlangSpecs] = None) -> Hypererlang:
    """Compute a hypererlang distributions which fits the data, where the
    length of all erlang distributions is equal to N. The fitting is done in k
    respective rounds, each reducing the number of configurations under
    consideration while increasing the convergence_criterium floc is appears
    only for compatibility reasons with scipy.stats.rv_continuous.fit.

    :param x: The data to fit to.
    :param specs: The specifications.

    :return: A hypererlang instance, fitted to x.
    """

    if specs is None:
        specs = HypererlangSpecs()

    fitted_hypererlang = Hypererlang(hyper=[1],
                                     a=1,
                                     lambd=1,
                                     paramno=specs.N * 3)
    fitted_hypererlang.fit(x, specs=specs)

    return fitted_hypererlang


def fit_expon(x: Union[List[float], np.ndarray, pd.Series]):
    """Fit exponential distributions to data.

    :param x: The data to fit to.

    :return: A scipy.stats.expon instance, fitted to x.
    """
    fitted_expon = scipy.stats.expon(*scipy.stats.expon.fit(x, floc=0))
    fitted_expon.paramno = len(fitted_expon.args) + len(fitted_expon.kwds) - 1
    fitted_expon.name = "exponential"
    return fitted_expon


def plot_distribution_fit(data: pd.Series,
                          distributions: List[Union[
                              Hypererlang, scipy.stats.rv_continuous]],
                          title: str = "<>") -> None:
    """Plot the distributions w.r.t to the data and some measurements to
    determine wether the distributions is a good or a bad fit.

    :param data: The data to which the distributions was fitted.
    :param distributions: The fitted distributions.
    :param title: Should include the ward and class from which the data came.
    """

    plot_num = len(distributions) + 1
    fig = plt.figure(figsize=(12, 4 * plot_num))
    dist_plot = fig.add_subplot(plot_num, 1, 1)
    bins = 50
    dist_plot.hist(data,
                   bins=bins,
                   density=True,
                   label=f"Histogram of the observed data with binsize={bins}")
    x_axis = np.arange(0, data.max(), 0.01)

    for i, distribution in enumerate(distributions):
        dist_plot.plot(x_axis,
                       distribution.pdf(x_axis),
                       label=f"Distribution: {distribution.dist.name}")
        prob_plot = fig.add_subplot(plot_num, 1, i + 2)
        scipy.stats.probplot(x=data, dist=distribution, plot=prob_plot)
        prob_plot.set_title(
            f"Probability plot with least squares fit for {title}, "
            f"distribution: {distribution.dist.name}")
        prob_plot.grid(axis="y")

    dist_plot.grid(axis="y")

    dist_plot.legend()
    dist_plot.set_title(title)

    fig.tight_layout()


def entropy_sum(q: np.ndarray, p: np.ndarray) -> float:
    """Build the relative entropy and sum over the axis.

    :param q: Distribution 1.
    :param p: Distribution 2.

    :return: The sum of the relative entropies.
    """
    return float(np.sum(scipy.stats.entropy(q, p)))


def entropy_max(q: np.ndarray, p: np.ndarray) -> float:
    """Build the relative entropy and max over the axis.

    :param q: Distribution 1.
    :param p: Distribution 2.

    :return: The maximum of the relative entropies.
    """
    return float(np.max(scipy.stats.entropy(q, p)))


def relative_distance(q: np.ndarray, p: np.ndarray) -> float:
    """Compute the maximal relative distance of two distributions.

    :param q: Distribution 1.
    :param p: Distribution 2.

    :return: The maximum of the distances.
    """

    dist = np.absolute(q - p) / q
    dist = np.nan_to_num(dist, 0)
    dist[dist == 1] = 0
    return np.max(dist)


def total_variation_distance(q: np.ndarray, p: np.ndarray) -> float:
    """Compute the total variation distance of two distributions q and p.

    :param q: Distribution 1.
    :param p: Distribution 2.

    :return: The total variation distance.
    """
    t_dist = np.sum(np.absolute(q - p)) / 2

    return t_dist


def chi2(data: Union[np.ndarray, pd.Series],
         distribution: Union[HyperDistribution, scipy.stats.rv_continuous],
         numbins: int = 0):
    """Compute chi2 test for distributions fit. Note that this is favorable for
    discrete random variables, but can be problematic for continuous variables.

    :param data: The observed data.
    :param distribution: The distributions which should be tested on goodness of fit.
    :param numbins: The number of bins to use.

    :return: The value of the chisquare distributions, the p value.
    """

    if not isinstance(distribution, HyperDistribution) and not hasattr(
            distribution, "paramno"):
        distribution.paramno = len(distribution.args) + len(
            distribution.kwds) - 1

    if numbins == 0:
        numbins = max(2 + distribution.paramno, len(data) // 50)
        if numbins > 50:
            numbins = 50

    f_obs, bin_edges = np.histogram(data, numbins, density=True)
    # there are minimum 2 bins
    f_exp = np.array([distribution.cdf(bin_edge) for bin_edge in bin_edges])
    f_exp[1:] -= f_exp[:-1]
    ddof = numbins - 1 - distribution.paramno

    chisq, p = scipy.stats.chisquare(f_obs, f_exp[1:], ddof)

    return chisq, p


def distribution_to_mean(distributions: np.ndarray) -> np.ndarray:
    """Compute the mean value to obtain the specific mean for exp-dist.

    :param distributions: The array of distributions which shall be converted.

    :return: An array of same shape, containing the repective means of the distributions.
    """
    out = np.zeros_like(distributions, dtype="float")
    for index, distribution in np.ndenumerate(distributions):
        if distribution != 0:
            out[index] = distribution.mean()
        else:
            out[index] = np.inf

    return out


def distribution_to_rate(distributions: np.ndarray) -> np.ndarray:
    """Compute the rate=1/mean value to obtain the specific rate for exp-dist.

    :param distributions: The array of distributions which shall be converted.

    :return: An array of same shape, containing the repective rates of the distributions.
    """

    out = distribution_to_mean(distributions)

    return 1 / out


def mean_to_distribution(means: np.ndarray) -> np.ndarray:
    """Take an array of means and converte them to an array with exponential
    distributions obtaining that mean.

    :param means: An array of means.

    :return: An array with the respective exponential distributions.
    """

    assert np.all(means >= 0)

    out = np.zeros_like(means, dtype="O")
    for index, mean in np.ndenumerate(means):
        if mean < np.inf:
            out[index] = scipy.stats.expon(scale=mean)

    return out


def rate_to_distribution(rates: np.ndarray) -> np.ndarray:
    """Take an array of rates and converte them to an array with exponential
    distributions, with mean=1/rate.

    :param rates: An array of rates.

    :return: An array with the respective exponential distributions.
    """

    assert np.all(rates >= 0)

    out = np.zeros_like(rates, dtype="O")
    for index, rate in np.ndenumerate(rates):
        if rate > 0:
            out[index] = scipy.stats.expon(scale=1 / rate)

    return out
