"""Provide some utilities."""

import heapq
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

from hoqunm.data_tools.base import OUTPUT_DIR

MODEL_COLORS = [
    cm.get_cmap("Blues")(0.8),
    cm.get_cmap("Greens")(0.8),
    cm.get_cmap("Purples")(0.8)
]

LOGGING_DIR = OUTPUT_DIR.joinpath("logging")
if not LOGGING_DIR.is_dir():
    LOGGING_DIR.mkdir()


class Heap:
    """A class which represents a heap with maximum length implemented the
    kinimum will be the root, only the minimal elements such  that no more than
    'length' are in hte heap will be kept.

    :param x: The data for the heap.
    :param length: The maximum length of the heap.
    """
    def __init__(self,
                 x: Optional[List[Any]] = None,
                 length: int = sys.maxsize):
        self.heap = x if x is not None else []
        heapq.heapify(self.heap)
        self.length = length

    def push(self, x: Any) -> None:
        """Push an element into the heap, if the heap is full, the element with
        the greatest value will be dropped.

        :param x: The element to push.
        """

        if len(self.heap) < self.length:
            heapq.heappush(self.heap, x)
        else:
            heapq.heappushpop(self.heap, x)

    def nlargest(self, n: int) -> List[Any]:
        """Return a list with the n largest elements.

        :param n: The number.
        :return: The n largest elements.
        """

        return heapq.nlargest(n, self.heap)

    def change_length(self, length: int):
        """Change the length of the heap. If the new length is lower, the
        greatest elements have to be dropped.

        :param length: The new length.
        """

        if self.length <= length:
            self.length = length
        else:
            self.heap = (self.nlargest(length))
            heapq.heapify(self.heap)

    def copy_to_list(self) -> List[Any]:
        """Create a copy of the list of heap.

        :return: The copied heap as a list.
        """

        return self.heap.copy()


def heatmap(data: np.ndarray,
            row_labels: Union[np.ndarray, List[str]],
            col_labels: Union[np.ndarray, List[str]],
            ax: Optional[plt.Axes] = None,
            cbar_kw: Optional[Dict[str, Any]] = None,
            cbarlabel: str = "",
            **kwargs: Any) -> Tuple[matplotlib.image.AxesImage, plt.colorbar]:
    """Create a heatmap from a numpy array and two lists of labels.

    :param data: A 2D numpy array of shape (N, M).
    :param row_labels: A list or array of length N with the labels for the rows.
    :param col_labels: A list or array of length M with the labels for the columns.
    :param ax: A `matplotlib.axes.Axes` instance to which the heatmap is plotted.
    If not provided, use current axes or create a new one.
    :param cbar_kw: A dictionary with arguments to `matplotlib.Figure.colorbar`.
    :param cbarlabel: The label for the colorbar.
    :param kwargs: All other arguments are forwarded to `imshow`.

    :return: Image and respective colorbar.
    """

    if not ax:
        ax = plt.gca()
    if cbar_kw is None:
        cbar_kw = dict()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(),
             rotation=-30,
             ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im: matplotlib.image.AxesImage,
                     data: Optional[Union[List[Any], np.ndarray]] = None,
                     valfmt: Any = "{x:.2f}",
                     textcolors: Optional[List[str]] = None,
                     threshold: Optional[float] = None,
                     **textkw: Any) -> List[matplotlib.text.Text]:
    """Annotate a heatmap.

    :param im: The AxesImage to be labeled.
    :param data: Data used to annotate.  If None, the image's data is used.
    :param valfmt: The format of the annotations inside the heatmap.  This should either
    use the string format method, e.g. "$ {x:.2f}", or be a `matplotlib.ticker.Formatter`.
    :param textcolors: A list or array of two color specifications.  The first is used for
    values below a threshold, the second for those above.
    :param threshold: Value in data units according to which the colors from textcolors are
    applied.  If None (the default) uses the middle of the colormap as separation.
    :param textkw: All other arguments are forwarded to each call to `text` used to create
    the text labels.

    :return: Textobject for annotation.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    data = np.asarray(data)

    if textcolors is None:
        textcolors = ["black", "white"]

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts: List[matplotlib.text.Text] = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def get_logger(name: str = "Logger",
               file_path: Optional[Path] = None) -> logging.Logger:
    """Create a logger with some special features.

    :param name: Name of the logger.
    :param file_path: File_path to save output to. If None, do not save to file.

    :return: Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s:%(name)s:%(message)s',
        '%Y-%m-%d %H:%M:%S',
    )

    if file_path is not None:
        fh = logging.FileHandler(file_path)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger
