from matplotlib import pyplot as plt
from typing import AnyStr, List, Union, Sequence, Tuple


def format_axes(
        axess: Union[plt.Axes, plt.Figure, List[plt.Axes]] = None,
        keep_box=False):

    if axess is None:
        axess = plt.gcf()

    if isinstance(axess, plt.Figure):
        axess = plt.gcf().get_axes()

    if not isinstance(axess, (list, tuple)):
        axess = [axess]

    for axes in axess:
        if not keep_box:
            axes.spines["top"].set_color("white")
            axes.spines["right"].set_color("white")

        axes.set_facecolor("white")


def format_axes_annotation(axess: Union[plt.Axes,
                                        plt.Figure,
                                        Sequence[plt.Axes]] = None,
                           font_size_labels=14,
                           font_size_titles=16):

    """
    format_axes_annotation applies default formatting to annotations
    on the axes.
    :param axess: Axes or an array of axes.
    :type axess: matplotlib axes or an array thereof
    :param font_size_labels: Font size for axes labels.
    :type font_size_labels: int or float
    :param font_size_titles: Font size for axes title.
    :type font_size_titles: int or float
    :rtype: None
    """

    if axess is None:
        axess = plt.gcf()

    if isinstance(axess, plt.Figure):
        axess = plt.gcf().get_axes()

    if not isinstance(axess, (list, tuple)):
        axess = [axess]

    for axes in axess:
        axes.xaxis.label.set_fontsize(font_size_labels)
        axes.yaxis.label.set_fontsize(font_size_labels)
        axes.title.set_fontsize(font_size_titles)


def apply_standard_formatting(figure: plt.Figure = None,
                              include_grid=True,
                              font_size_labels=14,
                              font_size_titles=16):
    if figure is None:
        figure = plt.gcf()

    for axes in figure.get_axes():
        format_axes_annotation(axes,
                               font_size_labels,
                               font_size_titles)
        format_axes(axes)

        if include_grid:
            axes.grid(linewidth=0.5)
