import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Sequence, Tuple, Union, List


def draw_line(
    data: list[ tuple[list[float], str] ],
    x_name: str,
    log_path: str,
    title: str = 'Comparison',
    y_name: str = 'Accuracy',
    x_ticks: list[str] = None,
    hide_ticks: bool = False
):
    plt.figure(figsize=(10, 6))

    # Define styles
    highlight_colors = ['blue', 'green', 'purple','red']  # Highlight colors for the first three lines
    weaken_color = 'gray'  # Color for weaker lines
    highlight_linewidth = 2.0
    weaken_linewidth = 0.5
    highlight_markers = ['o', 's', '^', 'p']
    weaken_markers = ['d', 'x', '*']  # Additional markers for weaker lines

    # Plot each data series with customized styles
    for idx, (values, label) in enumerate(data):
        if idx < 3:
            # Highlight the first three lines
            color = highlight_colors[idx]
            linewidth = highlight_linewidth
            marker = highlight_markers[idx]
            linestyle = '-'
        else:
            # Weaken the remaining lines
            color = weaken_color
            linewidth = weaken_linewidth
            marker = weaken_markers[(idx - 3) % len(weaken_markers)]
            linestyle = '--'
        
        plt.plot(range(len(values)), values, label=label, marker=marker, linestyle=linestyle, linewidth=linewidth, color=color)


    # Customize the plot
    plt.title(title, fontsize=16)
    plt.xlabel(x_name, fontsize=14)
    plt.ylabel(y_name, fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)

    if x_ticks:
        plt.xticks(ticks=range(len(x_ticks)), labels=x_ticks, fontsize=12)
    if hide_ticks:
        plt.gca().set_xticks([]) 
    
    # Save the plot
    file_path = f"{log_path}/{title}.png"
    plt.tight_layout()
    plt.savefig(file_path, dpi=200)
    plt.close()




def draw_line_chart(
    data: list[tuple[list[float], list[float], str]],
    x_name: str | None = None,
    log_path: str | None = None,
    title: str | None = None,
    y_name: str | None = None,
    x_ticks: list[str] | None = None,
    y_ticks: list[float] | None = None,
    show_ticks: bool = True,
    colors: list[str] | None = None,
    linestyles: list[str] | None = None,
    show_grid: bool = True,
    show_legend: bool = True,
    # --- new convenience knobs ---
    figsize: tuple[float, float] = (3.3, 2.2),   # physical size when tiled
    dpi: int = 300,                              # ensures sharp fonts/lines
    font_size: int = 8,                          # base font size ─ adjusts all text
    line_width: float = 1.8,                     # main line thickness
    marker_size: int = 5,                        # marker diameter
    tick_size: int = 7,                          # tick-label font
):
    """
    Draw a multi-series line chart with error bars and save it.

    The extra keyword arguments (figsize, dpi, …) are tuned so that
    six plots placed in a 2×3 grid remain readable after scaling.
    """
    if not data:
        raise ValueError("data must contain at least one series")

    n_points = len(data[0][0])
    for mean, std, _ in data:
        if len(mean) != n_points or len(std) != n_points:
            raise ValueError("all series must share the same length")

    # ------------------------------------------------------------------
    # global typographic tuning
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "figure.dpi": dpi,
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": font_size,
        "pdf.fonttype": 42,           # keep text as text in PDFs
        "svg.fonttype": "none",

        # --- new lines (Times New Roman) ---
        "font.family": "serif",               # use a serif font family
        "font.serif": ["Times New Roman"],    # and pick Times New Roman
        "mathtext.fontset": "stix",           # STIX imitates Times for math
    })

    # x-positions
    x = np.arange(1, n_points + 1)

    # fallback styling
    if colors is None:
        colors = [None] * len(data)
    if linestyles is None:
        linestyles = ["-"] * len(data)
    if len(colors) < len(data):
        colors += [None] * (len(data) - len(colors))
    if len(linestyles) < len(data):
        linestyles += ["-"] * (len(linestyles) - len(linestyles))

    fig, ax = plt.subplots(figsize=figsize)

    for (mean, std, label), c, ls in zip(data, colors, linestyles):
        ax.errorbar(
            x,
            mean,
            yerr=std,
            label=label,
            color=c,
            linestyle=ls,
            marker="o",
            markerfacecolor=c,        # filled markers stand out when shrunk
            linewidth=line_width,
            markersize=marker_size,
            capsize=2,
            elinewidth=line_width * 0.6,
            alpha=0.95,
        )

    # labels & title
    if x_name:
        ax.set_xlabel(x_name)
    if y_name:
        ax.set_ylabel(y_name)
    if title:
        ax.set_title(title, pad=8)

    # custom tick labels
    if x_ticks is not None:
        if len(x_ticks) != n_points:
            raise ValueError("x_ticks length must equal data length")
        ax.set_xticks(x)
        ax.set_xticklabels(x_ticks)

    if y_ticks is not None:
        ax.set_yticks(y_ticks)

    # grid
    if show_grid:
        ax.grid(axis="both", linestyle="--", linewidth=0.6, color="#cdcdcd")
        ax.set_axisbelow(True)
    else:
        ax.grid(False)

    # ticks / spines
    ax.tick_params(axis='both', which='major', length=4, width=0.8)
    for side in ["left", "bottom"]:
        ax.spines[side].set_linewidth(1.0)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)

    if not show_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    if show_legend:
        ax.legend(frameon=False, handlelength=2.0, loc="best")

    plt.tight_layout(pad=0.2)

    # save
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    return fig



def draw_heatmap(
    data: Union[np.ndarray, Sequence[Sequence[float]]],
    *,                               # force keyword-style usage (consistent with your API)
    row_labels: List[str] | None = None,
    x_name: str | None = None,
    log_path: str | None = None,
    title: str | None = None,
    cbar_label: str | None = None,
    x_ticks: List[str] | None = None,    # custom tick labels (centers of layer windows)
    y_ticks: List[str] | None = None,    # overrides row_labels if provided
    cmap: str = "Greens",
    vmin: float | None = 0.0,
    vmax: float | None = None,
    show_ticks: bool = True,
    show_colorbar: bool = True,
    # --- convenience knobs (parity with draw_line_chart) ---
    figsize: Tuple[float, float] = (4, 3),
    dpi: int = 300,
    font_size: int = 8,
    tick_size: int = 7,
    # annotations: list of (text, x, y) in data-coords
    annotations: List[Tuple[str, float, float]] | None = None,
):
    """
    Draw a heat-map of Avg. Indirect Effect values and (optionally) save it.

    Parameters
    ----------
    data : 2-D array-like
        Shape = (#rows, #cols).  Each entry is an AIE value.
    row_labels : list[str], optional
        Labels for y-axis (one per matrix row).  Defaults to ["r0", …].
    x_name, title, cbar_label : str, optional
        Axis / title strings.  Use None to omit.
    x_ticks : list[str], optional
        Custom tick labels along x (length must equal #cols OR len(xticks)).
    y_ticks : list[str], optional
        If given, overrides row_labels on y-axis.
    cmap, vmin, vmax : Matplotlib colour-map and limits.
    show_ticks, show_colorbar : bool
        Toggle tick labels / colour-bar.
    figsize, dpi, font_size, tick_size : visual styling shortcuts.
    annotations : list[tuple[str, float, float]], optional
        Draws text at (x, y) in data-space; useful for “early site”, etc.

    Returns
    -------
    matplotlib.figure.Figure | pathlib.Path
        Figure if `log_path` is None; otherwise the saved file path.
    """
    # ------------------------------------------------------------------
    # sanity & pre-flight
    # ------------------------------------------------------------------
    data = np.asarray(data, dtype=float)
    if data.ndim != 2:
        raise ValueError("`data` must be 2-D (rows × cols)")

    n_rows, n_cols = data.shape
    if row_labels is None:
        row_labels = [f"r{i}" for i in range(n_rows)]
    if len(row_labels) != n_rows:
        raise ValueError("row_labels length must equal #rows in `data`")

    # ------------------------------------------------------------------
    # global typography
    # ------------------------------------------------------------------
    plt.rcParams.update({
        "figure.dpi": dpi,
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": font_size,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",

        # --- new lines (Times New Roman) ---
        "font.family": "serif",               # use a serif font family
        "font.serif": ["Times New Roman"],    # and pick Times New Roman
        "mathtext.fontset": "stix",           # STIX imitates Times for math
    })

    # default colour limit
    if vmax is None:
        vmax = float(np.nanmax(data))

    fig, ax = plt.subplots(figsize=figsize)

    # ------------------------------------------------------------------
    # heat-map
    # ------------------------------------------------------------------
    pc = ax.pcolor(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    ax.invert_yaxis()                       # first row on top

    # y-axis labels
    ax.set_yticks(np.arange(n_rows) + 0.5)
    ax.set_yticklabels(y_ticks if y_ticks is not None else row_labels)

    # x-axis labels / ticks
    if x_ticks is not None:
        if len(x_ticks) != n_cols:
            raise ValueError("`x_ticks` length must equal #cols in `data`")
        ax.set_xticks(np.arange(n_cols) + 0.5)
        ax.set_xticklabels(x_ticks)
    if x_name:
        ax.set_xlabel(x_name)

    # axis title
    if title:
        ax.set_title(title, loc="left")

    # optional text annotations
    if annotations:
        for txt, x, y in annotations:
            ax.text(x, y, txt, ha="center", va="center", fontsize=font_size)

    # tick visibility
    if not show_ticks:
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    # colour-bar
    if show_colorbar:
        cbar = fig.colorbar(pc, ax=ax)
        if cbar_label:
            cbar.set_label(cbar_label)

    plt.tight_layout(pad=0.25)

    # ------------------------------------------------------------------
    # save or return
    # ------------------------------------------------------------------
    if log_path is not None:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        return path
    return fig