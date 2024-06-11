import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

# from cmcrameri import cm
# import scipy.interpolate as interp
# from matplotlib.colors import LinearSegmentedColormap

import grid_help as grh
import plot_help_MESA as phM

plt.rcParams.update(
    {
        "font.size": 22,
        "xtick.labelsize": "small",
        "ytick.labelsize": "small",
        "ytick.direction": "in",
        "xtick.direction": "in",  # tells matplotlib to plot the ticks inward
        "ytick.right": True,
        "xtick.top": True,  # tells matplotlib to plot the ticks also on the right and on the top
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,  # include minor ticks as well
        "xtick.major.width": 2,
        "ytick.major.width": 2,  # widht of major ticks
        "ytick.minor.width": 1.25,
        "xtick.minor.width": 1.25,
        # "font.family": "sans-serif",
        # "font.sans-serif": "Helvetica Neue",
        "pdf.fonttype": 42,
    }
)


figsize_onecolumn = (15, 10)
figsize_onecolumn_half = (7.5, 5)

dpi = 300


def pplot_burn_type(x, y1, y2, type, ax):
    if type < 1:
        return
    elif type == 1:
        ax.fill_between(x=x, y1=y1, y2=y2, ec=None, fc="yellow", alpha=0.5)
    elif type == 2:
        ax.fill_between(x=x, y1=y1, y2=y2, ec=None, fc="gold", lw=0, alpha=1)
    elif type == 3:
        ax.fill_between(x=x, y1=y1, y2=y2, ec=None, fc="orangered", lw=0, alpha=1)
    elif type > 3:
        ax.fill_between(x=x, y1=y1, y2=y2, ec=None, fc="firebrick", lw=0, alpha=1)


def plot_burn_type(x, y1, y2, type, ax):
    if type < 1:
        return

    fill_properties = {
        1: {"fc": "yellow", "alpha": 0.5},
        2: {"fc": "gold", "alpha": 1},
        3: {"fc": "orangered", "alpha": 1},
    }

    fill_property = fill_properties.get(type, {"fc": "firebrick", "alpha": 1})

    ax.fill_between(x=x, y1=y1, y2=y2, ec=None, lw=0, **fill_property)


def plot_mix_type(x, y1, y2, type, ax):
    if type < 1:
        return
    elif type == 1:
        ax.fill_between(
            x=x, y1=y1, y2=y2, ec="dimgrey", fc="None", hatch="/..\\", lw=0, alpha=1
        )
    elif type == 3:
        ax.fill_between(
            x=x,
            y1=y1,
            y2=y2,
            ec="dimgrey",
            fc="None",
            hatch=r"/+\/+\/+\/+\\",
            lw=0,
            alpha=1,
        )


def Figure1():
    fig, [ax, ax2] = plt.subplots(2, 1, figsize=(15, 10), dpi=300)

    # should be called somewhere else
    grid_directories = ["grid_Kippenhahn"]
    model_directories = grh.get_sorted_model_directories(grid_directories)
    model = model_directories[0]

    history_data = phM.get_hist_data(model)
    star_age = phM.get_data_array(history_data, "star_age")
    star_mass = phM.get_data_array(history_data, "star_mass")

    stepsize = 1
    # history = {"star_age": star_age[::stepsize], "star_mass": star_mass[::stepsize]}
    history = history_data

    # for i in range(1, 16):
    # data = np.loadtxt(f"source_data/2Msun_classic_history_burn_type_{i}.txt").T
    # history[f"burn_type_{i}"] = data[::stepsize]
    # data = np.loadtxt(f"source_data/2Msun_classic_history_mix_type_{i}.txt").T
    # history[f"mix_type_{i}"] = data[::stepsize]
    # data = np.loadtxt(f"source_data/2Msun_classic_history_burn_qtop_{i}.txt").T
    # history[f"burn_qtop_{i}"] = data[::stepsize]
    # data = np.loadtxt(f"source_data/2Msun_classic_history_mix_qtop_{i}.txt").T
    # history[f"mix_qtop_{i}"] = data[::stepsize]

    ts = len(history["star_age"])
    ax.plot(history["star_age"], history["star_mass"], "k-")
    rad = ax.fill_between(
        x=history["star_age"], y1=history["star_mass"], y2=0, color="lightgray", alpha=1
    )

    start = 0
    stop = 0
    for i in range(1, 16):
        type_name = f"burn_type_{i}"
        q_name = f"burn_qtop_{i}"
        q_name_old = f"burn_qtop_{i-1}"
        while stop != ts:
            type = int(history[type_name][start])
            stop = ts
            for k in range(start, ts):
                if history[type_name][k] != type or k == ts - 1:
                    if k == ts - 1:
                        stop = k + 1
                    else:
                        stop = k
                    if i == 1:
                        if stop - start > 2:
                            y1 = np.zeros(len(history["star_age"][start : stop + 1]))
                        else:
                            y1 = np.zeros(len(history["star_age"][start:stop]))
                    else:
                        y1 = (
                            history[q_name_old][start:stop]
                            * history["star_mass"][start:stop]
                        )
                        if k != ts - 1 and stop - start > 2:
                            extra_point = y1[-1] + (y1[-1] - y1[-2]) / (
                                history["star_age"][stop - 1]
                                - history["star_age"][stop - 2]
                            ) * (
                                history["star_age"][stop]
                                - history["star_age"][stop - 1]
                            )
                            y1 = np.append(y1, y1[-2])

                    y2 = history[q_name][start:stop] * history["star_mass"][start:stop]
                    if k != ts - 1 and stop - start > 2:
                        extra_point = y2[-1] + (y2[-1] - y2[-2]) / (
                            history["star_age"][stop - 1]
                            - history["star_age"][stop - 2]
                        ) * (history["star_age"][stop] - history["star_age"][stop - 1])
                        y2 = np.append(y2, y2[-2])
                    if stop - start > 2:
                        plot_burn_type(
                            history["star_age"][start : stop + 1], y1, y2, type, ax
                        )
                    else:
                        plot_burn_type(
                            history["star_age"][start:stop], y1, y2, type, ax
                        )

                    start = k
                    break
        start = 0
        stop = 0

    start = 0
    stop = 0
    for i in range(1, 16):
        type_name = f"mix_type_{i}"
        q_name = f"mix_qtop_{i}"
        q_name_old = f"mix_qtop_{i-1}"
        while stop != ts:
            type = int(history[type_name][start])
            stop = ts
            for k in range(start, ts):
                if history[type_name][k] != type or k == ts - 1:
                    if k == ts - 1:
                        stop = k + 1
                    else:
                        stop = k
                    if i == 1:
                        if stop - start > 2:
                            y1 = np.zeros(len(history["star_age"][start : stop + 1]))
                        else:
                            y1 = np.zeros(len(history["star_age"][start:stop]))
                    else:
                        y1 = (
                            history[q_name_old][start:stop]
                            * history["star_mass"][start:stop]
                        )
                        if k != ts - 1 and stop - start > 2:
                            extra_point = y1[-1] + (y1[-1] - y1[-2]) / (
                                history["star_age"][stop - 1]
                                - history["star_age"][stop - 2]
                            ) * (
                                history["star_age"][stop]
                                - history["star_age"][stop - 1]
                            )
                            y1 = np.append(y1, y1[-1])

                    y2 = history[q_name][start:stop] * history["star_mass"][start:stop]
                    if k != ts - 1 and stop - start > 2:
                        extra_point = y2[-1] + (y2[-1] - y2[-2]) / (
                            history["star_age"][stop - 1]
                            - history["star_age"][stop - 2]
                        ) * (history["star_age"][stop] - history["star_age"][stop - 1])
                        y2 = np.append(y2, y2[-1])
                    if stop - start > 2:
                        plot_mix_type(
                            history["star_age"][start : stop + 1], y1, y2, type, ax
                        )
                    else:
                        plot_mix_type(history["star_age"][start:stop], y1, y2, type, ax)
                    start = k
                    break
        start = 0
        stop = 0

    b1 = ax.fill_between(
        x=[0, 0], y1=[0, 0], y2=[0, 0], ec=None, fc="yellow", alpha=0.5
    )
    b2 = ax.fill_between(
        x=[0, 0], y1=[0, 0], y2=[0, 0], ec=None, fc="gold", lw=0, alpha=1
    )
    b3 = ax.fill_between(
        x=[0, 0], y1=[0, 0], y2=[0, 0], ec=None, fc="orangered", lw=0, alpha=1
    )
    b4 = ax.fill_between(
        x=[0, 0], y1=[0, 0], y2=[0, 0], ec=None, fc="firebrick", lw=0, alpha=1
    )

    conv = ax.fill_between(
        x=[-10, 0],
        y1=[-10, 0],
        y2=[0, 0],
        ec="dimgrey",
        fc="lightgray",
        hatch="/..\\",
        lw=0,
        alpha=1,
    )
    ov = ax.fill_between(
        x=[0, 0],
        y1=[0, 0],
        y2=[0, 0],
        ec="dimgrey",
        fc="lightgray",
        hatch=r"/+\/+\/+\/+\\",
        lw=0,
        alpha=1,
    )

    ax.set_xscale("log")
    ax.set_xlim(1e3, history["star_age"][-1])
    # ax.set_xlim(history['star_age'][0], history['star_age'][-1])
    ax.set_ylim(0, 1.1 * history["star_mass"][-1])
    ax.set_xlabel("star age (yr)")
    ax.set_ylabel(r"enclosed mass  ($M_\odot$)")

    ax.text(0.0125, 0.95, "a", ha="left", va="center", transform=ax.transAxes)
    ax = ax2

    # name = "2Msun_disk_mediated_28"
    # star_age = np.loadtxt(f"source_data/{name}_history_star_age.txt").T
    # star_mass = np.loadtxt(f"source_data/{name}_history_star_mass.txt").T
    stepsize = 7
    # history = {"star_age": star_age[1::stepsize], "star_mass": star_mass[1::stepsize]}

    # for i in range(1, 16):
    # data = np.loadtxt(f"source_data/{name}_history_burn_type_{i}.txt").T
    # history[f"burn_type_{i}"] = data[1::stepsize]
    # data = np.loadtxt(f"source_data/{name}_history_mix_type_{i}.txt").T
    # history[f"mix_type_{i}"] = data[1::stepsize]
    # data = np.loadtxt(f"source_data/{name}_history_burn_qtop_{i}.txt").T
    # history[f"burn_qtop_{i}"] = data[1::stepsize]
    # data = np.loadtxt(f"source_data/{name}_history_mix_qtop_{i}.txt").T
    # history[f"mix_qtop_{i}"] = data[1::stepsize]

    ts = len(history["star_age"])
    ax.plot(history["star_age"], history["star_mass"], "k-")
    rad = ax.fill_between(
        x=history["star_age"], y1=history["star_mass"], y2=0, color="lightgray", alpha=1
    )

    start = 0
    stop = 0
    for i in range(1, 16):
        type_name = f"burn_type_{i}"
        q_name = f"burn_qtop_{i}"
        q_name_old = f"burn_qtop_{i-1}"
        while stop != ts:
            type = int(history[type_name][start])
            stop = ts
            for k in range(start, ts):
                if history[type_name][k] != type or k == ts - 1:
                    if k == ts - 1:
                        stop = k + 1
                    else:
                        stop = k
                    if i == 1:
                        if stop - start > 2:
                            y1 = np.zeros(len(history["star_age"][start : stop + 1]))
                        else:
                            y1 = np.zeros(len(history["star_age"][start:stop]))
                    else:
                        y1 = (
                            history[q_name_old][start:stop]
                            * history["star_mass"][start:stop]
                        )
                        if k != ts - 1 and stop - start > 2:
                            extra_point = y1[-1] + (y1[-1] - y1[-2]) / (
                                history["star_age"][stop - 1]
                                - history["star_age"][stop - 2]
                            ) * (
                                history["star_age"][stop]
                                - history["star_age"][stop - 1]
                            )
                            y1 = np.append(y1, y1[-2])

                    y2 = history[q_name][start:stop] * history["star_mass"][start:stop]
                    if k != ts - 1 and stop - start > 2:
                        extra_point = y2[-1] + (y2[-1] - y2[-2]) / (
                            history["star_age"][stop - 1]
                            - history["star_age"][stop - 2]
                        ) * (history["star_age"][stop] - history["star_age"][stop - 1])
                        y2 = np.append(y2, y2[-2])
                    if stop - start > 2:
                        plot_burn_type(
                            history["star_age"][start : stop + 1], y1, y2, type, ax
                        )
                    else:
                        plot_burn_type(
                            history["star_age"][start:stop], y1, y2, type, ax
                        )

                    start = k
                    break
        start = 0
        stop = 0

    start = 0
    stop = 0
    for i in range(1, 16):
        type_name = f"mix_type_{i}"
        q_name = f"mix_qtop_{i}"
        q_name_old = f"mix_qtop_{i-1}"
        while stop != ts:
            type = int(history[type_name][start])
            stop = ts
            for k in range(start, ts):
                if history[type_name][k] != type or k == ts - 1:
                    if k == ts - 1:
                        stop = k + 1
                    else:
                        stop = k
                    if i == 1:
                        if stop - start > 2:
                            y1 = np.zeros(len(history["star_age"][start : stop + 1]))
                        else:
                            y1 = np.zeros(len(history["star_age"][start:stop]))
                    else:
                        y1 = (
                            history[q_name_old][start:stop]
                            * history["star_mass"][start:stop]
                        )
                        if k != ts - 1 and stop - start > 2:
                            extra_point = y1[-1] + (y1[-1] - y1[-2]) / (
                                history["star_age"][stop - 1]
                                - history["star_age"][stop - 2]
                            ) * (
                                history["star_age"][stop]
                                - history["star_age"][stop - 1]
                            )
                            y1 = np.append(y1, y1[-1])

                    y2 = history[q_name][start:stop] * history["star_mass"][start:stop]
                    if k != ts - 1 and stop - start > 2:
                        extra_point = y2[-1] + (y2[-1] - y2[-2]) / (
                            history["star_age"][stop - 1]
                            - history["star_age"][stop - 2]
                        ) * (history["star_age"][stop] - history["star_age"][stop - 1])
                        y2 = np.append(y2, y2[-1])
                    if stop - start > 2:
                        plot_mix_type(
                            history["star_age"][start : stop + 1], y1, y2, type, ax
                        )
                    else:
                        plot_mix_type(history["star_age"][start:stop], y1, y2, type, ax)
                    start = k
                    break
        start = 0
        stop = 0

    b1 = ax.fill_between(
        x=[0, 0], y1=[0, 0], y2=[0, 0], ec=None, fc="yellow", alpha=0.5
    )
    b2 = ax.fill_between(
        x=[0, 0], y1=[0, 0], y2=[0, 0], ec=None, fc="gold", lw=0, alpha=1
    )
    b3 = ax.fill_between(
        x=[0, 0], y1=[0, 0], y2=[0, 0], ec=None, fc="orangered", lw=0, alpha=1
    )
    b4 = ax.fill_between(
        x=[0, 0], y1=[0, 0], y2=[0, 0], ec=None, fc="firebrick", lw=0, alpha=1
    )

    conv = ax.fill_between(
        x=[-10, 0],
        y1=[-10, 0],
        y2=[0, 0],
        ec="dimgrey",
        fc="lightgray",
        hatch="/..\\",
        lw=0,
        alpha=1,
    )
    ov = ax.fill_between(
        x=[0, 0],
        y1=[0, 0],
        y2=[0, 0],
        ec="dimgrey",
        fc="lightgray",
        hatch=r"/+\/+\/+\/+\\",
        lw=0,
        alpha=1,
    )

    k = ax.legend(
        [
            b1,
            b2,
            b3,
            b4,
        ],
        [
            "$> 10$",
            "$> 10^2$",
            "$> 10^3$",
            "$> 10^4$",
        ],
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        fontsize="x-small",
        loc="upper left",
        bbox_to_anchor=(0, 0.75, 0, 0),
        title=r"Burning regions ( ${\rm erg}\,{\rm g}^{-1}\,{\rm s}^{-1}$)",
        title_fontsize="small",
        ncol=4,
        handlelength=3.5,
        labelspacing=1,
        frameon=False,
        columnspacing=1,
        handletextpad=-1.25,
    )

    l = ax.legend(
        [rad, conv, ov],
        [
            "radiative",
            "convective",
            "overshoot",
        ],
        numpoints=1,
        handler_map={tuple: HandlerTuple(ndivide=None)},
        fontsize="x-small",
        loc="upper left",
        bbox_to_anchor=(0, 0.95, 0, 0),
        title="Mixing regions                         ",
        title_fontsize="small",
        ncol=4,
        handlelength=3.5,
        labelspacing=1,
        frameon=False,
        columnspacing=1,
        handletextpad=-0.8,
    )
    for patch in l.get_patches():
        patch.set_height(20)
        patch.set_width(30)
        patch.set_y(-6)

    for patch in k.get_patches():
        patch.set_height(20)
        patch.set_width(30)
        patch.set_y(-6)

    ax.add_artist(k)

    ax.set_xscale("log")
    ax.set_xlim(1e3, history["star_age"][-1])
    ax.set_ylim(0, 1.1 * history["star_mass"][-1])
    ax.set_xlabel("star age (yr)")
    ax.set_ylabel(r"enclosed mass  ($M_\odot$)")

    ax.text(0.0125, 0.95, "b", ha="left", va="center", transform=ax.transAxes)

    plt.tight_layout()
    plt.savefig("images/Kippenhahn.pdf", bbox_inches=0)
    print("save Images/Kippenhahn.pdf")
    # plt.show()
    plt.close()


Figure1()
