import numpy as np
import matplotlib.pyplot as plt

# from matplotlib.legend_handler import HandlerTuple

import grid_help as grh
import plot_help_MESA as phM


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


def plot_Kippenhahn(model, ax, first=True):
    history_data = phM.get_hist_data(model)
    star_age = history_data["star_age"]
    star_mass = history_data["star_mass"]
    ts = len(star_age)

    ax.plot(star_age, star_mass, "k-")
    ax.fill_between(x=star_age, y1=star_mass, color="lightgray", alpha=1)

    plot_intervals(
        history_data, star_age, "burn_type", "burn_qtop", plot_burn_type, ts, ax
    )
    plot_intervals(
        history_data, star_age, "mix_type", "mix_qtop", plot_mix_type, ts, ax
    )

    # setup_legend(fig, ax, star_age, star_mass)
    # rewrite to set letter based on position of ax
    letter_top_left = "a" if first else "b"
    setup_axes(ax, star_age, star_mass, letter_top_left)

    return ax


def plot_intervals(
    history_data, star_age, type_prefix, q_prefix, plot_function, ts, ax
):
    for i in range(1, 16):
        type_name = f"{type_prefix}_{i}"
        q_name = f"{q_prefix}_{i}"
        q_name_old = f"{q_prefix}_{i - 1}"
        start, stop = 0, 0

        while stop != ts:
            type = int(history_data[type_name][start])
            stop = ts
            for k in range(start, ts):
                if history_data[type_name][k] != type or k == ts - 1:
                    stop = k + 1 if k == ts - 1 else k

                    y1, y2 = calculate_y_values(
                        history_data, star_age, q_name, q_name_old, start, stop, i, ts
                    )
                    plot_function(
                        (
                            star_age[start : stop + 1]
                            if stop - start > 2
                            else star_age[start:stop]
                        ),
                        y1,
                        y2,
                        type,
                        ax,
                    )

                    start = k
                    break
        start, stop = 0, 0


def calculate_y_values(history, star_age, q_name, q_name_old, start, stop, i, ts):
    if i == 1:
        y1_len = (
            len(star_age[start : stop + 1])
            if stop - start > 2
            else len(star_age[start:stop])
        )
        y1 = np.zeros(y1_len)
    else:
        y1 = history[q_name_old][start:stop] * history["star_mass"][start:stop]
        if stop - start > 2 and stop < ts:
            y1 = np.append(y1, y1[-2])
    y2 = history[q_name][start:stop] * history["star_mass"][start:stop]
    if stop - start > 2 and stop < ts:
        y2 = np.append(y2, y2[-2])
    return y1, y2


def setup_plot_elements(ax, star_age, star_mass):
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
        hatch=r"/..\\",
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

    rad = ax.fill_between(x=star_age, y1=star_mass, y2=0, color="lightgray", alpha=1)

    return b1, b2, b3, b4, rad, conv, ov


def setup_axes(ax, star_age, star_mass, letter_top_left):
    ax.set_xscale("log")
    ax.set_xlim(1e3, star_age[-1])
    ax.set_ylim(0, 1.1 * star_mass[-1])
    ax.set_xlabel("star age (yr)")
    ax.set_ylabel(r"enclosed mass  ($M_\odot$)")
    ax.text(
        0.0125, 0.95, letter_top_left, ha="left", va="center", transform=ax.transAxes
    )


grid_directories = ["grid_Kippenhahn"]
model_directories = grh.get_sorted_model_directories(grid_directories)
model = model_directories[2]
model_classical = model_directories[3]

fig, ax = plt.subplots(2, 1)
plot_Kippenhahn(model_classical, ax[0])
plot_Kippenhahn(model, ax[1], first=False)
# needs function to add title labels for burning regions
plt.show()
fig.savefig("Images/Kippenhahn.png")
