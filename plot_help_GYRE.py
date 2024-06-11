# from astropy.constants import M_sun, L_sun
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import re

import plot_help_MESA as phM

# import scienceplots
mpl.rcParams["lines.markersize"] = 6  # Default marker size
mpl.use("TkAgg")
# plt.style.use("science")
fontsize = 21
params = {
    "axes.labelsize": fontsize,
    "font.size": fontsize,
    "legend.fontsize": fontsize,
    "xtick.labelsize": fontsize,
    "ytick.labelsize": fontsize,
    "text.usetex": False,
    "figure.figsize": [16, 9],
}
mpl.rcParams.update(params)


def get_hist_data(modelpath, logdir="LOGS"):
    history_file = os.path.join(modelpath, logdir + "/history.data")
    history_data = np.genfromtxt(history_file, skip_header=5, names=True)
    return history_data


def get_summary_data(
    modelpath, gyremodelnumber, omega=0.25, logdir="GYRE", nonadiabatic=False
):
    gyremodelstring = get_gyre_model_string(gyremodelnumber, omega, nonadiabatic)
    summary_file = os.path.join(modelpath, logdir, gyremodelstring)
    summary_data = np.genfromtxt(summary_file, skip_header=5, names=True)
    return summary_data


def get_data_array(history_data, history_parameter):
    return history_data[history_parameter]


def get_gyre_data_array(summary_data, summary_parameter):
    return summary_data[summary_parameter]


def get_max_growth_rate(summary_data):
    growth_rates = get_gyre_data_array(summary_data, "Imomega")
    return np.max(growth_rates)


def get_frequencies_and_periods(history_data):
    frequencies = get_data_array(history_data, "Refreq")
    periods = 1.0 / frequencies
    return frequencies, periods


def calculate_period_spacings(periods):
    sorted_periods = np.sort(periods)
    period_spacings = np.diff(sorted_periods)
    return sorted_periods, period_spacings


def get_model_numbers(directory, omega=0.25, nonadiabatic=False, rewrite=False):
    adiabatic_string = "_ad" if nonadiabatic is False else "nad"
    model_numbers_file = os.path.join(
        directory, "model_numbers" + adiabatic_string + ".txt"
    )

    if os.path.exists(model_numbers_file) and rewrite is False:
        # Load model numbers from the existing file
        with open(model_numbers_file, "r") as file:
            model_numbers = [int(line.strip()) for line in file]
    else:
        # Extract model numbers using regular expressions
        gyremodelstring = get_gyre_model_string(omega=omega, nonadiabatic=nonadiabatic)
        files = os.listdir(directory)

        model_numbers = []
        for file in files:
            match = re.search(r"model_(\d+)" + gyremodelstring, file)
            if match:
                model_number = int(match.group(1))
                print(model_number)
                model_numbers.append(model_number)

        if model_numbers:
            # Write model numbers to the file
            with open(model_numbers_file, "w") as file:
                for number in model_numbers:
                    file.write(f"{number}\n")
        else:
            raise ValueError(
                f"Could not match model numbers for directory {directory}!"
            )

    return np.array(model_numbers)


def get_min_max_model_numbers(directory, omega=0.25, nonadiabatic=False):
    files = os.listdir(directory)
    print(files)

    # Extract model numbers using regular expressions
    gyremodelstring = get_gyre_model_string(omega=omega, nonadiabatic=nonadiabatic)
    model_numbers = []
    for file in files:
        match = re.search(r"model_(\d+)" + gyremodelstring, file)
        if match:
            model_number = int(match.group(1))
            model_numbers.append(model_number)

    if model_numbers:
        min_model = min(model_numbers)
        max_model = max(model_numbers)
        return min_model, max_model
    else:
        raise ValueError(f"Could not match model numbers for directory {directory}!")


def get_gyre_model_string(gyremodelnumber=None, omega=0.25, nonadiabatic=False):
    adiabatic = "_ad" if not nonadiabatic else "_nad"
    gyremodelstring = ".data.GYRE" + str(omega) + adiabatic + ".txt"
    if gyremodelnumber is not None:
        gyremodelstring = "model_" + str(gyremodelnumber) + gyremodelstring
    return gyremodelstring


def single_gyre_plot(
    model_directories,
    paramX,
    paramY,
    xlabel,
    ylabel,
    paramx_log=False,
    paramy_log=False,
    invert_xaxis=False,
    nonadiabatic=False,
    gyredir="GYRE",
):
    fig, ax = plt.subplots(1, 1)

    models_to_ignore = [
        "grid1/model3M",
        "grid1/model3p5M",
        "grid1/model4p5M_hist_GYRE",
        "grid1/model6M_hist",
        "grid1/model6p5M_hist",
        "grid1/model7M_hist",
        "grid1/model7p5M_hist",
    ]
    for model in model_directories:
        if model in models_to_ignore:
            continue
        history_data = phM.get_hist_data(model)
        modelpath = os.path.join(model, gyredir)
        model_numbers = get_model_numbers(modelpath, nonadiabatic=nonadiabatic)
        model_numbers = np.sort(model_numbers)
        model_numbers_nad = get_model_numbers(modelpath, nonadiabatic=False)
        model_numbers_nad = np.sort(model_numbers)
        indices = np.where(np.isin(model_numbers_nad, model_numbers))[0]
        max_growth_rate = 0
        growth_rates = []
        profile_index_data = phM.get_profile_index_data(model)
        hist_indices = profile_index_data["model_number"] - 1
        paramx_data = phM.get_hist_datapoints(history_data, paramX, hist_indices)
        paramy_data = phM.get_hist_datapoints(history_data, paramY, hist_indices)
        profile_numbers = profile_index_data["profile_number"]
        
        for model_n in model_numbers:
            summary_data = get_summary_data(model, model_n, nonadiabatic=nonadiabatic)
            new_maxgrowthrate = get_max_growth_rate(summary_data)
            growth_rates.append(new_maxgrowthrate)
            if new_maxgrowthrate > max_growth_rate:
                max_growth_rate = new_maxgrowthrate
        norm = mcolors.Normalize(vmin=0, vmax=max_growth_rate)
        colormap = cm.plasma

        for index in indices:
            profile_data_indices = np.where(index == profile_numbers)[0]  # Access the first element of the tuple
            if len(profile_data_indices) > 0:
                profile_data_index = profile_data_indices[0]  # Get the first index
                model_n = model_numbers_nad[index] 
                summary_data = get_summary_data(model, model_n, nonadiabatic=nonadiabatic)
                label_value = get_max_growth_rate(summary_data)
                color = colormap(norm(label_value))
                #print(paramy_data[profile_data_index])
                ax.scatter(paramx_data[profile_data_index], paramy_data[profile_data_index], color=color)
            else:
                print(f"No matches for model number: {index}")

    #ax.set_xlim([1150, 1400])
    #ax.set_ylim([0, 0.0001])
    ax.invert_xaxis()
    ax.legend()
    plt.show()


def plot_period_spacings(model_directories, model_n=None, gyredir="GYRE"):
    fig, ax = plt.subplots(1, 1)
    for model in model_directories:
        modelpath = os.path.join(model, gyredir)
        model_numbers = get_model_numbers(modelpath)
        print(model_numbers)
        model_numbers = model_n if model_n is not None else model_numbers
        for model_n in model_numbers:
            summary_data = get_summary_data(model, model_n)
            angular_degrees = get_gyre_data_array(summary_data, "l")
            frequencies, periods = get_frequencies_and_periods(summary_data)
            sorted_periods, period_spacings = calculate_period_spacings(periods)
            sorted_periods = sorted_periods[1:]
            # ax.plot(sorted_periods[:-1], period_spacings)
            for l_value in range(3):
                indices = np.where(angular_degrees == l_value)[0]
                print(indices)
                ax.scatter(
                    sorted_periods[indices[:-1]],
                    period_spacings[indices[:-1]],
                    s=10,
                    label=f"l={l_value}",
                )

    ax.set_xlabel("Period (days)")
    ax.set_ylabel("dP (days)")
    ax.set_title("Period Spacings")
    ax.legend()

    plt.show()


def plot_HR(model_directories, save=None, ignore_last=False, logdir="LOGS"):
    return single_plot(
        model_directories,
        "log_Teff",
        "log_L",
        r"$\log(T_{\mathrm{eff}} \, / \, \mathrm{K})$",
        r"$\log(L \, / \, L_{\odot})$",
        ("Mass", "star_mass", -1, r"$M_{\odot}$"),
        invert_xaxis=True,
        title="HR Diagram",
        save=save,
        ignore_last=ignore_last,
        logdir=logdir,
        birthline=True,
    )


def single_plot(
    model_directories,
    paramX,
    paramY,
    xlabel,
    ylabel,
    label,
    paramx_log=False,
    paramy_log=False,
    invert_xaxis=False,
    logxscale=False,
    logyscale=False,
    title=None,
    save=None,
    ignore_last=False,
    logdir="LOGS",
    birthline=True,
    instability=True,
    greyed_directory="grid_classical",
):
    """
    Generates a single plot for all models within the model_directories for the two Parameters X and Y.
    Label is a tuple which should be passed as labelname, parametername(MESA), unit
    """
    fig, ax = plt.subplots(1, 1)

    if birthline:
        data = np.genfromtxt(
            "external_data/birthline_logT_logL_Palla_Stahler1999.dat",
            names=("logT, logL"),
        )
        ax.plot(
            data["logT"],
            data["logL"],
            color="k",
            linestyle="--",
            linewidth=3,
            label="Birthline Palla & Stahler 1999",
        )

    if instability:
        ax.axvline(np.log10(11000))
        ax.axvline(np.log10(18700))

    for model in reversed(model_directories):
        history_data = get_hist_data(model, logdir=logdir)

        paramx_data = get_data_array(history_data, paramX)
        paramy_data = get_data_array(history_data, paramY)

        paramx_data = paramx_data[:-1] if ignore_last else paramx_data
        paramy_data = paramy_data[:-1] if ignore_last else paramy_data

        paramx_data = 10**paramx_data if paramx_log else paramx_data
        paramy_data = 10**paramy_data if paramy_log else paramy_data

        label_name, label_param, label_index, label_unit = label
        label_param_data = get_data_array(history_data, label_param)
        label_value = label_param_data[label_index]
        label_string = f"{label_name}: {label_value:.2f} {label_unit}"

        if greyed_directory in model:
            # greyed_model_string = label_string + " classical"
            ax.plot(paramx_data, paramy_data, linewidth=2, color="grey")
        else:
            norm = mcolors.Normalize(vmin=4, vmax=8.5)
            colormap = cm.plasma
            color = colormap(norm(label_value))
            ax.plot(
                paramx_data,
                paramy_data,
                linewidth=5,
                color=color,
                label=label_string,
            )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale("log") if logxscale else None
    ax.set_yscale("log") if logyscale else None
    ax.set_xlim(left=4)
    ax.set_ylim([1, 3.5])
    ax.invert_xaxis() if invert_xaxis else None
    ax.set_title(title) if title is not None else None
    ax.legend()

    # fix scatter size in legend
    # for handle in ax.legend_.legendHandles:
    #     if (
    #         type(handle).__name__ == "PathCollection"
    #     ):  # Check if handle is associated with scatter plot
    #         handle.set_sizes([100])

    fig.tight_layout()
    fig.savefig(save) if save is not None else plt.show()


def interpolate_first_diff_two_models(
    model_ref, model_tgt, paramA, paramB, deltaA, deltaB=None
):
    """
    Interpolates between the historical data of the reference model and the target model to identify the first significant difference in one or both of the provided parameters.

    Parameters:
    - model_ref (object): The reference model.
    - model_tgt (object): The target model.
    - paramA (str): The name of the first parameter.
    - paramB (str): The name of the second parameter.
    - deltaA (float): The threshold for significant difference in paramA.
    - deltaB (float, optional): The threshold for significant difference in paramB. If None, it takes the value of deltaA.

    Returns:
    - dif_index_ref (int): Index of the first significant difference in the reference model's history.
    - dif_index_tgt (int): Index of the first significant difference in the target model's history.
    """
    hist_ref = get_hist_data(model_ref)
    hist_tgt = get_hist_data(model_tgt)

    params_ref = [get_data_array(param, hist_ref) for param in (paramA, paramB)]
    params_tgt = [get_data_array(param, hist_tgt) for param in (paramA, paramB)]

    if deltaB is None:
        deltaB = deltaA

    differences = np.where(
        (params_tgt[0][: len(params_ref[0])] - params_ref[0] > deltaA)
        & (params_tgt[1][: len(params_ref[1])] - params_ref[1] > deltaB)
    )

    first_difference = differences[0][0]
    dif_index_ref = first_difference
    dif_index_tgt = first_difference + (len(params_tgt[0]) - len(params_ref[0]))

    return dif_index_ref, dif_index_tgt


def print_gyre_summary_names():
    print(
        "available gyre summary data names:\n ('E_norm', 'L_star', 'M_star', 'R_star', 'W', 'beta', 'dfreq_rot', 'eta', 'Refreq', 'Imfreq', 'l', 'm', 'n_g', 'n_p', 'n_pg', 'Reomega', 'Imomega')"
    )
