import matplotlib as mpl
#from matplotlib import legend
from matplotlib import legend
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import re
import os

# plt.style.use("science")
def update_plot_params(
    fig_width_pt = 0.855*398.338,   # Get this from LaTeX using \showthe\columnwidth
    fig_height_relative = None,
    legend_width = 3, # Additional space for legend on the side
    fontsize = 13,
    ):
    inches_per_pt = 1.0/72.27       # Convert pt to inch
    golden_mean = (np.sqrt(5)-1.0)/2.0 # Aesthetic ratio
    fig_width = fig_width_pt*inches_per_pt + legend_width # width in inches
    if fig_height_relative is None: # height in inches
        fig_height = fig_width*golden_mean 
    else:
        fig_height = fig_width * fig_height_relative
    fig_size =  [fig_width,fig_height]
    params = {#'backend': 'ps',
      'axes.labelsize': fontsize,
      'font.size': fontsize,
      'legend.fontsize': fontsize - 1,
      'xtick.labelsize': fontsize - 2,
      'ytick.labelsize': fontsize - 2,
      'text.usetex': True,
      'figure.figsize': fig_size}
    mpl.rcParams.update(params)


def get_hist_data(modelpath, logdir="LOGS"):
    history_file = os.path.join(modelpath, logdir + "/history.data")
    history_data = np.genfromtxt(history_file, skip_header=5, names=True)
    return history_data


def get_value_from_modelstring(modelpath, pattern):
    """
    Extracts a value from a given modelstring based on the provided regex pattern.
    
    Args:
    modelpath (str): The input modelstring containing the value to be extracted.
    pattern (str): The regex pattern to use for extracting the value. 
                   The pattern should use capturing groups to specify which parts of the match to extract.
    
    Returns:
    str: The extracted alpha value as a string in the format 'X.Y'.
    """
    match = re.search(pattern, modelpath)
    if match:
        return float(f"{match.group(1)}.{match.group(2)}")
    return None


def get_hist_data_array(history_data, history_parameter):
    return history_data[history_parameter]

def get_hist_datapoints(history_data, history_parameter, index):
    data_array = get_hist_data_array(history_data, history_parameter)
    return data_array[index]

def get_profile_index_data(modelpath, logdir="LOGS"):
    filepath = os.path.join(modelpath, logdir, "profiles.index")
    profile_data = np.genfromtxt(filepath, dtype=int, skip_header=1, names=["model_number", "priority", "profile_number"])
    return profile_data

def get_gyre_profile_data_old(modelpath, logdir="LOGS", file="gyre_models.txt"):
    filepath = os.path.join(modelpath, logdir, file)
    gyre_profile_data = np.genfromtxt(filepath, names=True)
    return gyre_profile_data


def get_filtered_models(models, *args, epsilon=0.001):
    """
    Filters the models to include only those ending in any of the specified values in args.

        Parameters:
        models (array-like): An array of model values.
        *args (float): Arbitrary number of float values to filter by.

        Returns:
        np.ndarray: Filtered array of models.
    """
    if not args:
        raise ValueError("Provide at least one float value to filter by.")

    # Create an array of boolean conditions for each value in args
    conditions = np.zeros(models.shape, dtype=bool)
    for val in args:
        conditions |= np.isclose(models % 1, val, atol=epsilon)

    # Filter models based on the combined conditions
    return models[conditions]


def get_model_passes_filter(model_value, *args, epsilon=0.001):
    """
    Determines if a single model value passes the filter based on specified values and precision.

    Parameters:
    model (float): A single model value.
    *args (float): Arbitrary number of float values to filter by.
    epsilon (float): Precision value for np.isclose (default is 0.001).

    Returns:
    bool: True if the model passes the filter, False otherwise.
    """
    if not args:
        raise ValueError("At least one value must be specified to filter by.")

    # Check if the model passes any of the conditions
    for val in args:
        if np.isclose(model_value % 1, val, atol=epsilon):
            return True

    # If none of the conditions are met, return False
    return False

def get_model_passes_filter_exp(model_value, *args, epsilon=0.001):
    """
    Determines if a single model value passes the filter based on specified values and precision.

    Parameters:
    model_value (float): A single model value.
    *args (float): Arbitrary number of float values to filter by.
    epsilon (float): Precision value for np.isclose (default is 0.001).

    Returns:
    bool: True if the model passes the filter, False otherwise.
    """
    if not args:
        raise ValueError("At least one value must be specified to filter by.")

    # Convert the model_value to a string in scientific notation format with high precision
    model_str = f"{model_value:.14e}"
    significant_str = model_str.split('e')[0]
    significant_value = float(significant_str)

    # Extract the fractional part of the significant value
    fractional_part = significant_value - int(significant_value)

    # Check if the model passes any of the conditions
    for val in args:
        # Convert the filter value to a string in scientific notation format with high precision
        val_str = f"{val:.14e}"
        val_significant_str = val_str.split('e')[0]
        val_significant_value = float(val_significant_str)
        
        if np.isclose(fractional_part % 1, val_significant_value % 1, atol=epsilon):
            return True

    # If none of the conditions are met, return False
    return False
  
def plot_gyre_HR(history_data, model, ax, gyrelabel):
    profile_index_data = get_profile_index_data(model)
    hist_indices = profile_index_data["model_number"] - 1
    logTeff = get_hist_datapoints(history_data, "log_Teff", hist_indices)
    logL = get_hist_datapoints(history_data, "log_L", hist_indices)
    ax.plot(logTeff, logL, linewidth=0.75, color="cyan", label=gyrelabel)
    #ax.plot(logTeff, logL, ".", markersize=3, color="cyan") not enough memory
    return ax
    


def plot_HR(
    model_directories,
    save=None,
    label=("Mass", "star_mass", -1, r"$M_{\odot}$"),
    labeluse=False,
    colorbar=(3, 8.5, False),
    xlim=None,
    ylim=None,
    filter_models=None,
    ignore_last=False,
    birthline=True,
    plot_gyre_models=False,
    instability=False,
    logdir="LOGS",
    gen_label_val_from_regx=None,
    linewidth=3,
    legend_outside=True,
):
    single_plot(
        model_directories,
        "log_Teff",
        "log_L",
        r"$\log(T_{\mathrm{eff}} \, / \, \mathrm{K})$",
        r"$\log(L \, / \, L_{\odot})$",
        label=label,
        labeluse=labeluse,
        colorbar=colorbar,
        xlim=xlim,
        ylim=ylim,
        invert_xaxis=True,
        title="HR Diagram",
        save=save,
        ignore_last=ignore_last,
        logdir=logdir,
        birthline=birthline,
        filter_models=filter_models,
        plot_gyre_models=plot_gyre_models,
        instability=instability,
        gen_label_val_from_regx=gen_label_val_from_regx,
        linewidth=linewidth,
        legend_outside=legend_outside
    )


def plot_Kippenhahn(model_directories, save=False):
    return


def single_plot(
    model_directories,
    paramX,
    paramY,
    xlabel,
    ylabel,
    label,
    colorbar=(3, 8.5, False),
    labeluse=False,
    xlim=None,
    ylim=None,
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
    instability=False,
    filter_models=None,
    plot_gyre_models=False,
    greyed_directory="grid_classical",
    gen_label_val_from_regx=None,
    linewidth=3,
    legend_outside=True,
):
    """
    Generates a single plot for all models within the model_directories for the two Parameters X and Y.
    Label is a tuple which should be passed as labelname, parametername(MESA), unit
    """
    plt.switch_backend('pgf')
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
            label=r"Birthline Palla""\n"r"\& Stahler 1999",
        )

    if instability:
        ax.axvline(np.log10(11000))
        ax.axvline(np.log10(18700))

    vmin, vmax, logscale = colorbar
    
    for model in reversed(model_directories):
        history_data = get_hist_data(model, logdir=logdir)

        paramx_data = get_hist_data_array(history_data, paramX)
        paramy_data = get_hist_data_array(history_data, paramY)

        paramx_data = paramx_data[:-1] if ignore_last else paramx_data
        paramy_data = paramy_data[:-1] if ignore_last else paramy_data

        paramx_data = 10**paramx_data if paramx_log else paramx_data
        paramy_data = 10**paramy_data if paramy_log else paramy_data

        label_name, label_param, label_index, label_unit = label
        label_value = 0
        if gen_label_val_from_regx is None: 
            label_param_data = get_hist_data_array(history_data, label_param) 
            label_value = label_param_data[label_index]
        else:
            label_value = get_value_from_modelstring(model, gen_label_val_from_regx)
        label_string = f"{label_name}: {label_value:.2f} {label_unit}"
        if labeluse:
            label_string = f"{label_name}: {label_value:.2e} {label_unit}"
        if filter_models is not None:
            if logscale is False and not get_model_passes_filter(
                label_value, *filter_models):
                label_string = "_nolegend_" 
            elif logscale is True and not get_model_passes_filter_exp(label_value, *filter_models):
                label_string = "_nolegend_"

        if greyed_directory in model:
            # greyed_model_string = label_string + " classical"
            greyed_model_string = "Classical models" if model == "grid_classical/model7M_hist1" else None
            ax.plot(paramx_data, paramy_data, linewidth=2, color="grey", label=greyed_model_string)
        else:
            norm = (
                mcolors.Normalize(vmin=vmin, vmax=vmax)
                if not logscale
                else mcolors.LogNorm(vmin=vmin, vmax=vmax)
            )
            colormap = cm.plasma
            color = colormap(norm(label_value))
            ax.plot(
                paramx_data,
                paramy_data,
                linewidth=linewidth,
                color=color,
                label=label_string,
                #alpha=0.5, Doesnt work unfortunately
            )
        models_to_ignore = [
            "grid_classical/model4M_hist1",
            "grid_classical/model5M_hist1",
            "grid_classical/model6M_hist1",
            "grid_classical/model7M_hist1",
        ]
        # Make this a function
        if plot_gyre_models and model not in models_to_ignore:
            gyrelabel = r"Gyre models" if model is model_directories[0] else None 
            plot_gyre_HR(history_data, model, ax, gyrelabel) 

    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5)) if legend_outside else ax.legend()
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xscale("log") if logxscale else None
    ax.set_yscale("log") if logyscale else None
    ax.set_xlim(xlim) if xlim is not None else None
    ax.set_ylim(ylim) if ylim is not None else None
    ax.invert_xaxis() if invert_xaxis else None
    ax.set_title(title) if title is not None else None
    #ax.legend()
    # fix scatter size in legend
    # for handle in ax.legend_.legendHandles:
    #     if (
    #         type(handle).__name__ == "PathCollection"
    #     ):  # Check if handle is associated with scatter plot
    #         handle.set_sizes([100])

    fig.tight_layout()
    fig.savefig(save, bbox_inches='tight') if save is not None else plt.show()


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
