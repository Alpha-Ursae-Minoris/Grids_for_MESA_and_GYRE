from matplotlib.pyplot import legend
import grid_help as grh
import plot_help_MESA as phM

def plot_acc_rate(save="Figures/acc_rates.pdf", xlim=None, ylim=None, filter_models=None, colorbar=(1e-7, 1e-2, True)):
    grid_directoires = ["grid_acc_rates"]
    acc_rate_pattern = r"(\d+)e(\d+)dot"
    model_directories = grh.get_sorted_model_directories(grid_directoires, pattern=acc_rate_pattern, reverseorder=True)
    phM.plot_HR(model_directories, 
        label=("Mass change", "star_mdot", 20, r"$\dot{M}_{\odot}$"),
        labeluse=True,
        colorbar=colorbar,
        save=save,
        xlim=xlim,
        ylim=ylim,
        filter_models=filter_models,
    )

def plot_efold():
    grid_directories = ["grid_efolding"]
    efold_pattern = r"(\d+)p(\d+)fold"
    model_directories = grh.get_sorted_model_directories(grid_directories, pattern=efold_pattern)
    phM.plot_HR(model_directories, 
                label=(r"$\tau$", "", 0, ""),       
                xlim=xlim, ylim=ylim, 
                gen_label_val_from_regx=efold_pattern, colorbar=(0, 1, False),
                save="Figures/efold.pdf")


def plot_HR_all_models(xlim, ylim):
    grid_directories = ["grid1", "grid2", "grid_classical"]
    model_directories = grh.get_sorted_model_directories(grid_directories)
    phM.plot_HR(
        model_directories,
        filter_models=(0.0, 0.5),
        save="Figures/HR_full.pdf",
        plot_gyre_models=True,
        xlim=xlim,
        ylim=ylim,
        linewidth=5,
    )

def plot_classical_only():
    grid_directoires = ["grid_classical"]
    model_directories = grh.get_sorted_model_directories(grid_directoires)
    phM.plot_HR(model_directories,
                save="Figures/Classical.pdf",
                legend_outside=False)

def plot_HR_initial():
    model_directories = grh.get_sorted_model_directories(["grid1"])
    phM.plot_HR(model_directories, save="Figures/HR_initial.pdf", legend_outside=False, filter_models=(0.0,), birthline=False)

xlim = [3.8, 4.35]
ylim = [1.0, 3.50]

# Full slide plots
phM.update_plot_params()
plot_HR_all_models(xlim=xlim, ylim=ylim)
plot_efold()
plot_acc_rate()
plot_acc_rate(xlim=xlim, ylim=ylim, save="Figures/acc_rates_zoomed.pdf", filter_models=(0.0, 0.5), colorbar=(1e-6, 1e-4, True))

# Half slide plots
phM.update_plot_params(legend_width=0, fig_height_relative=0.7)
plot_classical_only()
plot_HR_initial()
