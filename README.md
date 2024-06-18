# Grids for MESA and GYRE #
Grids for MESA and GYRE is a library of python modules to help with the organisation and plotting of MESA and GYRE data produced using grids.

#### While already functional, it should be noted that this repo is under active development, which may include breaking changes. ####

## The expected grid structure is as follows: ##
````bash
example_work_dir
├── grid1
│  ├── model1
│  │  └── LOGS
│  │     └── history.data
│  └── model2
└── grid2
````

## Features ##
The `grid_help.py` module is the starting point for the analysis. 
It locates all available models within the specified grid directories.
Failed models can be excluded by adding their name to the failed_models.txt file within the corresponding grid directory.

The `plot_help_MESA.py` module uses the input models obtained from the `grid_help.py` module. 
It generates a number of predefined plots such as `plot_HR` and soon a Kippenhahn(currently in the `plot_Kippen.py` module) and Kiel diagram.

Similarly the `plot_help_GYRE.py` module uses the input models obtained from the `grid_help.py` module to plot GYRE data. 

## Example Usage ##
````python
import grid_help as grh
import plot_help_MESA as phM

grid_directories = ["grid1", "grid2"]
model_directories = grh.get_sorted_model_directories(grid_directoires)
phM.plot_HR(model_directories)
````
