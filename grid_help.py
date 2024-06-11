import os
import re

def extract_model_directories(grid_directories, debug=False):
    # Prepend all directory strings to fix relative paths
    #["../" + grid_directory for grid_directory in grid_directories]

    # The blueprint_directory and all models inside
    # the failed_models_file will be ignored
    blueprint_directory = "blueprint"
    failed_models_file = "failed_models.txt"
 
    # Initialize lists for directories to ignore and model directories
    directories_to_ignore = []
    model_directories = []

    for grid_directory in grid_directories:
        # Check for failed models file and add its entries to directories_to_ignore
        failed_models_path = os.path.join(grid_directory, failed_models_file)
        if os.path.exists(failed_models_path):
            with open(failed_models_path, "r") as f:
                directories_to_ignore.extend(
                    [os.path.join(grid_directory, d.strip()) for d in f.read().splitlines()]
                )

        # Add blueprint directory to directories_to_ignore
        directories_to_ignore.append(os.path.join(grid_directory, blueprint_directory))

        # Add directories to model_directories if they are not in directories_to_ignore
        for d in os.listdir(grid_directory):
            dir_path = os.path.join(grid_directory, d)
            if os.path.isdir(dir_path) and dir_path not in directories_to_ignore:
                model_directories.append(dir_path)

    if debug:
        print("Directories to ignore:", directories_to_ignore)
        print("Model directories:", model_directories)

    return model_directories


def extract_model_mass(directory):
    # Extract the mass from the directory name
    massp = re.findall(r"(\d+)p(\d+)", directory)
    if massp:
        return float(f"{massp[0][0]}.{massp[0][1]}")
    else:
        mass = re.findall(r"(\d+)M", directory)
        if mass:
            return float(f"{mass[0]}.0")
        raise ValueError(f"Could not extract mass from directory: {directory}")

def extract_value_regex(directory, pattern, reverseorder=False):
    match = re.search(pattern, directory)
    if match:
        if len(match.groups()) == 2:
            if reverseorder:
                return (int(match.group(2)), int(match.group(1)))  # (second value, first value)
            else:
                return (int(match.group(1)), int(match.group(2)))  # (first value, second value)
        elif len(match.groups()) == 1:
            return (int(match.group(1)), 0)
    raise ValueError(f"Could not match regex pattern in directory: {directory}")

def get_sorted_model_directories(grid_directories, debug=False, pattern=None, reverseorder=False):
    model_directories = extract_model_directories(grid_directories, debug=debug)
    
    if pattern is None:
        key = extract_model_mass
    else:
        def key(directory):
            return extract_value_regex(directory, pattern, reverseorder)
    
    return sort_directories(model_directories, key=key)

def sort_directories(model_directories, key=extract_model_mass):
    return sorted(model_directories, key=key)

if __name__ == "__main__":
    grid_directories = ["grid1", "grid2"]
    models = get_sorted_model_directories(grid_directories)
    print(models)
