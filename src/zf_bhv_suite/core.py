# src/zf_bhv_suite/core.py

import pandas as pd
import os
import glob
import sys
import json
from typing import Optional, Union, List, Dict

# Import constants from the local config file
from .config import ID_COLS


################################################################################
# FUNCTION 1: avgDayNight
################################################################################

def avgDayNight(input_dir: str, output_dir: Optional[str] = None) -> None:
    """
    Calculates the average of 'day' and 'night' columns for each CSV file
    in a directory and saves the updated files.
    ... [rest of the function code as previously provided]
    """

    # Set output_dir to input_dir if not specified
    if output_dir is None:
        output_dir = input_dir

    # Check if input_dir exists
    if not os.path.isdir(input_dir):
        sys.exit(f"Error: The input directory '{input_dir}' does not exist.")

    # Check if output_dir exists and create it if it doesn't
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)

    search_path = os.path.join(input_dir, "*.csv")
    csv_files = glob.glob(search_path)

    if not csv_files:
        print(f"Warning: No CSV files found in the input directory '{input_dir}'.")
        return

    print(f"Processing {len(csv_files)} files...")

    for file_path in csv_files:
        file_name = os.path.basename(file_path)
        print(f"  - Processing {file_name}")

        try:
            df = pd.read_csv(file_path)

            # Dynamic column identification
            day_cols = [col for col in df.columns if 'day' in col.lower() and col.lower() != 'day']
            night_cols = [col for col in df.columns if 'night' in col.lower()]

            if not day_cols and not night_cols:
                print(f"    Skipping {file_name}: No 'day' or 'night' columns found for averaging.")
                continue

            # Calculate means
            if day_cols:
                df['avgDay'] = df[day_cols].mean(axis=1)

            if night_cols:
                df['avgNight'] = df[night_cols].mean(axis=1)

            output_file_path = os.path.join(output_dir, file_name)
            df.to_csv(output_file_path, index=False)

        except Exception as e:
            print(f"    An error occurred while processing {file_name}: {e}")
            continue

    print("\nProcessing complete.")


################################################################################
# FUNCTION 2: normalise_data
################################################################################

def normalise_data(
    input_dirs: List[str],
    export_path: str,
    control_grouping: Union[str, Dict[str, str]],
    time_windows: List[str],
    parameters: Union[List[str], str] = 'all',
    save_individual_files: bool = False
) -> None:
    """
    Normalises behavioural data within experiments and combines results into a
    single, wide-format master CSV.

    Normalisation is done by subtracting an experiment-specific baseline mean
    from individual fish values.

    Args:
        input_dirs (List[str]): A list of directories, where each directory
            contains all CSVs for one unique experiment/clutch.
        export_path (str): The full path and filename for the final master CSV
            (e.g., '/path/to/data/master_normalised.csv').
        control_grouping (Union[str, Dict[str, str]]): Defines the baseline(s).
            - If str (e.g., 'wt_untreated'): All groups are normalised to this single baseline.
            - If dict: Defines a custom map for multi-baseline normalisation.
              Example: {'wt_drug': 'wt_untreated', 'mutant_drug': 'mutant_untreated'}
        time_windows (List[str]): The column names containing the data to be normalised.
        parameters (Union[List[str], str]): The behavioural parameters (CSV file prefixes) to process.
            Defaults to 'all'.
        save_individual_files (bool): If True, saves each normalised CSV in a
            subfolder named 'bhvparams_normalised' next to the master file. Defaults to False.
    """

    print("--- Starting Normalisation Process ---")

    # --- PHASE 1: PREPARATION AND ARGUMENT HANDLING ---

    # Get the parent directory for all outputs and the master file name
    output_dir = os.path.dirname(export_path)
    if not output_dir:
        output_dir = os.getcwd() # Use current working directory if only a filename is given

    master_filename = os.path.basename(export_path)

    # Filenames
    info_csv_filename = "normalisation_info.csv"
    metadata_json_filename = "normalisation_metadata.json"

    # 1. Handle the 'parameters' argument
    if isinstance(parameters, str) and parameters != 'all':
        parameters = [parameters]

    # 2. Input/Output Path Checks and Setup
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    normalized_files_dir = os.path.join(output_dir, 'bhvparams_normalised')
    if save_individual_files and not os.path.exists(normalized_files_dir):
        os.makedirs(normalized_files_dir)

    # Check for output conflict (cannot overwrite input)
    for in_dir in input_dirs:
        if os.path.abspath(in_dir) == os.path.abspath(output_dir):
            sys.exit(f"Error: Output directory '{output_dir}' cannot be the same as an input_dir.")

    # 3. Handle 'control_grouping' (Generate the final baseline_map)
    baseline_map = {}

    if isinstance(control_grouping, dict):
        baseline_map = control_grouping

    elif isinstance(control_grouping, str):
        single_baseline_group = control_grouping
        all_groups = set()

        # Read a sample file to find all available groups
        try:
            sample_dir = input_dirs[0]
            search_path = os.path.join(sample_dir, "*.csv")
            sample_file_paths = glob.glob(search_path)

            if not sample_file_paths:
                sys.exit(f"Error: No CSV files found in the sample directory: {input_dirs[0]}")

            sample_df = pd.read_csv(sample_file_paths[0])
            all_groups.update(sample_df['grp'].unique())

        except KeyError:
            sys.exit("Error: Sample CSV file is missing the required 'grp' column for normalisation setup.")

        # Create the map: every found group points to the single_baseline_group
        for group in all_groups:
            baseline_map[group] = single_baseline_group

    if not baseline_map:
         sys.exit("Error: Could not generate a valid baseline map. Check control_grouping input.")


    # 4. Initialize Containers
    metadata_list = []
    # This list will hold one wide-format DataFrame for each experiment
    experiment_dfs_for_master = []
    mean_audit_data = [] # Container for exporting the calculated means for auditing

    # --- PHASE 2: PROCESSING AND NORMALISATION ---

    # ID_COLS is now imported from .config

    for i, dir_path in enumerate(input_dirs):
        # Experiment ID column name is 'experiment', populated with 'experiment_1', etc.
        experiment_id = f"experiment_{i + 1}"
        print(f"\nProcessing {experiment_id}: {os.path.basename(dir_path)}")

        search_path = os.path.join(dir_path, "*.csv")
        csv_files = glob.glob(search_path)

        # Filter CSVs by 'parameters' list
        if parameters != 'all':
            csv_files = [f for f in csv_files if os.path.basename(f).split('_')[0] in parameters]

        if not csv_files:
            print(f"Warning: No files matched parameters {parameters} in {dir_path}. Skipping.")
            continue

        experiment_metadata = {
            'experiment_id': experiment_id,
            'input_path': dir_path,
            'baseline_map': baseline_map,
            'calculated_baselines': {}
        }

        # Initialize the wide-format DF for THIS experiment only
        exp_wide_df = None

        # Determine unique baseline groups required for THIS experiment
        unique_baselines = set(baseline_map.values())

        # --- File Loop ---
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            parameter = file_name.split('_')[0]

            try:
                df = pd.read_csv(file_path)

                # Check for required ID columns (all elements of ID_COLS except 'experiment', which is added later)
                core_id_cols = ID_COLS[:-1]
                if not all(col in df.columns for col in core_id_cols):
                    print(f"    Skipping {file_name}: Missing core ID columns ({core_id_cols}).")
                    continue

                # QC Check: Ensure all time windows exist
                if not all(col in df.columns for col in time_windows):
                    print(f"    Skipping {file_name}: Missing required time_windows columns.")
                    continue

                # QC Check: Ensure all required baselines exist in this experiment
                if not all(baseline in df['grp'].unique() for baseline in unique_baselines):
                    print(f"    Skipping {file_name}: Missing one or more required baseline groups in the 'grp' column.")
                    continue

                # 7. Calculate Baseline Means
                calculated_means = {}
                for window in time_windows:
                    for baseline in unique_baselines:
                        mean_val = df[df['grp'] == baseline][window].mean()
                        key = f"{baseline}_{window}"
                        calculated_means[key] = mean_val

                        # Collect data for the audit log
                        mean_audit_data.append({
                            'experiment_id': experiment_id,
                            'parameter': parameter,
                            'time_window': window,
                            'baseline_group': baseline,
                            'calculated_mean': mean_val
                        })

                    # 8. Normalise Data (Per Time Window)
                    new_col_name = f"{parameter}_{window}_normalised"
                    df[new_col_name] = float('nan')

                    for target_grp, baseline_grp in baseline_map.items():
                        # Find the mean value calculated for this specific baseline group and window
                        baseline_key = f"{baseline_grp}_{window}"
                        baseline_mean = calculated_means[baseline_key]

                        # Apply normalisation: value - baseline_mean
                        mask = df['grp'] == target_grp
                        df.loc[mask, new_col_name] = df.loc[mask, window] - baseline_mean

                # Store calculated means in metadata
                experiment_metadata['calculated_baselines'][parameter] = calculated_means

                # 9. Finalise DataFrame (Individual)
                df['experiment'] = experiment_id

                # Filter rows to include only those defined as keys in baseline_map (groups to normalise)
                df = df[df['grp'].isin(baseline_map.keys())].copy()

                # Save Individual File (Optional)
                if save_individual_files:
                    output_file_path = os.path.join(normalized_files_dir, file_name.replace('.csv', '_normalised.csv'))
                    df.to_csv(output_file_path, index=False)

                # 10. Extract and Merge (The Wide-Format Step for THIS EXPERIMENT)

                # Get only the normalised columns for merging
                norm_cols = [col for col in df.columns if col.endswith('_normalised')]

                # temp_df only contains ID columns (from config) and the normalised columns for the current parameter
                temp_df = df[ID_COLS + norm_cols].drop_duplicates(subset=ID_COLS)

                if exp_wide_df is None:
                    exp_wide_df = temp_df
                else:
                    # Merge temp_df into the growing experiment-wide DF
                    exp_wide_df = pd.merge(
                        exp_wide_df,
                        temp_df,
                        on=ID_COLS,
                        how='outer'
                    )

            except Exception as e:
                print(f"    An error occurred while processing {file_name}: {e}. Skipping file.")
                continue

        # Append the complete, wide-format DF for this experiment to the master list
        if exp_wide_df is not None:
             experiment_dfs_for_master.append(exp_wide_df)

        metadata_list.append(experiment_metadata)


    # --- PHASE 3: FINAL EXPORT AND REPORTING ---

    print("\n--- Finalizing Output ---")

    if not experiment_dfs_for_master:
        print("Warning: No data was successfully processed to create the master file.")
        return

    # 11. Export Master CSV - Concatenate all experiment-wide DFs vertically
    master_normalization_df = pd.concat(experiment_dfs_for_master, ignore_index=True)

    # Save to the user-specified export_path
    output_master_path = os.path.join(output_dir, master_filename)
    master_normalization_df.to_csv(output_master_path, index=False)
    print(f"Saved master file to: {output_master_path}")

    # 12. Export Metadata JSON File (Archival Context)
    output_metadata_path = os.path.join(output_dir, metadata_json_filename)
    with open(output_metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=4)
    print(f"Saved normalisation metadata to: {output_metadata_path}")

    # 13. Export Info CSV (Auditable Log)
    if mean_audit_data:
        audit_df = pd.DataFrame(mean_audit_data)
        audit_path = os.path.join(output_dir, info_csv_filename)
        audit_df.to_csv(audit_path, index=False)
        print(f"Saved normalisation info (audit log) to: {audit_path}")

    print("\nNormalisation complete.")
