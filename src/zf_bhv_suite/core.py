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
    export_path: str, # This is now the required output DIRECTORY for parameter files
    control_grouping: Union[str, Dict[str, str]],
    time_windows: List[str],
    parameters: Union[List[str], str] = 'all'
) -> None:
    """
    Normalises behavioural data within experiments and aggregates results into
    separate CSV files, one for each behavioural parameter, containing data
    from all experiments.

    Normalisation is done by subtracting an experiment-specific baseline mean
    from individual fish values.

    Args:
        input_dirs (List[str]): A list of directories, where each directory
            contains all CSVs for one unique experiment/clutch.
        export_path (str): The path to the folder where the final parameter-specific
            normalised CSV files will be saved.
        control_grouping (Union[str, Dict[str, str]]): Defines the baseline(s).
            - If str (e.g., 'wt_untreated'): All groups are normalised to this single baseline.
            - If dict: Defines a custom map for multi-baseline normalisation.
              Example: {'wt_drug': 'wt_untreated', 'mutant_drug': 'mutant_untreated'}
        time_windows (List[str]): The column names containing the data to be normalised.
        parameters (Union[List[str], str]): The behavioural parameters (CSV file prefixes) to process.
            Defaults to 'all'.
    
    The wide-format master summary CSV (master_normalised_summary.csv), which
    combines all normalised parameters across all experiments, is saved
    unconditionally inside the 'normalisation_info' folder.
    """

    print("--- Starting Normalisation Process ---")
    
    # NOTE: The _natural_sort_df helper function has been removed to ensure
    # that no sorting is applied during concatenation, preserving original row order.

    # --- PHASE 1: PREPARATION AND ARGUMENT HANDLING ---

    output_dir = export_path
    
    # Define the subdirectory for info files
    info_subdir = os.path.join(output_dir, 'normalisation_info')
    
    # Standard output filenames (saved in info_subdir)
    info_csv_filename = os.path.join(info_subdir, "normalisation_audit.csv")
    metadata_json_filename = os.path.join(info_subdir, "normalisation_metadata.json")
    master_summary_filename = os.path.join(info_subdir, "master_normalised_summary.csv")

    # 1. Handle the 'parameters' argument
    if isinstance(parameters, str) and parameters != 'all':
        parameters = [parameters]

    # 2. Input/Output Path Checks and Setup
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
        
    # Ensure the info subdirectory exists
    if not os.path.exists(info_subdir):
        os.makedirs(info_subdir)
        print(f"Created info directory: {info_subdir}")


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
            # Ensure 'grp' column exists
            if 'grp' not in sample_df.columns:
                 sys.exit("Error: Sample CSV file is missing the required 'grp' column for normalisation setup.")

            all_groups.update(sample_df['grp'].unique())

        except Exception as e:
            sys.exit(f"Error reading sample file for group definition: {e}")

        # Create the map: every found group points to the single_baseline_group
        for group in all_groups:
            baseline_map[group] = single_baseline_group

    if not baseline_map:
        sys.exit("Error: Could not generate a valid baseline map. Check control_grouping input.")


    # 4. Initialize Containers
    metadata_list = []
    # This dictionary aggregates data by parameter across ALL experiments
    all_parameter_dfs: Dict[str, pd.DataFrame] = {}
    mean_audit_data = [] # Container for exporting the calculated means for auditing
    master_summary_dfs_for_concat = []

    # --- PHASE 2: PROCESSING AND NORMALISATION ---

    for i, dir_path in enumerate(input_dirs):
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

        # Initialize the wide-format DF for THIS experiment only (if summary is needed)
        exp_wide_df_for_summary = None

        # Determine unique baseline groups required for THIS experiment
        unique_baselines = set(baseline_map.values())

        # --- File Loop (Processes one parameter file per experiment) ---
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            parameter = file_name.split('_')[0]

            try:
                df = pd.read_csv(file_path)

                # 5. QC Checks 
                core_id_cols = ID_COLS[:-1] # 'grp', 'fish_id', 'well_id'
                if not all(col in df.columns for col in core_id_cols) or not all(col in df.columns for col in time_windows):
                    print(f"    Skipping {file_name}: Missing core columns or time windows.")
                    continue

                if not all(baseline in df['grp'].unique() for baseline in unique_baselines):
                    print(f"    Skipping {file_name}: Missing one or more required baseline groups in the 'grp' column.")
                    continue

                # 6. Calculate Baseline Means and Normalise Data (Per Time Window)
                calculated_means = {}
                norm_cols_for_summary = [] # Track the normalized columns created in this parameter file

                for window in time_windows:
                    new_col_name = f"{parameter}_{window}_normalised"
                    norm_cols_for_summary.append(new_col_name)
                    df[new_col_name] = float('nan')

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

                    for target_grp, baseline_grp in baseline_map.items():
                        baseline_key = f"{baseline_grp}_{window}"
                        baseline_mean = calculated_means[baseline_key]

                        # Apply normalisation: value - baseline_mean
                        mask = df['grp'] == target_grp
                        df.loc[mask, new_col_name] = df.loc[mask, window] - baseline_mean

                # Store calculated means in metadata
                experiment_metadata['calculated_baselines'][parameter] = calculated_means

                # 7. Finalise DataFrame (Add 'experiment' and filter to target groups)
                df['experiment'] = experiment_id
                # Filter rows to include only those defined as keys in baseline_map (groups to normalise)
                df = df[df['grp'].isin(baseline_map.keys())].copy()
                
                # 8. Primary Output Aggregation (Long Format, Parameter-Specific)
                
                # Columns needed for the long-format parameter file: ID columns + newly created normalised columns
                cols_to_keep = ID_COLS + norm_cols_for_summary
                temp_df_long = df[cols_to_keep].drop_duplicates(subset=ID_COLS)

                if parameter not in all_parameter_dfs:
                    all_parameter_dfs[parameter] = temp_df_long
                else:
                    # Concatenate this experiment's data vertically to the running parameter DataFrame
                    # This preserves the original row order as requested.
                    all_parameter_dfs[parameter] = pd.concat([
                        all_parameter_dfs[parameter], 
                        temp_df_long
                    ], ignore_index=True)


                # 9. Master Summary Aggregation (Wide Format, Per Experiment - now mandatory)
                # temp_df_wide only contains ID columns and the normalised columns for the current parameter
                # This DF will be merged horizontally with others from the same experiment
                temp_df_wide = df[ID_COLS + norm_cols_for_summary].drop_duplicates(subset=ID_COLS)

                if exp_wide_df_for_summary is None:
                    exp_wide_df_for_summary = temp_df_wide
                else:
                    # Merge temp_df_wide into the growing experiment-wide DF
                    exp_wide_df_for_summary = pd.merge(
                        exp_wide_df_for_summary,
                        temp_df_wide,
                        on=ID_COLS,
                        how='outer'
                    )

            except Exception as e:
                print(f"    An error occurred while processing {file_name}: {e}. Skipping file.")
                continue

        # Append the complete, wide-format DF for this experiment to the master list (which is now always generated)
        if exp_wide_df_for_summary is not None:
             master_summary_dfs_for_concat.append(exp_wide_df_for_summary)

        metadata_list.append(experiment_metadata)


    # --- PHASE 3: FINAL EXPORT AND REPORTING ---

    print("\n--- Finalizing Output ---")

    if not all_parameter_dfs:
        print("Warning: No data was successfully processed to create output files.")
        return

    # 10. Export Parameter CSVs (The new primary output)
    for parameter, df_long in all_parameter_dfs.items():
        # No sorting applied here. The dataframe is concatenated sequentially.
        output_file_path = os.path.join(output_dir, f"{parameter}_normalised.csv")
        df_long.to_csv(output_file_path, index=False)
        print(f"Saved parameter file: {os.path.basename(output_file_path)}")

    # 11. Export Master Summary CSV (saved in info_subdir, now mandatory)
    if master_summary_dfs_for_concat:
        # Concatenate preserves the order of the list elements (experiment 1, then 2, etc.)
        master_summary_df = pd.concat(master_summary_dfs_for_concat, ignore_index=True)
        
        # No sorting applied here.
        output_master_path = master_summary_filename # Uses the path defined in info_subdir
        master_summary_df.to_csv(output_master_path, index=False)
        print(f"Saved master summary file to: {output_master_path}")

    # 12. Export Metadata JSON File (Archival Context, saved in info_subdir)
    output_metadata_path = metadata_json_filename # Uses the path defined in info_subdir
    with open(output_metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=4)
    print(f"Saved normalisation metadata to: {output_metadata_path}")

    # 13. Export Info CSV (Auditable Log, saved in info_subdir)
    if mean_audit_data:
        audit_df = pd.DataFrame(mean_audit_data)
        audit_path = info_csv_filename # Uses the path defined in info_subdir
        audit_df.to_csv(audit_path, index=False)
        print(f"Saved normalisation info (audit log) to: {audit_path}")

    print("\nNormalisation complete.")
    