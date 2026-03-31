#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#normalisation.py
"""
Normalisation functions for zebrafish behavioural data.

This module contains functions for normalising behavioural parameters
across experiments using baseline group corrections.
"""


# -------------------------------------------------------------------------
# == IMPORT PACKAGES ==
# -------------------------------------------------------------------------
import glob
import json
import os
import sys

import pandas as pd
from typing import Optional, Union, List, Dict, Any, Tuple

from .config import (
    CORE_ID_COLS,
    ID_COLS,
    NORMALISED_SUFFIX,
    INFO_SUBDIR,
    AUDIT_FILENAME,
    METADATA_FILENAME,
    MASTER_SUMMARY_FILENAME
)



# -------------------------------------------------------------------------
# === HELPER FUNCTIONS ===
# -------------------------------------------------------------------------
    # helper functions (private, prefixed with _)


def _validate_input_data(
    df: pd.DataFrame,
    required_cols: List[str],
    time_windows: List[str],
    file_name: str
) -> bool:
    """
    Validate that a DataFrame has the required structure for normalisation.
    
    Checks for presence of core identifier columns and time window columns.
    Prints informative warnings if validation fails.
    
    Args:
        df (pd.DataFrame): DataFrame to validate.
        required_cols (List[str]): List of required column names (typically CORE_ID_COLS).
        time_windows (List[str]): List of time window column names that should contain data.
        file_name (str): Name of the file being validated (for error messages).
    
    Returns:
        bool: True if validation passes, False otherwise.
    
    Example:
        >>> is_valid = _validate_input_data(
        ...     df=my_dataframe,
        ...     required_cols=['grp', 'fish', 'date', 'box'],
        ...     time_windows=['avgDay', 'avgNight'],
        ...     file_name='sleepHours.csv'
        ... )
    """
    
    # Check for required ID columns
    missing_id_cols = [col for col in required_cols if col not in df.columns]
    if missing_id_cols:
        print(
            f"    Skipping {file_name}: Missing required ID columns: "
            f"{missing_id_cols}"
        )
        return False
    
    # Check for time window columns
    missing_time_windows = [tw for tw in time_windows if tw not in df.columns]
    if missing_time_windows:
        print(
            f"    Skipping {file_name}: Missing required time window columns: "
            f"{missing_time_windows}"
        )
        return False
    
    # Check for completely empty DataFrame
    if df.empty:
        print(f"    Skipping {file_name}: DataFrame is empty.")
        return False
    
    return True


def _calculate_baseline_means(
    df: pd.DataFrame,
    baseline_groups: List[str],
    time_windows: List[str],
    parameter: str,
    experiment_id: str
) -> Tuple[Dict[str, float], List[Dict]]:
    """
    Calculate mean values for baseline groups across time windows.
    
    For each combination of baseline group and time window, calculates
    the mean value that will be used for normalisation. Also generates
    audit records for tracking what baselines were calculated.
    
    Args:
        df (pd.DataFrame): DataFrame containing the behavioural data.
        baseline_groups (List[str]): List of group names to calculate means for
            (e.g., ['wt_untreated', 'mutant_untreated']).
        time_windows (List[str]): List of time window column names
            (e.g., ['avgDay', 'avgNight']).
        parameter (str): Name of the behavioural parameter being processed
            (e.g., 'sleepHours').
        experiment_id (str): Identifier for the current experiment
            (e.g., 'experiment_1').
    
    Returns:
        Tuple containing:
            - calculated_means (Dict[str, float]): Dictionary mapping 
              '{baseline}_{timewindow}' to mean value.
              Example: {'wt_untreated_avgDay': 12.5, 'wt_untreated_avgNight': 8.3}
            - audit_records (List[Dict]): List of dictionaries for audit logging,
              each containing experiment_id, parameter, time_window, 
              baseline_group, and calculated_mean.
    
    Example:
        >>> means, audit = _calculate_baseline_means(
        ...     df=my_df,
        ...     baseline_groups=['wt_untreated'],
        ...     time_windows=['avgDay', 'avgNight'],
        ...     parameter='sleepHours',
        ...     experiment_id='experiment_1'
        ... )
        >>> print(means)
        {'wt_untreated_avgDay': 12.5, 'wt_untreated_avgNight': 8.3}
    """
    
    calculated_means = {}
    audit_records = []
    
    # Calculate mean for each baseline group and time window combination
    for baseline in baseline_groups:
        for window in time_windows:
            # Filter to baseline group and calculate mean
            baseline_data = df[df['grp'] == baseline][window]
            mean_val = baseline_data.mean()
            
            # Store with key format: baseline_timewindow
            key = f"{baseline}_{window}"
            calculated_means[key] = mean_val
            
            # Create audit record for this calculation
            audit_records.append({
                'experiment_id': experiment_id,
                'parameter': parameter,
                'time_window': window,
                'baseline_group': baseline,
                'calculated_mean': mean_val
            })
    
    return calculated_means, audit_records


def _apply_normalisation(
    df: pd.DataFrame,
    baseline_map: Dict[str, str],
    baseline_means: Dict[str, float],
    time_windows: List[str],
    parameter: str
) -> pd.DataFrame:
    """
    Apply normalisation transformation by subtracting baseline means.
    
    For each group-timewindow combination, subtracts the appropriate baseline
    mean from individual fish values. Creates new columns with normalised data
    using the naming convention: {parameter}_{timewindow}.
    
    Args:
        df (pd.DataFrame): DataFrame containing raw behavioural data.
        baseline_map (Dict[str, str]): Mapping of target groups to their baseline groups.
            Example: {'wt_drug': 'wt_untreated', 'mutant_drug': 'mutant_untreated'}
        baseline_means (Dict[str, float]): Dictionary of calculated baseline means
            with keys in format '{baseline}_{timewindow}'.
            Example: {'wt_untreated_avgDay': 12.5}
        time_windows (List[str]): List of time window column names to normalise
            (e.g., ['avgDay', 'avgNight']).
        parameter (str): Name of the behavioural parameter being processed
            (e.g., 'sleepHours').
    
    Returns:
        pd.DataFrame: Copy of input DataFrame with new normalised columns added.
            New columns are named as '{parameter}_{timewindow}'.
            Example: 'sleepHours_avgDay', 'sleepHours_avgNight'
    
    Notes:
        - Only rows where 'grp' matches a key in baseline_map are normalised
        - Normalisation formula: normalised_value = raw_value - baseline_mean
        - Original columns are preserved
    
    Example:
        >>> df_normalised = _apply_normalisation(
        ...     df=my_df,
        ...     baseline_map={'wt_drug': 'wt_untreated'},
        ...     baseline_means={'wt_untreated_avgDay': 10.0},
        ...     time_windows=['avgDay'],
        ...     parameter='sleepHours'
        ... )
        >>> # df_normalised now has column 'sleepHours_avgDay' with normalised values
    """
    
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Process each time window
    for window in time_windows:
        # Create column name for normalised data (long format for internal use)
        normalised_col = f"{parameter}_{window}"
        
        # Initialise column with NaN
        df[normalised_col] = float('nan')
        
        # Apply normalisation for each target group
        for target_grp, baseline_grp in baseline_map.items():
            # Get the baseline mean for this combination
            baseline_key = f"{baseline_grp}_{window}"
            baseline_mean = baseline_means[baseline_key]
            
            # Apply normalisation: raw_value - baseline_mean
            mask = df['grp'] == target_grp
            df.loc[mask, normalised_col] = df.loc[mask, window] - baseline_mean
    
    return df


def _prepare_parameter_output(
    df: pd.DataFrame,
    parameter: str,
    time_windows: List[str],
    original_cols: List[str],
    experiment_id: str
) -> pd.DataFrame:
    """
    Format DataFrame for individual parameter CSV output.
    
    Prepares the normalised data for export to parameter-specific CSV files.
    Handles column ordering, renaming, and deduplication. The output format
    includes all original columns plus renamed normalised columns.
    
    Column naming convention for output:
    - Original columns: kept as-is (e.g., 'grp', 'fish', 'avgDay')
    - Normalised columns: renamed from '{parameter}_{timewindow}' to '{timewindow}_normalised'
      (e.g., 'sleepHours_avgDay' becomes 'avgDay_normalised')
    
    Args:
        df (pd.DataFrame): DataFrame with normalised data (output from _apply_normalisation).
        parameter (str): Name of the behavioural parameter (e.g., 'sleepHours').
        time_windows (List[str]): List of time window column names (e.g., ['avgDay', 'avgNight']).
        original_cols (List[str]): List of column names from the original input file.
        experiment_id (str): Identifier for the current experiment (e.g., 'experiment_1').
    
    Returns:
        pd.DataFrame: Formatted DataFrame ready for CSV export with columns:
            1. 'experiment' (first)
            2. All original input columns (in original order)
            3. Normalised columns with '_normalised' suffix
    
    Notes:
        - Duplicates are removed based on ID_COLS
        - 'experiment' column is added if not present
        - Column order is: experiment, original columns, normalised columns
    
    Example:
        >>> df_output = _prepare_parameter_output(
        ...     df=df_normalised,
        ...     parameter='sleepHours',
        ...     time_windows=['avgDay', 'avgNight'],
        ...     original_cols=['grp', 'fish', 'date', 'box', 'avgDay', 'avgNight'],
        ...     experiment_id='experiment_1'
        ... )
        >>> print(df_output.columns)
        ['experiment', 'grp', 'fish', 'date', 'box', 'avgDay', 'avgNight', 
         'avgDay_normalised', 'avgNight_normalised']
    """
    
    # Add experiment column if not present
    if 'experiment' not in df.columns:
        df = df.copy()
        df['experiment'] = experiment_id
    
    # Build list of normalised column names (long format: parameter_timewindow)
    normalised_cols_long = [f"{parameter}_{window}" for window in time_windows]
    
    # Check that normalised columns actually exist in the DataFrame
    missing_normalised_cols = [
        col for col in normalised_cols_long if col not in df.columns
        ]
    if missing_normalised_cols:
        raise ValueError(
        f"Expected normalised columns not found in DataFrame: {missing_normalised_cols}. "
        f"This may indicate an issue in the normalisation step."
        )
    
    # Create renaming map: long name -> short name with '_normalised' suffix
    rename_map = {
        f"{parameter}_{window}": f"{window}{NORMALISED_SUFFIX}"
        for window in time_windows
    }
    
    # Build final column order
        # 1. Start with 'experiment'
        # 2. Add original columns (excluding 'experiment' if it's there)
    cols_minus_experiment = [col for col in original_cols if col != 'experiment']
    final_cols = ['experiment'] + cols_minus_experiment
        # 3. Add normalised columns (still in long format for now)
    final_cols.extend(normalised_cols_long)
    
    # Select columns and check for duplicates
    df_output = df[final_cols]
    original_row_count = len(df_output)
    df_output = df_output.drop_duplicates(subset=ID_COLS)
    
    # Simple warning if duplicates were found
    if len(df_output) < original_row_count:
        rows_removed = original_row_count - len(df_output)
        print(
            f"    Warning: Removed {rows_removed} duplicate row(s) in {parameter} "
            f"for {experiment_id}. Check input data if unexpected."
        )
    
    # Rename normalised columns to short format
    df_output = df_output.rename(columns=rename_map)
    
    return df_output


def _prepare_summary_output(
    df: pd.DataFrame,
    parameter: str,
    time_windows: List[str],
    experiment_id: str
) -> pd.DataFrame:
    """
    Format DataFrame for master summary CSV output (wide format).
    
    Prepares data for the master summary file which combines all parameters
    across all experiments. Only includes ID columns and normalised values
    (raw data columns are excluded).
    
    Column naming convention for output:
    - ID columns: 'experiment', 'grp', 'fish', 'date', 'box'
    - Normalised columns: '{parameter}_{timewindow}' 
      (e.g., 'sleepHours_avgDay', 'activityPercentage_avgNight')
    
    Args:
        df (pd.DataFrame): DataFrame with normalised data (output from _apply_normalisation).
        parameter (str): Name of the behavioural parameter (e.g., 'sleepHours').
        time_windows (List[str]): List of time window column names (e.g., ['avgDay', 'avgNight']).
        experiment_id (str): Identifier for the current experiment (e.g., 'experiment_1').
    
    Returns:
        pd.DataFrame: Wide-format DataFrame with columns:
            - ID columns (experiment, grp, fish, date, box)
            - Normalised data columns with long names (parameter_timewindow)
    
    Notes:
        - Only normalised values are included (raw data excluded)
        - Duplicates are removed based on ID_COLS
        - Output is designed to be merged with other parameters
    
    Example:
        >>> df_summary = _prepare_summary_output(
        ...     df=df_normalised,
        ...     parameter='sleepHours',
        ...     time_windows=['avgDay', 'avgNight'],
        ...     experiment_id='experiment_1'
        ... )
        >>> print(df_summary.columns)
        ['experiment', 'grp', 'fish', 'date', 'box', 
         'sleepHours_avgDay', 'sleepHours_avgNight']
    """
    
    # Add experiment column if not present
    if 'experiment' not in df.columns:
        df = df.copy()
        df['experiment'] = experiment_id
    
    # Build list of normalised column names (long format: parameter_timewindow)
    normalised_cols = [f"{parameter}_{window}" for window in time_windows]
    
    # Check that normalised columns exist
    missing_cols = [col for col in normalised_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Expected normalised columns not found in DataFrame: {missing_cols}. "
            f"This may indicate an issue in the normalisation step."
        )
    
    # Define columns for master summary: ID columns + normalised columns only
    summary_cols = ID_COLS + normalised_cols
    
    # Select columns and remove duplicates
    df_summary = df[summary_cols].drop_duplicates(subset=ID_COLS)
    
    # Simple warning if duplicates were found
    original_row_count = len(df[summary_cols])
    if len(df_summary) < original_row_count:
        rows_removed = original_row_count - len(df_summary)
        print(
            f"    Warning: Removed {rows_removed} duplicate row(s) in summary output "
            f"for {parameter}, {experiment_id}. Check input data if unexpected."
        )
    
    return df_summary


def _generate_audit_logs(
    metadata_list: List[Dict],
    audit_data: List[Dict],
    info_subdir: str
) -> None:
    """
    Generate and export normalisation metadata and audit logs.
    
    Creates two files for documentation and reproducibility:
    1. Metadata JSON: Contains baseline mappings and calculated means for each experiment
    2. Audit CSV: Contains a row for each baseline mean calculation
    
    Args:
        metadata_list (List[Dict]): List of metadata dictionaries, one per experiment.
            Each dict contains: experiment_id, input_path, baseline_map, calculated_baselines.
        audit_data (List[Dict]): List of audit record dictionaries from baseline calculations.
            Each dict contains: experiment_id, parameter, time_window, baseline_group, calculated_mean.
        info_subdir (str): Path to the directory where audit files will be saved.
    
    Returns:
        None: Files are saved directly to disk.
    
    Notes:
        - Metadata JSON is useful for understanding normalisation configuration
        - Audit CSV is useful for verifying baseline calculations
        - Both files are saved in the info_subdir (typically 'normalisation_info/')
    
    Example:
        >>> _generate_audit_logs(
        ...     metadata_list=[{'experiment_id': 'exp_1', ...}],
        ...     audit_data=[{'experiment_id': 'exp_1', 'parameter': 'sleepHours', ...}],
        ...     info_subdir='/path/to/output/normalisation_info'
        ... )
        Saved normalisation metadata to: /path/to/output/normalisation_info/normalisation_metadata.json
        Saved normalisation audit log to: /path/to/output/normalisation_info/normalisation_audit.csv
    """
    
    # Define output file paths
    metadata_path = os.path.join(info_subdir, METADATA_FILENAME)
    audit_path = os.path.join(info_subdir, AUDIT_FILENAME)
    
    # Export metadata JSON
    with open(metadata_path, 'w') as f:
        json.dump(metadata_list, f, indent=4)
    print(f"Saved: {os.path.basename(metadata_path)} (in {INFO_SUBDIR}/)") #print(f"Saved normalisation metadata to: {metadata_path}")
    
    # Export audit CSV (if there's data to export)
    if audit_data:
        audit_df = pd.DataFrame(audit_data)
        audit_df.to_csv(audit_path, index=False)
        print(f"Saved: {os.path.basename(audit_path)} (in {INFO_SUBDIR}/)") #print(f"Saved normalisation audit log to: {audit_path}")
    else:
        print("Warning: No audit data to export. Skipping audit CSV creation.")



# -------------------------------------------------------------------------
# === PUBLIC FUNCTIONS ===
# -------------------------------------------------------------------------


def normalise_data(
    input_dirs: List[str],
    export_path: str,
    control_grouping: Union[str, Dict[str, str]],
    time_windows: List[str],
    parameters: Union[List[str], str] = 'all',
    experiment_names: Optional[List[str]] = None
) -> None:
    """
    Normalise behavioural data within experiments and aggregate results into parameter-specific files.
    
    Performs baseline normalisation by subtracting experiment-specific baseline means from individual
    fish values. Processes multiple experiments and aggregates results across experiments for each
    behavioural parameter. Generates both individual parameter CSV files and a master summary file.
    
    Args:
        input_dirs (List[str]): List of directories, where each directory contains all CSV files
            for one unique experiment/clutch.
        export_path (str): Path to the folder where normalised CSV files will be saved.
        control_grouping (Union[str, Dict[str, str]]): Defines the baseline group(s) for normalisation.
            - If str (e.g., 'wt_untreated'): All groups are normalised to this single baseline.
            - If dict: Defines custom map for multi-baseline normalisation.
              Example: {'wt_drug': 'wt_untreated', 'mutant_drug': 'mutant_untreated'}
        time_windows (List[str]): Column names containing the data to be normalised
            (e.g., ['avgDay', 'avgNight']).
        parameters (Union[List[str], str]): Behavioural parameters (CSV filename prefixes) to process.
            Use 'all' to process all CSV files found. Defaults to 'all'.
        experiment_names (Optional[List[str]]): Custom names for experiments (e.g., ['20240115_clutch1']).
            If None, experiments are named 'experiment_1', 'experiment_2', etc. Defaults to None.
    
    Returns:
        None: Files are saved directly to disk.
    
    Output Files:
        1. Individual parameter CSVs (in export_path/):
           - Named as '{parameter}_normalised.csv'
           - Contain all original columns plus normalised columns with '_normalised' suffix
           - Aggregate data across all experiments
        
        2. Master summary CSV (in export_path/normalisation_info/):
           - Named 'master_normalised_summary.csv'
           - Wide format with all parameters combined
           - Only includes ID columns and normalised data
        
        3. Audit files (in export_path/normalisation_info/):
           - 'normalisation_metadata.json': Configuration and baseline means
           - 'normalisation_audit.csv': Detailed log of baseline calculations
    
    Raises:
        SystemExit: If input directories don't exist or no valid data is found.
        ValueError: If data validation fails or required columns are missing.
    
    Notes:
        - Normalisation formula: normalised_value = raw_value - baseline_mean
        - Baseline means are calculated separately for each experiment
        - Only groups specified in control_grouping are normalised
        - Files without required columns are skipped with warnings
    
    Example:
        >>> # Single baseline normalisation
        >>> normalise_data(
        ...     input_dirs=['/path/to/exp1', '/path/to/exp2'],
        ...     export_path='/path/to/normalised',
        ...     control_grouping='wt_untreated',
        ...     time_windows=['avgDay', 'avgNight'],
        ...     parameters=['sleepHours', 'activityPercentage']
        ... )
        
        >>> # Multi-baseline normalisation
        >>> normalise_data(
        ...     input_dirs=['/path/to/exp1', '/path/to/exp2'],
        ...     export_path='/path/to/normalised',
        ...     control_grouping={'wt_drug': 'wt_untreated', 'mut_drug': 'mut_untreated'},
        ...     time_windows=['avgDay', 'avgNight'],
        ...     parameters='all'
        ... )
    """
    print("=" * 60)
    print("--- Starting Normalisation Process ---")
    print(f"--- processing {len(input_dirs)} experiment(s)")
    print("=" * 60)
    
    print(f"\nOutput directory:")
    print(f"  {export_path}")
    
    # =========================================================================
    # PHASE 1: SETUP AND ARGUMENT HANDLING
    # =========================================================================
    
    output_dir = export_path
    info_subdir = os.path.join(output_dir, INFO_SUBDIR)
    
    # Create output directories if needed
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    if not os.path.exists(info_subdir):
        os.makedirs(info_subdir)
        print(f"Created info directory: {info_subdir}")
    
    # Handle 'parameters' argument
    if isinstance(parameters, str) and parameters != 'all':
        parameters = [parameters]
    
    # Build baseline_map from control_grouping
    baseline_map = {}
    
    if isinstance(control_grouping, dict):
        baseline_map = control_grouping
    
    elif isinstance(control_grouping, str):
        # Single baseline: map all groups to this baseline
        single_baseline_group = control_grouping
        
        # Read a sample file to find all available groups
        try:
            sample_dir = input_dirs[0]
            search_path = os.path.join(sample_dir, "*.csv")
            sample_file_paths = glob.glob(search_path)
            
            if not sample_file_paths:
                sys.exit(f"Error: No CSV files found in sample directory: {sample_dir}")
            
            sample_df = pd.read_csv(sample_file_paths[0])
            
            if 'grp' not in sample_df.columns:
                sys.exit("Error: Sample CSV missing required 'grp' column for normalisation setup.")
            
            all_groups = sample_df['grp'].unique()
            
            # Map every group to the single baseline
            for group in all_groups:
                baseline_map[group] = single_baseline_group
        
        except Exception as e:
            sys.exit(f"Error reading sample file for group definition: {e}")
    
    if not baseline_map:
        sys.exit("Error: Could not generate a valid baseline map. Check control_grouping input.")
    
    # Initialize data containers
    metadata_list = []
    all_parameter_dfs: Dict[str, pd.DataFrame] = {}
    mean_audit_data = []
    master_summary_dfs_for_concat = []
    
    # =========================================================================
    # PHASE 2: PROCESS EACH EXPERIMENT
    # =========================================================================
    
    for i, dir_path in enumerate(input_dirs):
        # Generate experiment ID
        if experiment_names and i < len(experiment_names):
            experiment_id = experiment_names[i]
        else:
            experiment_id = f"experiment_{i + 1}"
        
        print(f"\nProcessing {experiment_id}") #: {os.path.basename(dir_path)}")
        
        # Find CSV files in this experiment directory
        search_path = os.path.join(dir_path, "*.csv")
        csv_files = glob.glob(search_path)
        
        # Filter by parameters if specified
        if parameters != 'all':
            csv_files = [f for f in csv_files if os.path.basename(f).split('_')[0] in parameters]
        
        if not csv_files:
            print(f"Warning: No matching files found in {dir_path}. Skipping.")
            continue
        
        # Initialize experiment metadata
        experiment_metadata = {
            'experiment_id': experiment_id,
            'input_path': dir_path,
            'baseline_map': baseline_map,
            'calculated_baselines': {}
        }
        
        # Initialize wide-format DataFrame for master summary (this experiment only)
        exp_wide_df_for_summary = None
        
        # Determine unique baseline groups required
        unique_baselines = list(set(baseline_map.values()))
        
        # ---------------------------------------------------------------------
        # Process each parameter file in this experiment
        # ---------------------------------------------------------------------
        
        for file_path in csv_files:
            file_name = os.path.basename(file_path)
            parameter = file_name.split('_')[0]
            
            print(f"  - Processing {file_name}")
            
            try:
                # Read CSV
                df = pd.read_csv(file_path)
                
                # Capture original columns before modifications
                original_input_cols = df.columns.tolist()
                
                # Validate input data structure
                if not _validate_input_data(df, CORE_ID_COLS, time_windows, file_name):
                    continue
                
                # Check baseline groups exist
                if not all(baseline in df['grp'].unique() for baseline in unique_baselines):
                    print(
                        f"    Skipping {file_name}: Missing one or more required "
                        f"baseline groups in 'grp' column."
                    )
                    continue
                
                # Calculate baseline means
                calculated_means, audit_records = _calculate_baseline_means(
                    df=df,
                    baseline_groups=unique_baselines,
                    time_windows=time_windows,
                    parameter=parameter,
                    experiment_id=experiment_id
                )
                
                # Store audit records and metadata
                mean_audit_data.extend(audit_records)
                experiment_metadata['calculated_baselines'][parameter] = calculated_means
                
                # Apply normalisation
                df_normalised = _apply_normalisation(
                    df=df,
                    baseline_map=baseline_map,
                    baseline_means=calculated_means,
                    time_windows=time_windows,
                    parameter=parameter
                )
                
                # Filter to only groups being normalised
                df_normalised = df_normalised[df_normalised['grp'].isin(baseline_map.keys())].copy()
                
                # Prepare for individual parameter CSV output
                df_for_parameter_csv = _prepare_parameter_output(
                    df=df_normalised,
                    parameter=parameter,
                    time_windows=time_windows,
                    original_cols=original_input_cols,
                    experiment_id=experiment_id
                )
                
                # Aggregate parameter data across experiments
                if parameter not in all_parameter_dfs:
                    all_parameter_dfs[parameter] = df_for_parameter_csv
                else:
                    all_parameter_dfs[parameter] = pd.concat([
                        all_parameter_dfs[parameter],
                        df_for_parameter_csv
                    ], ignore_index=True)
                
                # Prepare for master summary output
                df_for_summary = _prepare_summary_output(
                    df=df_normalised,
                    parameter=parameter,
                    time_windows=time_windows,
                    experiment_id=experiment_id
                )
                
                # Merge into experiment-wide summary
                if exp_wide_df_for_summary is None:
                    exp_wide_df_for_summary = df_for_summary
                else:
                    exp_wide_df_for_summary = pd.merge(
                        exp_wide_df_for_summary,
                        df_for_summary,
                        on=ID_COLS,
                        how='outer'
                    )
            
            except Exception as e:
                print(f"    Error processing {file_name}: {e}. Skipping file.")
                continue
        
        # Store experiment-wide summary
        if exp_wide_df_for_summary is not None:
            master_summary_dfs_for_concat.append(exp_wide_df_for_summary)
        
        # Store experiment metadata
        metadata_list.append(experiment_metadata)
    
    # =========================================================================
    # PHASE 3: EXPORT RESULTS
    # =========================================================================
    
    print("\n--- Finalising Output ---")
    
    if not all_parameter_dfs:
        print("Warning: No data was successfully processed. No output files created.")
        return
    
    # Export individual parameter CSVs
    for parameter, df_long in all_parameter_dfs.items():
        output_file_path = os.path.join(output_dir, f"{parameter}{NORMALISED_SUFFIX}.csv")
        df_long.to_csv(output_file_path, index=False)
        print(f"Saved: {os.path.basename(output_file_path)}") #print(f"Saved parameter file: {os.path.basename(output_file_path)}")
    
    # Export master summary CSV
    if master_summary_dfs_for_concat:
        master_summary_df = pd.concat(master_summary_dfs_for_concat, ignore_index=True)
        master_summary_path = os.path.join(info_subdir, MASTER_SUMMARY_FILENAME)
        master_summary_df.to_csv(master_summary_path, index=False)
        print(f"Saved: {os.path.basename(master_summary_path)} (in {INFO_SUBDIR}/)") #print(f"Saved master summary file to: {master_summary_path}")
    
    # Generate audit logs
    _generate_audit_logs(
        metadata_list=metadata_list,
        audit_data=mean_audit_data,
        info_subdir=info_subdir
    )
    
    print("\nNormalisation complete.")