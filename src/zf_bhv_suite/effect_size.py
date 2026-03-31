#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Effect size analysis functions for zebrafish behavioural data.

This module provides functions for calculating effect sizes using DABEST
(Data Analysis using Bootstrap-Coupled Estimation) with visualisation.

DABEST reference:
Ho, J., Tumkaya, T., Aryal, S., Choi, H., & Claridge-Chang, A. (2019).
Moving beyond P values: data analysis with estimation graphics.
Nature Methods, 16(7), 565-566.
"""

import pandas as pd
import os
import glob
import sys
import numpy as np
from typing import Optional, List, Dict, Any, Union
from PIL import Image, ImageDraw, ImageFont

import dabest
import matplotlib.pyplot as plt

from .config import CORE_ID_COLS


# =============================================================================
# HELPER FUNCTIONS (Private - prefixed with _)
# =============================================================================

def _extract_parameter_name(
    df: pd.DataFrame,
    file_path: str
) -> str:
    """
    Extract parameter name from DataFrame or filename.
    
    Attempts to extract parameter name from 'parameter' column in DataFrame.
    If column doesn't exist or is empty, extracts from filename.
    
    Args:
        df: DataFrame to check for 'parameter' column.
        file_path: File path to extract parameter from if needed.
    
    Returns:
        str: Parameter name.
    
    Raises:
        ValueError: If parameter cannot be determined from data or filename.
    
    Examples:
        >>> # From DataFrame with 'parameter' column
        >>> df = pd.DataFrame({'parameter': ['sleepHours', 'sleepHours']})
        >>> _extract_parameter_name(df, '/path/to/file.csv')
        'sleepHours'
        
        >>> # From filename when 'parameter' column missing
        >>> df = pd.DataFrame({'data': [1, 2, 3]})
        >>> _extract_parameter_name(df, '/path/to/sleepHours_250701_07.csv')
        'sleepHours'
        
        >>> # From normalised filename
        >>> df = pd.DataFrame({'data': [1, 2, 3]})
        >>> _extract_parameter_name(df, '/path/to/sleepHours_normalised.csv')
        'sleepHours'
    """
    
    # =========================================================================
    # METHOD 1: Try to get from 'parameter' column in DataFrame
    # =========================================================================
    
    if 'parameter' in df.columns:
        # Get unique parameter values
        param_values = df['parameter'].dropna().unique()
        
        if len(param_values) > 0:
            # Use the first non-null value
            parameter = str(param_values[0])
            return parameter
        else:
            # Column exists but is empty - fall through to filename method
            pass
    
    # =========================================================================
    # METHOD 2: Extract from filename
    # =========================================================================
    
    if file_path == 'DataFrame':
        # Can't extract from filename if input was a DataFrame
        raise ValueError(
            "Cannot determine parameter name: 'parameter' column is missing or empty "
            "and input was a DataFrame (no filename available). "
            "Please add a 'parameter' column to your DataFrame."
        )
    
    # Get the filename without path
    filename = os.path.basename(file_path)
    
    # Remove .csv extension
    filename_no_ext = filename.replace('.csv', '')
    
    # Extract parameter name by splitting on underscore and taking first part
    # Handles formats like:
    #   - sleepHours_250701_07.csv → sleepHours
    #   - sleepHours_normalised.csv → sleepHours
    #   - activityPercentage.csv → activityPercentage
    
    parts = filename_no_ext.split('_')
    parameter = parts[0]
    
    if not parameter:
        raise ValueError(
            f"Cannot determine parameter name from filename: {filename}. "
            f"Expected format: 'parameter_*.csv' or 'parameter.csv'"
        )
    
    return parameter


def _generate_colour_palette(
    control_group: str,
    treatment_groups: List[str],
    custom_palette: Optional[Dict[str, str]]
) -> Dict[str, str]:
    """
    Generate colour palette for groups with control as grey.
    
    If custom_palette is provided, uses it as-is. Otherwise generates a default
    palette where control group is mid-grey (#666666) and treatment groups get
    distinct colours from matplotlib's tab10 palette.
    
    Args:
        control_group: Name of control group.
        treatment_groups: List of treatment group names.
        custom_palette: Optional custom colour mapping. If provided, used directly.
    
    Returns:
        Dict[str, str]: Mapping of group names to hex colour codes.
            Example: {'wt': '#666666', 'drug1': '#1f77b4', 'drug2': '#ff7f0e'}
    
    Notes:
        - Default control colour is #666666 (mid-grey)
        - Treatment colours come from matplotlib's tab10 palette
        - If more than 10 treatment groups, colours will cycle
    """
    
    # If custom palette provided, use it as-is (trust the user)
    if custom_palette is not None:
        print("Using custom colour palette")
        return custom_palette
    
    # Generate default palette
    print("Generating default colour palette (control = grey)")
    
    # Start with control as grey
    colour_palette = {
        control_group: '#666666'  # Mid-grey for control
    }
    
    # Get matplotlib's tab10 colours for treatments
    import matplotlib.colors as mcolors
    tab10_colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Assign colours to treatment groups
    for i, treatment in enumerate(treatment_groups):
        # Cycle through colours if more than 10 treatments
        color_idx = i % len(tab10_colors)
        colour_palette[treatment] = tab10_colors[color_idx]
    
    # Warn if colour cycling will occur
    if len(treatment_groups) > 10:
        print(f"  Warning: {len(treatment_groups)} treatment groups detected. "
              f"Colours will cycle after the first 10 groups.")
    
    return colour_palette


def _generate_multi_group_colour_palette(
    comparison_pairs: Dict[str, str],
    custom_palette: Optional[Dict[str, str]]
) -> Dict[str, str]:
    """
    Generate colour palette for multi-group comparisons.
    
    If custom_palette is provided, uses it as-is. Otherwise generates a default
    palette with alternating grey controls and colourful treatments.
    
    Args:
        comparison_pairs: Dict mapping treatments to their controls.
        custom_palette: Optional custom colour mapping.
    
    Returns:
        Dict[str, str]: Mapping of group names to hex colour codes.
    
    Notes:
        - Controls get grey shades
        - Treatments get colours from matplotlib's tab10 palette
    """
    
    if custom_palette is not None:
        print("Using custom colour palette")
        return custom_palette
    
    print("Generating default colour palette (controls = grey shades, treatments = colourful)")
    
    import matplotlib.colors as mcolors
    tab10_colors = list(mcolors.TABLEAU_COLORS.values())
    
    # Grey shades for controls
    grey_shades = ['#666666', '#888888', '#aaaaaa', '#999999', '#777777']
    
    colour_palette = {}
    
    for i, (treatment, control) in enumerate(comparison_pairs.items()):
        # Assign grey to control
        grey_idx = i % len(grey_shades)
        colour_palette[control] = grey_shades[grey_idx]
        
        # Assign colour to treatment
        color_idx = i % len(tab10_colors)
        colour_palette[treatment] = tab10_colors[color_idx]
    
    return colour_palette


def _save_analysis_info(
    output_dir: str,
    control_group: str,
    treatment_groups: List[str],
    colour_palette: Dict[str, str]
) -> None:
    """
    Save analysis configuration to info.txt.
    
    Creates a text file documenting the groups analysed and their colour mappings
    for reference and reproducibility.
    
    Args:
        output_dir: Directory where info.txt will be saved.
        control_group: Name of control group.
        treatment_groups: List of treatment group names.
        colour_palette: Colour mapping for all groups.
    
    Returns:
        None: File saved to disk.
    """
    
    info_file_path = os.path.join(output_dir, 'info.txt')
    
    # Get DABEST version
    try:
        dabest_version = dabest.__version__
    except AttributeError:
        dabest_version = "unknown"
    
    with open(info_file_path, 'w') as f:
        f.write("DABEST Shared Control Analysis Configuration\n")
        f.write("=" * 50 + "\n\n")
        
        # DABEST version info
        f.write(f"DABEST Version: {dabest_version}\n")
        f.write(f"Analysis Method: Shared Control (Bootstrap-Coupled Estimation)\n")
        f.write(f"Confidence Interval: 95%\n\n")
        
        # Control group
        f.write("Control Group:\n")
        f.write(f"  - {control_group}\n\n")
        
        # Treatment groups
        f.write("Treatment Groups:\n")
        for treatment in treatment_groups:
            f.write(f"  - {treatment}\n")
        f.write("\n")
        
        # Colour palette
        f.write("Colour Palette:\n")
        for group, colour in colour_palette.items():
            f.write(f"  - {group}: {colour}\n")
    
    print(f"Analysis info saved to: {info_file_path}")


def _save_multi_group_analysis_info(
    output_dir: str,
    comparison_pairs: Dict[str, str],
    colour_palette: Dict[str, str]
) -> None:
    """
    Save multi-group analysis configuration to info.txt.
    
    Creates a text file documenting the comparison pairs and their colour mappings
    for reference and reproducibility.
    
    Args:
        output_dir: Directory where info.txt will be saved.
        comparison_pairs: Treatment-control pairs.
        colour_palette: Colour mapping for all groups.
    
    Returns:
        None: File saved to disk.
    """
    
    info_file_path = os.path.join(output_dir, 'info.txt')
    
    # Get DABEST version
    try:
        dabest_version = dabest.__version__
    except AttributeError:
        dabest_version = "unknown"
    
    with open(info_file_path, 'w') as f:
        f.write("DABEST Multi-Group Analysis Configuration\n")
        f.write("=" * 50 + "\n\n")
        
        # DABEST version info
        f.write(f"DABEST Version: {dabest_version}\n")
        f.write(f"Analysis Method: Multi-Group (Bootstrap-Coupled Estimation)\n")
        f.write(f"Confidence Interval: 95%\n\n")
        
        # Comparison pairs
        f.write("Comparison Pairs:\n")
        for treatment, control in comparison_pairs.items():
            f.write(f"  {treatment} vs {control}\n")
        f.write("\n")
        
        # Colour palette
        f.write("Colour Palette:\n")
        for group, colour in colour_palette.items():
            f.write(f"  - {group}: {colour}\n")
    
    print(f"Analysis info saved to: {info_file_path}")


def _generate_legend_image(
    output_dir: str,
    colour_palette: Dict[str, str]
) -> None:
    """
    Generate visual legend showing colour mappings.
    
    Creates a PNG image (legend.png) displaying each group name with its
    corresponding colour square for easy reference.
    
    Args:
        output_dir: Directory where legend.png will be saved.
        colour_palette: Colour mapping for all groups.
    
    Returns:
        None: Image saved to disk.
    
    Notes:
        - Attempts to use Arial font; falls back to default if not available
        - Image dimensions adjust based on number of groups
        - Each group gets a coloured square + text label
    """
    
    legend_image_path = os.path.join(output_dir, 'legend.png')
    
    # Image layout configuration
    item_height = 30      # Height of each group's entry
    padding = 10          # Padding around elements
    font_size = 18        # Font size for text
    square_size = 20      # Size of the colour square
    image_width = 400     # Fixed width
    
    # Calculate image height based on number of groups
    image_height = len(colour_palette) * item_height + 2 * padding
    
    # Create a blank white image
    img = Image.new('RGB', (image_width, image_height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Try to load a system font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        try:
            # Alternative: Try Arial on macOS
            font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", font_size)
        except IOError:
            # Fall back to default font
            font = ImageFont.load_default()
            print("  Note: Using default font for legend (Arial not found)")
    
    # Draw each group with its colour
    y_offset = padding
    for group, colour_hex in colour_palette.items():
        # Draw the colour square
        square_top_left = (padding, y_offset + (item_height - square_size) // 2)
        square_bottom_right = (
            padding + square_size,
            y_offset + (item_height - square_size) // 2 + square_size
        )
        draw.rectangle(
            [square_top_left, square_bottom_right],
            fill=colour_hex
        )
        
        # Draw the group name
        text_position = (
            padding + square_size + padding,
            y_offset + (item_height - font_size) // 2
        )
        draw.text(
            text_position,
            group,
            fill="black",
            font=font
        )
        
        y_offset += item_height
    
    # Save the image
    img.save(legend_image_path)
    print(f"Legend image saved to: {legend_image_path}")


def _discover_parameter_files(
    input_data: Union[str, List[str], pd.DataFrame],
    parameters: Union[List[str], str],
    time_windows: List[str]
    ) -> List[Dict[str, Any]]:
    """
    Discover and validate files to analyse.
    
    Handles different input types (file/list/folder/DataFrame) and filters
    based on parameters specification. Validates that each file has required
    columns.
    
    Args:
        input_data: Input data (file path, list of paths, folder path, or DataFrame).
        parameters: 'all' or list of parameter names to filter.
        time_windows: List of time window column names (for validation).
    
    Returns:
        List[Dict[str, Any]]: List of dicts, each containing:
            - 'path': File path (or 'DataFrame' if input was DataFrame)
            - 'dataframe': Loaded pandas DataFrame
            - 'parameter': Parameter name extracted from data or filename
    
    Raises:
        ValueError: If input type is invalid or no valid files found.
        FileNotFoundError: If specified file or folder doesn't exist.
    """
    
    files_to_analyse = []
    
    # =========================================================================
    # STEP 1: BUILD LIST OF FILES TO CHECK
    # =========================================================================
    
    csv_paths = []  # Will hold list of file paths to check
    
    if isinstance(input_data, pd.DataFrame):
        # Input is a DataFrame - process directly (no file paths)
        print("Input is a DataFrame")
        # We'll handle this separately after the file processing
        
    elif isinstance(input_data, str):
        if os.path.isdir(input_data):
            # Input is a folder
            print(f"Scanning folder: {input_data}")
            csv_paths = glob.glob(os.path.join(input_data, "*.csv"))
            
            if not csv_paths:
                raise ValueError(f"No CSV files found in folder: {input_data}")
            
            print(f"Found {len(csv_paths)} CSV file(s) in folder")
            
        elif os.path.isfile(input_data):
            # Input is a single file
            print(f"Processing single file: {os.path.basename(input_data)}")
            csv_paths = [input_data]
            
        else:
            raise FileNotFoundError(f"Input path does not exist: {input_data}")
    
    elif isinstance(input_data, list):
        # Input is a list of file paths
        print(f"Processing list of {len(input_data)} file(s)")
        
        # Validate all files exist
        for file_path in input_data:
            if not os.path.isfile(file_path):
                print(f"Warning: File not found: {file_path}. Skipping.")
            else:
                csv_paths.append(file_path)
        
        if not csv_paths:
            raise ValueError("No valid files found in the provided list")
    
    else:
        raise ValueError(
            f"input_data must be a file path, folder path, list of paths, or DataFrame. "
            f"Got: {type(input_data)}"
        )
    
    # =========================================================================
    # STEP 2: PROCESS FILES (validate and filter)
    # =========================================================================
    
    for csv_path in csv_paths:
        try:
            # Load the CSV
            df = pd.read_csv(csv_path)
            
            # --- VALIDATION: Check required columns ---
            required_cols = set(CORE_ID_COLS + time_windows)
            missing_cols = required_cols - set(df.columns)
            
            if missing_cols:
                print(f"  Skipping {os.path.basename(csv_path)}: Missing columns {missing_cols}")
                continue
            
            # --- EXTRACT PARAMETER NAME ---
            param_name = _extract_parameter_name(df, csv_path)
            
            # --- FILTERING: Apply parameter filter if specified ---
            if parameters != 'all':
                # Check if this parameter should be included
                if 'parameter' in df.columns:
                    # Check parameter column value
                    param_value = df['parameter'].iloc[0]
                    if param_value not in parameters:
                        # This parameter not in the filter list - skip silently
                        continue
                else:
                    # No 'parameter' column - use extracted name from filename
                    if param_name not in parameters:
                        continue
            
            # --- FILE PASSED ALL CHECKS - ADD TO ANALYSIS LIST ---
            files_to_analyse.append({
                'path': csv_path,
                'dataframe': df,
                'parameter': param_name
            })
            
            print(f"  ✓ {os.path.basename(csv_path)} → parameter: {param_name}")
            
        except Exception as e:
            print(f"  Error reading {os.path.basename(csv_path)}: {e}. Skipping.")
            continue
    
    # =========================================================================
    # STEP 3: HANDLE DATAFRAME INPUT (if applicable)
    # =========================================================================
    
    if isinstance(input_data, pd.DataFrame):
        df = input_data
        
        # Validate required columns
        required_cols = set(CORE_ID_COLS + time_windows)
        missing_cols = required_cols - set(df.columns)
        
        if missing_cols:
            raise ValueError(f"DataFrame is missing required columns: {missing_cols}")
        
        # Extract parameter name
        param_name = _extract_parameter_name(df, "DataFrame")
        
        # Apply parameter filter if specified
        if parameters != 'all':
            if 'parameter' in df.columns:
                param_value = df['parameter'].iloc[0]
                if param_value not in parameters:
                    raise ValueError(
                        f"DataFrame parameter '{param_value}' not in specified parameters list: {parameters}"
                    )
            else:
                raise ValueError(
                    "Cannot filter DataFrame by parameters: 'parameter' column not found. "
                    "Either add 'parameter' column or use parameters='all'."
                )
        
        files_to_analyse.append({
            'path': 'DataFrame',
            'dataframe': df,
            'parameter': param_name
        })
        
        print(f"  ✓ DataFrame → parameter: {param_name}")
    
    # =========================================================================
    # STEP 4: FINAL VALIDATION
    # =========================================================================
    
    if not files_to_analyse:
        raise ValueError(
            "No valid files found to analyse. Possible reasons:\n"
            "  - Files missing required columns (grp, fish, date, box + time_windows)\n"
            "  - Parameter filter excluded all files\n"
            "  - Files could not be read"
        )
    
    print(f"\n→ Total files to analyse: {len(files_to_analyse)}")
    
    return files_to_analyse


def _run_dabest_analysis(
    df: pd.DataFrame,
    parameter: str,
    time_window: str,
    control_group: str,
    treatment_groups: List[str],
    colour_palette: Dict[str, str],
    output_dir: str,
    stats_dir: Optional[str],
    plot_width: int,
    plot_height: int,
    raw_marker_size: int,
    contrast_marker_size: int
) -> Dict[str, Any]:
    """
    Run DABEST analysis for one parameter-timewindow combination.
    
    Performs bootstrap-coupled estimation, generates effect size plot, and
    extracts statistical results. Saves plot and optionally detailed statistics.
    
    Args:
        df: DataFrame containing the data to analyse.
        parameter: Parameter name (for labelling).
        time_window: Time window column name to analyse.
        control_group: Name of control group.
        treatment_groups: List of treatment group names.
        colour_palette: Colour mapping for groups.
        output_dir: Directory to save plot.
        stats_dir: Directory to save detailed stats (None if not saving).
        plot_width: Plot width in inches.
        plot_height: Plot height in inches.
    
    Returns:
        Dict[str, Any]: Result dictionary containing:
            - 'parameter': Parameter name
            - 'time_window': Time window analysed
            - 'plot_path': Path to saved plot (PDF)
            - 'stats_tests': DABEST statistical tests DataFrame
    
    Raises:
        Exception: If DABEST analysis fails (caught and logged by caller).
    
    Notes:
        - Uses DABEST's mean_diff with 95% CI
        - Plot saved as PDF for publication quality
        - Only statistical tests saved (not results, as tests contain more detail)
    """
    
    # =========================================================================
    # STEP 1: PREPARE DATA
    # =========================================================================
    
    # Build list of all groups (control + treatments)
    all_groups = [control_group] + treatment_groups
    
    # Filter DataFrame to only include the groups we're analysing
    df_filtered = df[df['grp'].isin(all_groups)].copy()
    
    if df_filtered.empty:
        raise ValueError(
            f"No data found for groups: {all_groups}. "
            f"Check that group names match the 'grp' column in your data."
        )
    
    # =========================================================================
    # STEP 2: RUN DABEST ANALYSIS
    # =========================================================================
    
    # Load data into DABEST
    # idx is a tuple: (control, treatment1, treatment2, ...)
    shared_control_dabest = dabest.load(
        df_filtered,
        idx=tuple(all_groups),  # Control first, then treatments
        x="grp",
        y=time_window,
        ci=95
    )
    
    # Calculate mean difference with bootstrap
    mean_diff = shared_control_dabest.mean_diff
    
    # =========================================================================
    # STEP 3: GENERATE AND SAVE PLOT
    # =========================================================================
    
    # Construct y-axis label (auto-generated, always)
    raw_axis_label = f"{parameter}_{time_window}"
    
    # Generate DABEST plot
    try:
        fig = mean_diff.plot(
            custom_palette=colour_palette,
            raw_marker_size=raw_marker_size,        # User-configurable
            raw_desat=0.9,
            contrast_desat=0.9,
            contrast_marker_size=contrast_marker_size,  # User-configurable
            raw_alpha=1,
            swarm_side="center",
            raw_label=raw_axis_label  # Auto-generated
        )
    except KeyError as e:
        # This happens when colour palette is missing a group
        raise ValueError(
            f"Colour palette error: {e}\n"
            f"The colour palette is missing one or more groups present in the data.\n"
            f"Groups in colour palette: {list(colour_palette.keys())}\n"
            f"Groups in data: {all_groups}\n"
            f"Make sure your custom colour palette includes all groups."
        )
    
    # Set figure size
    fig.set_size_inches(plot_width, plot_height)
    
    # Add title
    plt.title(f"{parameter} - {time_window}")
    
    # Save plot as PDF (publication quality, vector graphics)
    plot_filename = f"dabest_{parameter}_{time_window}.pdf"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, format='pdf', bbox_inches="tight")
    plt.close(fig)
    
    print(f"    Plot saved: {plot_filename}")
    
    # =========================================================================
    # STEP 4: EXTRACT STATISTICS
    # =========================================================================
    
    # Get statistical tests (contains all necessary info including results)
    stats_tests_df = mean_diff.statistical_tests
    
    # =========================================================================
    # STEP 5: SAVE DETAILED STATISTICS (if requested)
    # =========================================================================
    
    if stats_dir is not None:
        # Save detailed tests only
        tests_filename = f"stats_test_{parameter}_{time_window}.csv"
        tests_path = os.path.join(stats_dir, tests_filename)
        stats_tests_df.to_csv(tests_path, index=False)
        
        print(f"    Stats saved: {tests_filename}")
    
    # =========================================================================
    # STEP 6: RETURN RESULT METADATA
    # =========================================================================
    
    return {
        'parameter': parameter,
        'time_window': time_window,
        'plot_path': plot_path,
        'stats_tests': stats_tests_df  # Only tests, not results
    }


def _run_dabest_analysis_multi_group(
    df: pd.DataFrame,
    parameter: str,
    time_window: str,
    comparison_pairs: Dict[str, str],
    colour_palette: Dict[str, str],
    output_dir: str,
    stats_dir: Optional[str],
    plot_width: int,
    plot_height: int,
    raw_marker_size: int,
    contrast_marker_size: int
) -> Dict[str, Any]:
    """
    Run DABEST multi-group analysis for one parameter-timewindow combination.
    
    Performs bootstrap-coupled estimation for multiple treatment-control pairs,
    generates effect size plot, and extracts statistical results.
    
    Args:
        df: DataFrame containing the data to analyse.
        parameter: Parameter name (for labelling).
        time_window: Time window column name to analyse.
        comparison_pairs: Dict mapping treatment groups to their control groups.
        colour_palette: Colour mapping for groups.
        output_dir: Directory to save plot.
        stats_dir: Directory to save detailed stats (None if not saving).
        plot_width: Plot width in inches.
        plot_height: Plot height in inches.
        raw_marker_size: Size of individual data point markers.
        contrast_marker_size: Size of effect size markers.
    
    Returns:
        Dict[str, Any]: Result dictionary containing:
            - 'parameter': Parameter name
            - 'time_window': Time window analysed
            - 'plot_path': Path to saved plot (PDF)
            - 'stats_tests': DABEST statistical tests DataFrame
    
    Raises:
        Exception: If DABEST analysis fails (caught and logged by caller).
    
    Notes:
        - Uses DABEST's mean_diff with 95% CI
        - Each treatment compared only to its paired control
        - Plot saved as PDF for publication quality
    """
    
    # =========================================================================
    # STEP 1: PREPARE DATA
    # =========================================================================
    
    # Build list of all groups needed (remove duplicates, preserve order)
    all_groups = []
    for treatment, control in comparison_pairs.items():
        all_groups.extend([control, treatment])
    all_groups = list(dict.fromkeys(all_groups))  # Remove duplicates, preserve order
    
    # Filter DataFrame to only include the groups we're analysing
    df_filtered = df[df['grp'].isin(all_groups)].copy()
    
    if df_filtered.empty:
        raise ValueError(
            f"No data found for groups: {all_groups}. "
            f"Check that group names match the 'grp' column in your data."
        )
    
    # =========================================================================
    # STEP 2: RUN DABEST ANALYSIS
    # =========================================================================
    
    # Build idx as list of tuples: [(control1, treatment1), (control2, treatment2), ...]
    idx_tuples = [(control, treatment) for treatment, control in comparison_pairs.items()]
    
    # Load data into DABEST
    multi_group_dabest = dabest.load(
        df_filtered,
        idx=idx_tuples,  # List of tuples for multi-group
        x="grp",
        y=time_window,
        ci=95
    )
    
    # Calculate mean difference with bootstrap
    mean_diff = multi_group_dabest.mean_diff
    
    # =========================================================================
    # STEP 3: GENERATE AND SAVE PLOT
    # =========================================================================
    
    # Construct y-axis label (auto-generated, always)
    raw_axis_label = f"{parameter}_{time_window}"
    
    try:
        # Generate DABEST plot
        fig = mean_diff.plot(
            custom_palette=colour_palette,
            raw_marker_size=raw_marker_size,
            raw_desat=0.9,
            contrast_desat=0.9,
            contrast_marker_size=contrast_marker_size,
            raw_alpha=1,
            swarm_side="center",
            raw_label=raw_axis_label
        )
    except KeyError as e:
        # This happens when colour palette is missing a group
        raise ValueError(
            f"Colour palette error: {e}\n"
            f"The colour palette is missing one or more groups present in the data.\n"
            f"Groups in colour palette: {list(colour_palette.keys())}\n"
            f"Groups in data: {all_groups}\n"
            f"Make sure your custom colour palette includes all groups."
        )
    
    # Set figure size
    fig.set_size_inches(plot_width, plot_height)
    
    # Add title
    plt.title(f"{parameter} - {time_window}")
    
    # Save plot as PDF (publication quality, vector graphics)
    plot_filename = f"dabest_{parameter}_{time_window}.pdf"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, format='pdf', bbox_inches="tight")
    plt.close(fig)
    
    print(f"    Plot saved: {plot_filename}")
    
    # =========================================================================
    # STEP 4: EXTRACT STATISTICS
    # =========================================================================
    
    # Get statistical tests
    stats_tests_df = mean_diff.statistical_tests
    
    # =========================================================================
    # STEP 5: SAVE DETAILED STATISTICS (if requested)
    # =========================================================================
    
    if stats_dir is not None:
        # Save detailed tests only
        tests_filename = f"stats_test_{parameter}_{time_window}.csv"
        tests_path = os.path.join(stats_dir, tests_filename)
        stats_tests_df.to_csv(tests_path, index=False)
        
        print(f"    Stats saved: {tests_filename}")
    
    # =========================================================================
    # STEP 6: RETURN RESULT METADATA
    # =========================================================================
    
    return {
        'parameter': parameter,
        'time_window': time_window,
        'plot_path': plot_path,
        'stats_tests': stats_tests_df
    }


def _analyse_single_parameter(
    file_info: Dict[str, Any],
    time_windows: List[str],
    control_group: str,
    treatment_groups: List[str],
    colour_palette: Dict[str, str],
    output_dir: str,
    stats_dir: Optional[str],
    plot_width: int,
    plot_height: int,
    raw_marker_size: int,
    contrast_marker_size: int
) -> List[Dict[str, Any]]:
    """
    Analyse one parameter file across all time windows.
    
    Loops through each time window and runs DABEST analysis, generating plots
    and statistics for each parameter-timewindow combination.
    
    Args:
        file_info: Dict containing 'dataframe', 'parameter', and 'path'.
        time_windows: List of time window column names to analyse.
        control_group: Name of control group.
        treatment_groups: List of treatment group names.
        colour_palette: Colour mapping for groups.
        output_dir: Main output directory for plots.
        stats_dir: Directory for detailed stats (None if not saving).
        plot_width: Plot width in inches.
        plot_height: Plot height in inches.
    
    Returns:
        List[Dict[str, Any]]: List of result dicts, one per time window analysed.
            Each dict contains analysis metadata and statistics.
    """
    
    df = file_info['dataframe']
    parameter = file_info['parameter']
    file_path = file_info['path']
    
    print(f"\n--- Analysing parameter: {parameter} ---")
    if file_path != 'DataFrame':
        print(f"    Source: {os.path.basename(file_path)}")
    
    results = []
    
    # Loop through each time window
    for time_window in time_windows:
        print(f"  → Time window: {time_window}")
        
        # Validate that this time window column exists
        if time_window not in df.columns:
            print(f"    Warning: Column '{time_window}' not found in data. Skipping.")
            continue
        
        # Check if column has data
        if df[time_window].isna().all():
            print(f"    Warning: Column '{time_window}' contains only NaN values. Skipping.")
            continue
        
        try:
            # Run DABEST analysis for this parameter-timewindow combination
            result = _run_dabest_analysis(
                df=df,
                parameter=parameter,
                time_window=time_window,
                control_group=control_group,
                treatment_groups=treatment_groups,
                colour_palette=colour_palette,
                output_dir=output_dir,
                stats_dir=stats_dir,
                plot_width=plot_width,
                plot_height=plot_height,
                raw_marker_size=raw_marker_size,          
                contrast_marker_size=contrast_marker_size  
            )
            
            results.append(result)
            print(f"    ✓ Analysis complete")
            
        except Exception as e:
            print(f"    ✗ Error during analysis: {e}")
            # Continue with next time window instead of failing completely
            continue
    
    if not results:
        print(f"  Warning: No analyses completed for {parameter}")
    else:
        print(f"  → Completed {len(results)}/{len(time_windows)} time window(s)")
    
    return results


def _analyse_single_parameter_multi_group(
    file_info: Dict[str, Any],
    time_windows: List[str],
    comparison_pairs: Dict[str, str],
    colour_palette: Dict[str, str],
    output_dir: str,
    stats_dir: Optional[str],
    plot_width: int,
    plot_height: int,
    raw_marker_size: int,
    contrast_marker_size: int
) -> List[Dict[str, Any]]:
    """
    Analyse one parameter file across all time windows (multi-group version).
    
    Loops through each time window and runs DABEST multi-group analysis.
    
    Args:
        file_info: Dict containing 'dataframe', 'parameter', and 'path'.
        time_windows: List of time window column names to analyse.
        comparison_pairs: Dict mapping treatment groups to control groups.
        colour_palette: Colour mapping for groups.
        output_dir: Main output directory for plots.
        stats_dir: Directory for detailed stats (None if not saving).
        plot_width: Plot width in inches.
        plot_height: Plot height in inches.
        raw_marker_size: Size of individual data point markers.
        contrast_marker_size: Size of effect size markers.
    
    Returns:
        List[Dict[str, Any]]: List of result dicts, one per time window analysed.
    """
    
    df = file_info['dataframe']
    parameter = file_info['parameter']
    file_path = file_info['path']
    
    print(f"\n--- Analysing parameter: {parameter} ---")
    if file_path != 'DataFrame':
        print(f"    Source: {os.path.basename(file_path)}")
    
    results = []
    
    # Loop through each time window
    for time_window in time_windows:
        print(f"  → Time window: {time_window}")
        
        # Validate that this time window column exists
        if time_window not in df.columns:
            print(f"    Warning: Column '{time_window}' not found in data. Skipping.")
            continue
        
        # Check if column has data
        if df[time_window].isna().all():
            print(f"    Warning: Column '{time_window}' contains only NaN values. Skipping.")
            continue
        
        try:
            # Run DABEST multi-group analysis for this parameter-timewindow combination
            result = _run_dabest_analysis_multi_group(
                df=df,
                parameter=parameter,
                time_window=time_window,
                comparison_pairs=comparison_pairs,
                colour_palette=colour_palette,
                output_dir=output_dir,
                stats_dir=stats_dir,
                plot_width=plot_width,
                plot_height=plot_height,
                raw_marker_size=raw_marker_size,
                contrast_marker_size=contrast_marker_size
            )
            
            results.append(result)
            print(f"    ✓ Analysis complete")
            
        except Exception as e:
            print(f"    ✗ Error during analysis: {e}")
            # Continue with next time window instead of failing completely
            continue
    
    if not results:
        print(f"  Warning: No analyses completed for {parameter}")
    else:
        print(f"  → Completed {len(results)}/{len(time_windows)} time window(s)")
    
    return results


def _create_summary_csvs(
    all_results: List[Dict[str, Any]],
    output_dir: str,
    stats_dir: Optional[str]
) -> None:
    """
    Consolidate individual statistics into summary CSV file.
    
    Merges statistics from all parameter-timewindow analyses into a comprehensive
    summary file for easy review and further analysis.
    
    Args:
        all_results: List of result dicts from all analyses.
        output_dir: Main output directory for summary file.
        stats_dir: Directory containing detailed stats (for reference, not used).
    
    Returns:
        None: Summary CSV file saved to disk.
    
    Output Files:
        - dabestStats_TESTS_summary.csv: Consolidated statistical tests
          with columns: parameter, time_window, control, test, result, 
          difference, ci, bca_low, bca_high, pvalue_*, etc.
    
    Notes:
        - 'result' column formatted as: "difference (CI% CI low - high)"
        - All numerical values formatted to 3 significant figures
        - Rows ordered by parameter, then time_window
    """
    
    print("\n--- Creating summary statistics ---")
    
    if not all_results:
        print("No results to summarise.")
        return
    
    # =========================================================================
    # STEP 1: COLLECT AND PROCESS INDIVIDUAL DATAFRAMES
    # =========================================================================
    
    summary_dfs = []
    
    for result in all_results:
        parameter = result['parameter']
        time_window = result['time_window']
        stats_tests_df = result['stats_tests'].copy()
        
        # Add metadata columns at the beginning
        stats_tests_df.insert(0, 'time_window', time_window)
        stats_tests_df.insert(0, 'parameter', parameter)
        
        # Create formatted 'result' column
        # Format: "difference (CI% CI low - high)"
        # Example: "2.45 (95% CI 1.23 - 3.67)"
        stats_tests_df['result'] = stats_tests_df.apply(
            lambda row: f"{row['difference']:.3g} ({row['ci']:.0f}% CI {row['bca_low']:.3g} - {row['bca_high']:.3g})",
            axis=1
        )
        
        # Reorder columns to put 'result' after 'test'
        if 'test' in stats_tests_df.columns:
            cols = list(stats_tests_df.columns)
            # Remove 'result' from wherever it is
            cols.remove('result')
            # Find position of 'test' column
            test_idx = cols.index('test')
            # Insert 'result' right after 'test'
            cols.insert(test_idx + 1, 'result')
            # Reorder DataFrame
            stats_tests_df = stats_tests_df[cols]
        
        summary_dfs.append(stats_tests_df)
    
    # =========================================================================
    # STEP 2: CONCATENATE ALL DATAFRAMES
    # =========================================================================
    
    final_summary_df = pd.concat(summary_dfs, ignore_index=True)
    
    # =========================================================================
    # STEP 3: SAVE SUMMARY CSV
    # =========================================================================
    
    summary_filename = "dabestStats_TESTS_summary.csv"
    summary_path = os.path.join(output_dir, summary_filename)
    final_summary_df.to_csv(summary_path, index=False)
    
    print(f"Summary statistics saved: {summary_filename}")
    print(f"  Total comparisons: {len(final_summary_df)}")
    print(f"  Parameters analysed: {final_summary_df['parameter'].nunique()}")



# =============================================================================
# MAIN PUBLIC FUNCTIONS
# =============================================================================

# TODO: Consider refactoring shared_control and multi_group to reduce
#       duplication once usage patterns are established. Possible approaches:
#       1. Unified _analyse_single_parameter() helper
#       2. Shared _run_dabest_analysis_core() for plotting/stats
#       3. Unified _generate_colour_palette()


def dabest_shared_control(
    input_data: Union[str, List[str], pd.DataFrame],
    time_windows: List[str],
    control_group: str,
    treatment_groups: List[str],
    output_dir: str,
    parameters: Union[List[str], str] = 'all',
    colour_palette: Optional[Dict[str, str]] = None,
    plot_width: int = 3,
    plot_height: int = 6,
    raw_marker_size: int = 4,
    contrast_marker_size: int = 8,
    save_detailed_stats: bool = True,
    save_stats: bool = True  # NEW
) -> Dict[str, Any]:
    """
    Perform DABEST shared control analysis on behavioural data.
    
    Compares multiple treatment groups to a single shared control group using
    bootstrap-coupled estimation statistics. Generates effect size plots and
    statistical summaries for each parameter and time window combination.
    
    This function is a wrapper around the DABEST package 
    (https://github.com/ACCLAB/DABESTPY) for convenient use with zebrafish 
    behavioural data.
    
    Args:
        input_data (Union[str, List[str], pd.DataFrame]): Input data, can be:
            - Path to a single CSV file
            - Path to a directory containing CSV files
            - List of paths to CSV files
            - pandas DataFrame
        time_windows (List[str]): Column names to analyse (e.g., ['avgDay', 'avgNight']).
            These columns must exist in the input data.
        control_group (str): Name of the control group (first comparison group).
        treatment_groups (List[str]): Names of treatment groups to compare against control.
        output_dir (str): Directory where plots and statistics will be saved.
            Will be created if it doesn't exist.
        parameters (Union[List[str], str]): Parameters to analyse. Options:
            - 'all': Analyse all CSV files with required columns (default)
            - List[str]: Only analyse CSVs where 'parameter' column matches list
        colour_palette (Optional[Dict[str, str]]): Custom colour mapping for groups.
            Format: {group_name: hex_colour}. If None, generates default palette
            with control as grey (#666666) and treatments as distinct colours.
        plot_width (int): Width of generated plots in inches. Defaults to 5.
        plot_height (int): Height of generated plots in inches. Defaults to 6.
        raw_marker_size (int): Size of individual data point markers. Defaults to 4.
        contrast_marker_size (int): Size of effect size markers. Defaults to 8.
        save_detailed_stats (bool): If True, saves detailed statistics for each
            parameter-timewindow combination to 'detailed_stats/' subfolder.
            If False, only saves summary statistics. Defaults to True.
        save_stats (bool): If True, saves all statistical outputs (detailed and/or summary).
            If False, only generates plots. Useful for iterative plot refinement. Defaults to True.
    
    Returns:
        Dict[str, Any]: Summary dictionary containing:
            - 'files_analysed': Number of parameter files analysed
            - 'plots_created': Number of plots generated
            - 'output_dir': Path to output directory
            - 'results': List of dicts with details for each analysis
    
    Raises:
        ValueError: If input data is invalid or required columns are missing.
        FileNotFoundError: If specified files or directories don't exist.
    
    Output Files:
        Main output directory contains:
            - Effect size plots: dabest_{parameter}_{timewindow}.pdf
            - Summary statistics: dabestStats_TESTS_summary.csv
            - Colour legend: legend.png
            - Configuration info: info.txt
        
        If save_detailed_stats=True, 'detailed_stats/' subfolder contains:
            - Individual stats: stats_test_{parameter}_{timewindow}.csv
    
    Notes:
        - Input CSVs must contain columns: 'grp', 'fish', 'date', 'box' (CORE_ID_COLS)
          plus the specified time_windows columns
        - When using folder input with parameters='all', processes all CSVs with valid columns
        - When using folder input with parameters list, only processes CSVs where
          'parameter' column value matches the list
        - DABEST uses 95% confidence intervals by default
        - Bootstrap resampling is performed automatically by DABEST
    
    Example:
        >>> # Analyse all parameters in a folder
        >>> results = dabest_shared_control(
        ...     input_data='/path/to/data_folder',
        ...     time_windows=['avgDay', 'avgNight'],
        ...     control_group='wt',
        ...     treatment_groups=['drug1', 'drug2', 'drug3'],
        ...     output_dir='/path/to/output'  # <-- Now required in all examples
        ... )
        
        >>> # Analyse specific parameters with custom colours
        >>> results = dabest_shared_control(
        ...     input_data='/path/to/data_folder',
        ...     time_windows=['avgDay'],
        ...     control_group='wt',
        ...     treatment_groups=['mut1', 'mut2'],
        ...     output_dir='/path/to/output',
        ...     parameters=['sleepHours', 'activityPercentage'],
        ...     colour_palette={'wt': '#666666', 'mut1': '#ff6b6b', 'mut2': '#4ecdc4'}
        ... )
        
        >>> # Analyse a single file
        >>> results = dabest_shared_control(
        ...     input_data='/path/to/sleepHours.csv',
        ...     time_windows=['avgDay', 'avgNight'],
        ...     control_group='wt',
        ...     treatment_groups=['drug'],
        ...     output_dir='/path/to/output'
        ... )
    """
    
    print("=" * 70)
    print("DABEST SHARED CONTROL ANALYSIS")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: DISCOVER AND VALIDATE INPUT FILES
    # =========================================================================
    
    print("\n--- Step 1: Discovering input files ---")
    files_to_analyse = _discover_parameter_files(
        input_data=input_data,
        parameters=parameters,
        time_windows=time_windows
    )
    
    # =========================================================================
    # STEP 2: SETUP OUTPUT DIRECTORIES
    # =========================================================================
    
    # =========================================================================
    # STEP 2: SETUP OUTPUT DIRECTORIES
    # =========================================================================
    
    print("\n--- Step 2: Setting up output directories ---")
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create detailed stats directory if requested AND stats saving is enabled
    if save_stats and save_detailed_stats:
        stats_dir = os.path.join(output_dir, 'detailed_stats')
        os.makedirs(stats_dir, exist_ok=True)
        print(f"Detailed stats directory: {stats_dir}")
    elif save_stats:
        stats_dir = None
        print("Detailed stats will not be saved (save_detailed_stats=False)")
    else:
        stats_dir = None
        print("Statistics will not be saved (save_stats=False)")
    
    # =========================================================================
    # STEP 3: GENERATE COLOUR PALETTE
    # =========================================================================
    
    print("\n--- Step 3: Generating colour palette ---")
    colour_palette = _generate_colour_palette(
        control_group=control_group,
        treatment_groups=treatment_groups,
        custom_palette=colour_palette
    )
    
    # =========================================================================
    # STEP 4: SAVE ANALYSIS INFO AND LEGEND
    # =========================================================================
    
    print("\n--- Step 4: Saving analysis information ---")
    _save_analysis_info(
        output_dir=output_dir,
        control_group=control_group,
        treatment_groups=treatment_groups,
        colour_palette=colour_palette
    )
    
    _generate_legend_image(
        output_dir=output_dir,
        colour_palette=colour_palette
    )
    
    # =========================================================================
    # STEP 5: ANALYSE EACH PARAMETER FILE
    # =========================================================================
    
    print("\n--- Step 5: Running DABEST analyses ---")
    all_results = []
    
    for file_info in files_to_analyse:
        result = _analyse_single_parameter(
            file_info=file_info,
            time_windows=time_windows,
            control_group=control_group,
            treatment_groups=treatment_groups,
            colour_palette=colour_palette,
            output_dir=output_dir,
            stats_dir=stats_dir,
            plot_width=plot_width,
            plot_height=plot_height,
            raw_marker_size=raw_marker_size,
            contrast_marker_size=contrast_marker_size
        )
        all_results.extend(result)
    
    
    # =========================================================================
    # STEP 6: CREATE SUMMARY STATISTICS
    # =========================================================================
    
    if save_stats and all_results:  # Only if save_stats=True
        _create_summary_csvs(
            all_results=all_results,
            output_dir=output_dir,
            stats_dir=stats_dir
        )
    elif not save_stats:
        print("\n--- Skipping summary statistics (save_stats=False) ---")
    else:
        print("\nWarning: No analyses were completed. No summary created.")
    
    # =========================================================================
    # STEP 7: RETURN SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Files analysed: {len(files_to_analyse)}")
    print(f"Plots created: {len(all_results)}")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)
    
    return {
        'files_analysed': len(files_to_analyse),
        'plots_created': len(all_results),
        'output_dir': output_dir,
        'results': all_results
    }


def dabest_multi_group(
    input_data: Union[str, List[str], pd.DataFrame],
    time_windows: List[str],
    comparison_pairs: Dict[str, str],
    output_dir: str,
    parameters: Union[List[str], str] = 'all',
    colour_palette: Optional[Dict[str, str]] = None,
    plot_width: int = 5,
    plot_height: int = 6,
    raw_marker_size: int = 4,
    contrast_marker_size: int = 8,
    save_detailed_stats: bool = True,
    save_stats: bool = True
) -> Dict[str, Any]:
    """
    Perform DABEST multi-group analysis on behavioural data.
    
    Compares multiple treatment-control pairs where each treatment has its own
    dedicated control group. Uses bootstrap-coupled estimation statistics and
    generates effect size plots for each parameter and time window combination.
    
    This function is a wrapper around the DABEST package 
    (https://github.com/ACCLAB/DABESTPY) for convenient use with zebrafish 
    behavioural data.
    
    Args:
        input_data (Union[str, List[str], pd.DataFrame]): Input data, can be:
            - Path to a single CSV file
            - Path to a directory containing CSV files
            - List of paths to CSV files
            - pandas DataFrame
        time_windows (List[str]): Column names to analyse (e.g., ['avgDay', 'avgNight']).
            These columns must exist in the input data.
        comparison_pairs (Dict[str, str]): Treatment-control pairs to compare.
            Format: {treatment_group: control_group}
            Example: {'wt_drug': 'wt_baseline', 'mut_drug': 'mut_baseline'}
        output_dir (str): Directory where plots and statistics will be saved.
            Will be created if it doesn't exist.
        parameters (Union[List[str], str]): Parameters to analyse. Options:
            - 'all': Analyse all CSV files with required columns (default)
            - List[str]: Only analyse CSVs where 'parameter' column matches list
        colour_palette (Optional[Dict[str, str]]): Custom colour mapping for groups.
            Format: {group_name: hex_colour}. If None, generates default palette
            with alternating control/treatment colours.
        plot_width (int): Width of generated plots in inches. Defaults to 5.
        plot_height (int): Height of generated plots in inches. Defaults to 6.
        raw_marker_size (int): Size of individual data point markers. Defaults to 4.
        contrast_marker_size (int): Size of effect size markers. Defaults to 8.
        save_detailed_stats (bool): If True, saves detailed statistics for each
            parameter-timewindow combination to 'detailed_stats/' subfolder.
            If False, only saves summary statistics. Defaults to True.
        save_stats (bool): If True, saves all statistical outputs (detailed and/or summary).
            If False, only generates plots. Useful for iterative plot refinement. Defaults to True.
    
    Returns:
        Dict[str, Any]: Summary dictionary containing:
            - 'files_analysed': Number of parameter files analysed
            - 'plots_created': Number of plots generated
            - 'output_dir': Path to output directory
            - 'results': List of dicts with details for each analysis
    
    Raises:
        ValueError: If input data is invalid or required columns are missing.
        FileNotFoundError: If specified files or directories don't exist.
    
    Output Files:
        Main output directory contains:
            - Effect size plots: dabest_{parameter}_{timewindow}.pdf
            - Summary statistics: dabestStats_TESTS_summary.csv
            - Colour legend: legend.png
            - Configuration info: info.txt
        
        If save_detailed_stats=True, 'detailed_stats/' subfolder contains:
            - Individual stats: stats_test_{parameter}_{timewindow}.csv
    
    Notes:
        - Input CSVs must contain columns: 'grp', 'fish', 'date', 'box' (CORE_ID_COLS)
          plus the specified time_windows columns
        - Each treatment is compared only to its paired control (not shared)
        - DABEST uses 95% confidence intervals by default
        - Bootstrap resampling is performed automatically by DABEST
    
    Example:
        >>> # Compare drug treatments where each genotype has baseline
        >>> results = dabest_multi_group(
        ...     input_data='/path/to/data',
        ...     time_windows=['avgDay_normalised'],
        ...     comparison_pairs={
        ...         'wt_drug': 'wt_baseline',
        ...         'mut_drug': 'mut_baseline'
        ...     },
        ...     output_dir='/path/to/output'
        ... )
        
        >>> # With custom colours
        >>> results = dabest_multi_group(
        ...     input_data='/path/to/data',
        ...     time_windows=['avgDay'],
        ...     comparison_pairs={
        ...         'treated1': 'control1',
        ...         'treated2': 'control2'
        ...     },
        ...     output_dir='/path/to/output',
        ...     colour_palette={
        ...         'control1': '#cccccc',
        ...         'treated1': '#ff6b6b',
        ...         'control2': '#999999',
        ...         'treated2': '#4ecdc4'
        ...     }
        ... )
    """
    
    print("=" * 70)
    print("DABEST MULTI-GROUP ANALYSIS")
    print("=" * 70)
    
    # =========================================================================
    # STEP 1: DISCOVER AND VALIDATE INPUT FILES
    # =========================================================================
    
    print("\n--- Step 1: Discovering input files ---")
    files_to_analyse = _discover_parameter_files(
        input_data=input_data,
        parameters=parameters,
        time_windows=time_windows
    )
    
    # =========================================================================
    # STEP 2: SETUP OUTPUT DIRECTORIES
    # =========================================================================
    
    print("\n--- Step 2: Setting up output directories ---")
    
    # Create main output directory
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Create detailed stats directory if requested AND stats saving is enabled
    if save_stats and save_detailed_stats:
        stats_dir = os.path.join(output_dir, 'detailed_stats')
        os.makedirs(stats_dir, exist_ok=True)
        print(f"Detailed stats directory: {stats_dir}")
    elif save_stats:
        stats_dir = None
        print("Detailed stats will not be saved (save_detailed_stats=False)")
    else:
        stats_dir = None
        print("Statistics will not be saved (save_stats=False)")
    
    # =========================================================================
    # STEP 3: GENERATE COLOUR PALETTE
    # =========================================================================
    
    print("\n--- Step 3: Generating colour palette ---")
    colour_palette = _generate_multi_group_colour_palette(
        comparison_pairs=comparison_pairs,
        custom_palette=colour_palette
    )
    
    # =========================================================================
    # STEP 4: SAVE ANALYSIS INFO AND LEGEND
    # =========================================================================
    
    print("\n--- Step 4: Saving analysis information ---")
    _save_multi_group_analysis_info(
        output_dir=output_dir,
        comparison_pairs=comparison_pairs,
        colour_palette=colour_palette
    )
    
    _generate_legend_image(
        output_dir=output_dir,
        colour_palette=colour_palette
    )
    
    # =========================================================================
    # STEP 5: ANALYSE EACH PARAMETER FILE
    # =========================================================================
    
    print("\n--- Step 5: Running DABEST analyses ---")
    all_results = []
    
    for file_info in files_to_analyse:
        result = _analyse_single_parameter_multi_group(
            file_info=file_info,
            time_windows=time_windows,
            comparison_pairs=comparison_pairs,
            colour_palette=colour_palette,
            output_dir=output_dir,
            stats_dir=stats_dir,
            plot_width=plot_width,
            plot_height=plot_height,
            raw_marker_size=raw_marker_size,
            contrast_marker_size=contrast_marker_size
        )
        all_results.extend(result)
    
    # =========================================================================
    # STEP 6: CREATE SUMMARY STATISTICS
    # =========================================================================
    
    if save_stats and all_results:
        _create_summary_csvs(
            all_results=all_results,
            output_dir=output_dir,
            stats_dir=stats_dir
        )
    elif not save_stats:
        print("\n--- Skipping summary statistics (save_stats=False) ---")
    else:
        print("\nWarning: No analyses were completed. No summary created.")
    
    # =========================================================================
    # STEP 7: RETURN SUMMARY
    # =========================================================================
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Files analysed: {len(files_to_analyse)}")
    print(f"Plots created: {len(all_results)}")
    print(f"Results saved to: {output_dir}")
    print("=" * 70)
    
    return {
        'files_analysed': len(files_to_analyse),
        'plots_created': len(all_results),
        'output_dir': output_dir,
        'results': all_results
    }
