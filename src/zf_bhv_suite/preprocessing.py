#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Preprocessing functions for zebrafish behavioural data.

This module contains functions for preparing FramebyFrame outputs
for downstream normalisation and visualisation.
"""

import pandas as pd
import os
import glob
import sys
from typing import Optional, List, Dict, Any, Union

from .config import AVG_DAY_COL, AVG_NIGHT_COL


def avgDayNight(
    input_data: Union[str, List[str]],
    output_dir: Optional[str] = None
) -> None:
    """
    Calculate the average of day and night columns for CSV files.
    
    Scans CSV file(s), identifies columns containing 'day' or 'night' in their 
    names (case-insensitive), and adds new columns 'avgDay' and 'avgNight' 
    containing the mean values across all identified day/night columns respectively. 
    The column named 'day' itself is excluded from averaging to avoid confusion 
    with day number indicators.
    
    This function is typically used to summarise multi-day experiments (e.g., day1, day2) 
    into single representative values before normalisation.
    
    Args:
        input_data (Union[str, List[str]]): Can be:
            - Path to a single CSV file
            - Path to a directory containing CSV files
            - List of paths to CSV files
        output_dir (Optional[str]): Path to directory for saving processed files. 
            If None, files are saved back to their original location (overwriting). 
            For single file or list of files: saves to same directory as input.
            For directory input: saves to that directory.
            Directory will be created if it doesn't exist. Defaults to None.
    
    Returns:
        None: Processed files are saved directly to disk with the same filenames.
    
    Notes:
        - Files without 'day' or 'night' columns are skipped with a warning
        - If only day OR night columns exist, only that average is calculated
        - Original columns are preserved in the output
        - Errors in individual files don't stop processing of other files
        - For windows of interest (WOI) averaging, see avgWOI() [future implementation]
    
    Example:
        >>> # Process all CSVs in a folder
        >>> avgDayNight('/path/to/fbf_outputs')
        
        >>> # Process a single file
        >>> avgDayNight('/path/to/sleepHours.csv', '/path/to/output')
        
        >>> # Process specific files
        >>> avgDayNight(['/path/to/file1.csv', '/path/to/file2.csv'])
    """
    
    # Determine input type and build list of files to process
    files_to_process = []
    
    if isinstance(input_data, str):
        if os.path.isdir(input_data):
            # Input is a directory
            search_path = os.path.join(input_data, "*.csv")
            files_to_process = glob.glob(search_path)
            
            # Set default output_dir to input directory if not specified
            if output_dir is None:
                output_dir = input_data
                
        elif os.path.isfile(input_data):
            # Input is a single file
            files_to_process = [input_data]
            
            # Set default output_dir to same directory as input file
            if output_dir is None:
                output_dir = os.path.dirname(input_data)
                
        else:
            sys.exit(f"Error: The input path '{input_data}' does not exist.")
    
    elif isinstance(input_data, list):
        # Input is a list of file paths
        files_to_process = input_data
        
        # Set default output_dir to directory of first file
        if output_dir is None and files_to_process:
            output_dir = os.path.dirname(files_to_process[0])
    
    else:
        sys.exit(f"Error: input_data must be a string (file/directory path) or list of file paths.")
    
    # Validate we have files to process
    if not files_to_process:
        print(f"Warning: No CSV files found to process.")
        return
    
    # Validate input files exist
    for file_path in files_to_process:
        if not os.path.isfile(file_path):
            print(f"Warning: File not found: {file_path}. Skipping.")
            files_to_process.remove(file_path)
    
    if not files_to_process:
        print(f"Warning: No valid CSV files found to process.")
        return
    
    # Create output directory if needed
    if not os.path.exists(output_dir):
        print(f"Creating output directory: {output_dir}")
        os.makedirs(output_dir)
    
    print(f"Processing {len(files_to_process)} file(s)...")
    
    # Process each file
    for file_path in files_to_process:
        file_name = os.path.basename(file_path)
        print(f"  - Processing {file_name}")
        
        try:
            df = pd.read_csv(file_path)
            
            # Identify day and night columns (exclude 'day' column itself)
            day_cols = [
                col for col in df.columns 
                if 'day' in col.lower() and col.lower() != 'day'
            ]
            night_cols = [
                col for col in df.columns 
                if 'night' in col.lower()
            ]
            
            # Skip if no relevant columns found
            if not day_cols and not night_cols:
                print(
                    f"    Skipping {file_name}: No 'day' or 'night' columns "
                    f"found for averaging."
                )
                continue
            
            # Calculate averages
            if day_cols:
                df[AVG_DAY_COL] = df[day_cols].mean(axis=1)
            
            if night_cols:
                df[AVG_NIGHT_COL] = df[night_cols].mean(axis=1)
            
            # Save processed file
            output_file_path = os.path.join(output_dir, file_name)
            df.to_csv(output_file_path, index=False)
            
        except Exception as e:
            print(f"    An error occurred while processing {file_name}: {e}")
            continue
    
    print("\nProcessing complete.")


# Commented out future functions
# def validate_fbf_output(
#     input_dir: str,
#     expected_parameters: Optional[List[str]] = None
# ) -> Dict[str, Any]:
#     """Validate FramebyFrame output structure."""
#     pass

# def merge_parameters(
#     input_dir: str,
#     parameters: List[str],
#     output_file: str
# ) -> None:
#     """Merge multiple parameter files into one wide-format DataFrame."""
#     pass