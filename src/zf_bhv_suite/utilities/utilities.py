#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 14:59:11 2026

@author: talia
"""


### file_utils.py

def find_csv_files(
    directory: str,
    pattern: str = "*.csv"
) -> List[str]:
    """Find CSV files matching pattern."""

def create_output_directory(
    path: str,
    exist_ok: bool = True
) -> None:
    """Create output directory with error handling."""

def get_parameter_from_filename(
    filename: str
) -> str:
    """Extract parameter name from filename."""
    
    
### validation.py


def validate_columns_exist(
    df: pd.DataFrame,
    required_cols: List[str],
) -> None:
    """Check required columns exist, raise informative error if not."""

def validate_groups_exist(
    df: pd.DataFrame,
    required_groups: List[str],
    group_col: str = 'grp'
) -> None:
    """Check required groups exist in data."""

def validate_file_exists(
    filepath: str
) -> None:
    """Check file exists, raise error if not."""
    
    

### data_transforms.py

def long_to_wide(
    df: pd.DataFrame,
    id_cols: List[str],
    value_cols: List[str]
) -> pd.DataFrame:
    """Convert long format to wide format."""

def filter_to_groups(
    df: pd.DataFrame,
    groups: List[str],
    group_col: str = 'grp'
) -> pd.DataFrame:
    """Filter DataFrame to specific groups."""