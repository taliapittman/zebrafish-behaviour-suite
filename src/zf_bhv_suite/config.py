# config.py
"""
Configuration constants for zf_bhv_suite package.

Contains shared constants, default values, and naming conventions
used across the package.
"""

# --- Core Column Identifiers ---
# These columns uniquely identify individual fish across experiments
CORE_ID_COLS = ['grp', 'fish', 'date', 'box']
ID_COLS = ['experiment'] + CORE_ID_COLS  # Full identifier including experiment used following normalisation

# --- File Patterns ---
# Standard file naming patterns
CSV_PATTERN = "*.csv"

# --- Column Name Conventions ---
# Standard column names created by preprocessing functions
AVG_DAY_COL = 'avgDay'
AVG_NIGHT_COL = 'avgNight'


# --- Normalisation Constants ---
# Default time windows for normalisation
DEFAULT_TIME_WINDOWS = ['avgDay', 'avgNight']

# File naming for normalisation outputs
NORMALISED_SUFFIX = '_normalised'
INFO_SUBDIR = 'normalisation_info'
AUDIT_FILENAME = 'normalisation_audit.csv'
METADATA_FILENAME = 'normalisation_metadata.json'
MASTER_SUMMARY_FILENAME = 'master_normalised_summary.csv'


