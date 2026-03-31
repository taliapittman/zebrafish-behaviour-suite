# Zebrafish Behaviour Suite

A Python package for processing, normalising, and analysing zebrafish larval behavioural data from FramebyFrame outputs.

## Overview

`zf_bhv_suite` provides a streamlined pipeline for zebrafish behavioural analysis:

1. **Preprocessing** - Average day/night data from multi-day experiments
2. **Normalisation** - Normalise parameters to control groups
3. **Effect Size Analysis** - DABEST bootstrap estimation with visualisation

Built specifically for outputs from François Kroll's [FramebyFrame](https://github.com/francoiskroll/FramebyFrame) R package.

## Features

- **Flexible input** - Works with single files, folders, or lists of files
- **Publication-ready plots** - PDF outputs with customisable aesthetics
- **Boostrapped statistics** - DABEST bootstrap-coupled estimation
- **Customisable colours** - Default palettes or custom colour schemes
- **Comprehensive outputs** - Plots, statistics, and audit trails

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Install from GitHub
```bash
pip install git+https://github.com/taliapittman/zebrafish-behaviour-suite.git
```

### Install in development mode (for contributing)
```bash
git clone https://github.com/taliapittman/zebrafish-behaviour-suite.git
cd zebrafish-behaviour-suite
pip install -e .
```

## Quick Start

### Basic workflow: Raw data → Effect size plots
```python
from zf_bhv_suite import avgDayNight, normalise_data, dabest_shared_control

# Step 1: Average day/night data
avgDayNight(
    input_data='/path/to/framebyframe_output',
    output_dir='/path/to/averaged_data'
)

# Step 2: Normalise to control
normalise_data(
    input_dirs=['/path/to/averaged_data'],
    export_path='/path/to/normalised_data',
    control_grouping='wt',
    time_windows=['avgDay', 'avgNight']
)

# Step 3: Effect size analysis
dabest_shared_control(
    input_data='/path/to/normalised_data',
    time_windows=['avgDay_normalised', 'avgNight_normalised'],
    control_group='wt',
    treatment_groups=['drug1', 'drug2'],
    output_dir='/path/to/results'
)
```

## Documentation

### 1. Preprocessing with `avgDayNight()`

Calculate average of day and night columns across multiple days.

**Input:** FramebyFrame parameter files with columns like `day1`, `day2`, `night1`, `night2`

**Output:** Same files with added `avgDay` and `avgNight` columns
```python
from zf_bhv_suite import avgDayNight

# Process all CSV files in a folder
avgDayNight(
    input_data='/path/to/framebyframe_output',
    output_dir='/path/to/output'
)

# Process a single file
avgDayNight(
    input_data='/path/to/sleepHours.csv',
    output_dir='/path/to/output'
)

# Process specific files
avgDayNight(
    input_data=[
        '/path/to/sleepHours.csv',
        '/path/to/activityPercentage.csv'
    ],
    output_dir='/path/to/output'
)
```

**Parameters:**
- `input_data`: Path to file, folder, or list of file paths
- `output_dir`: Where to save processed files (optional, defaults to input location)

---

### 2. Normalisation with `normalise_data()`

Normalise behavioural parameters to control group baseline. Ideal for combining multiple experiments.

**Input:** Processed CSV files with `avgDay` and `avgNight` columns

**Output:** Normalised CSV files with `_normalised` suffix, plus audit files
```python
from zf_bhv_suite import normalise_data

# Single baseline normalisation
normalise_data(
    input_dirs=['/path/to/experiment1', '/path/to/experiment2'],
    export_path='/path/to/normalised_output',
    control_grouping='wt',
    time_windows=['avgDay', 'avgNight'],
    parameters='all'  # or ['sleepHours', 'activityPercentage']
)

# Multi-baseline normalisation (different controls per treatment)
normalise_data(
    input_dirs=['/path/to/data'],
    export_path='/path/to/output',
    control_grouping={
        'drug1': 'wt',
        'drug2': 'vehicle'
    },
    time_windows=['avgDay', 'avgNight']
)
```

**Parameters:**
- `input_dirs`: List of experiment directories
- `export_path`: Output directory
- `control_grouping`: Control group name (str) or mapping (dict)
- `time_windows`: Columns to normalise (e.g., `['avgDay', 'avgNight']`)
- `parameters`: `'all'` or list of specific parameters
- `experiment_names`: Optional custom names for experiments

**Outputs:**
- `{parameter}_normalised.csv` - Normalised data for each parameter
- `normalisation_info/` - Audit logs and metadata

---

### 3. Effect Size Analysis with `dabest_shared_control()`

Compare multiple treatment groups to a shared control using DABEST bootstrap estimation.

**Input:** Normalised CSV files or raw FramebyFrame outputs

**Output:** PDF plots and statistical summaries
```python
from zf_bhv_suite import dabest_shared_control

# Basic analysis
results = dabest_shared_control(
    input_data='/path/to/normalised_data',
    time_windows=['avgDay_normalised', 'avgNight_normalised'],
    control_group='wt',
    treatment_groups=['drug1', 'drug2', 'drug3'],
    output_dir='/path/to/results'
)

# With custom aesthetics
results = dabest_shared_control(
    input_data='/path/to/data',
    time_windows=['avgDay'],
    control_group='wt',
    treatment_groups=['mut1', 'mut2'],
    output_dir='/path/to/results',
    colour_palette={
        'wt': '#666666',
        'mut1': '#ff6b6b',
        'mut2': '#4ecdc4'
    },
    plot_width=6,
    plot_height=7,
    raw_marker_size=5,
    contrast_marker_size=10
)

# Analyse specific parameters only
results = dabest_shared_control(
    input_data='/path/to/folder',
    time_windows=['avgDay'],
    control_group='wt',
    treatment_groups=['treatment1', 'treatment2'],
    output_dir='/path/to/results',
    parameters=['sleepHours', 'activityPercentage']
)
```

**Parameters:**
- `input_data`: File path, folder path, list of paths, or DataFrame
- `time_windows`: Columns to analyse (e.g., `['avgDay_normalised']`)
- `control_group`: Name of control group
- `treatment_groups`: List of treatment group names
- `output_dir`: Output directory (required)
- `parameters`: `'all'` or list of specific parameters (default: `'all'`)
- `colour_palette`: Custom colours (optional, defaults to grey control + colourful treatments)
- `plot_width`, `plot_height`: Plot dimensions in inches (default: 5, 6)
- `raw_marker_size`: Size of data point markers (default: 4)
- `contrast_marker_size`: Size of effect size markers (default: 8)
- `save_detailed_stats`: Save individual stats files (default: `True`)
- `save_stats`: Save any statistics (default: `True`, set to `False` for plot-only refinement)

**Outputs:**
- `dabest_{parameter}_{timewindow}.pdf` - Effect size plots
- `dabestStats_TESTS_summary.csv` - Consolidated statistics
- `legend.png` - Colour reference
- `info.txt` - Analysis configuration
- `detailed_stats/` - Individual statistical tests (if `save_detailed_stats=True`)

---

## Complete Example Workflow
```python
from zf_bhv_suite import avgDayNight, normalise_data, dabest_shared_control

# Define paths
raw_data = '/path/to/framebyframe/bhvparams_250225_16'
output_base = '/path/to/analysis_output'

# Step 1: Preprocess - average across days
avgDayNight(
    input_data=raw_data,
    output_dir=f'{output_base}/01_averaged'
)

# Step 2: Normalise to control
normalise_data(
    input_dirs=[f'{output_base}/01_averaged'],
    export_path=f'{output_base}/02_normalised',
    control_grouping='wt',
    time_windows=['avgDay', 'avgNight'],
    parameters='all',
    experiment_names=['exp_250225_16']
)

# Step 3: Effect size analysis
results = dabest_shared_control(
    input_data=f'{output_base}/02_normalised',
    time_windows=['avgDay_normalised', 'avgNight_normalised'],
    control_group='wt',
    treatment_groups=['drug_low', 'drug_medium', 'drug_high'],
    output_dir=f'{output_base}/03_dabest',
    parameters='all'
)

print(f"Analysis complete! Created {results['plots_created']} plots.")
```

---

## Tips & Best Practices

### File Organization
```
your_experiment/
├── framebyframe_output/     # Raw FramebyFrame outputs
├── averaged/                # After avgDayNight()
├── normalised/              # After normalise_data()
└── results/                 # After dabest_shared_control()
```

### Iterative Plot Refinement
```python
# First pass: Check plots without saving stats
dabest_shared_control(
    input_data='/path/to/data',
    time_windows=['avgDay'],
    control_group='wt',
    treatment_groups=['drug1'],
    output_dir='/path/to/test',
    save_stats=False,  # Skip stats
    raw_marker_size=4
)

# Adjust and re-run
dabest_shared_control(
    input_data='/path/to/data',
    time_windows=['avgDay'],
    control_group='wt',
    treatment_groups=['drug1'],
    output_dir='/path/to/test',
    save_stats=False,
    raw_marker_size=6,  # Bigger markers
    colour_palette={'wt': '#888888', 'drug1': '#e74c3c'}
)

# Final version with stats
dabest_shared_control(
    input_data='/path/to/data',
    time_windows=['avgDay'],
    control_group='wt',
    treatment_groups=['drug1'],
    output_dir='/path/to/final',
    save_stats=True,  # Now save everything
    raw_marker_size=6,
    colour_palette={'wt': '#888888', 'drug1': '#e74c3c'}
)
```

### Group Naming

- Group names must exactly match the `grp` column in your data
- Names are case-sensitive: `'wt'` ≠ `'WT'`
- Check your data: `pd.read_csv('file.csv')['grp'].unique()`

---

## Requirements

- Python ≥ 3.8
- pandas ≥ 1.3.0
- numpy ≥ 1.20.0
- dabest ≥ 2023.2.14
- matplotlib ≥ 3.3.0
- Pillow ≥ 8.0.0

---

## Referneces/Citation

**For the DABEST methodology:**
> Ho, J., Tumkaya, T., Aryal, S., Choi, H., & Claridge-Chang, A. (2019). Moving beyond P values: data analysis with estimation graphics. *Nature Methods*, 16(7), 565-566. https://doi.org/10.1038/s41592-019-0470-3

**For FramebyFrame (input data format):**
> Kroll, F., et al. (2021). A simple and effective F0 knockout method for rapid screening of behaviour and other complex phenotypes. *eLife*, 10, e59683. https://doi.org/10.7554/eLife.59683

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Acknowledgments

- **DABEST-Python** - Bootstrap estimation statistics ([GitHub](https://github.com/ACCLAB/DABESTPY))
- **FramebyFrame** - Zebrafish behavioural tracking ([GitHub](https://github.com/francoiskroll/FramebyFrame))

---

## Version History

### v0.1.0 (2026-01-26)
- Initial release
- `avgDayNight()` - Day/night averaging
- `normalise_data()` - Baseline normalisation
- `dabest_shared_control()` - Effect size analysis with shared control
