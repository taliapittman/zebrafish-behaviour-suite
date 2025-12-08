import pandas as pd # Used to create and read data (DataFrames) for testing
import os           # Used for file path manipulation (joining directories/filenames)
import shutil       # Used for removing directories recursively (setup and cleanup)
import json         # Used to read and verify the JSON output
import random       # Used for generating random data
from typing import List, Dict # Required for type hinting List[str], Dict
# Import the functions we want to test from our main package code
from zf_bhv_suite.core import avgDayNight, normalise_data

# Set random seed for reproducibility in tests
random.seed(42)

# =================================================================
# 1. Configuration Constants
# =================================================================

# Define the input and output directories for testing (relative to the project root)
TEST_INPUT_DIR = "tests/test_input"    # Where we place the fake input CSVs
TEST_OUTPUT_DIR = "tests/test_output" # Where the core functions should save their results

# Define the input directory for the avgDayNight test (Crucial addition/restoration)
AVG_DAY_NIGHT_INPUT_DIR = os.path.join(TEST_INPUT_DIR, "avgDayNight_test_data")

# =================================================================
# 2. Setup Helper Function (Test Fixture Data)
# =================================================================

def create_avgDayNight_test_data_files(directory):
    """
    Generates two specific DataFrames (activityTotalPx and sleepHours)
    to test the avgDayNight function with realistic column names.

    Args:
        directory: The path where the CSV files should be saved.
    """
    os.makedirs(directory, exist_ok=True)

    # --- File 1: activityTotalPx_ExpA.csv ---
    data_px = {
        'parameter': ['activityTotalPx'] * 5,
        'date': [250110] * 5,
        'box': [10] * 5,
        'fish': ['f1', 'f2', 'f3', 'f4', 'f5'],
        'grp': ['wt', 'wt', 'wt', 'mutant', 'mutant'],
        'night0': [0.1611736, 0.139643, 0.0680831, 0.0311381, 0.1914448],
        'day1': [0.0673521, 0.0185945, 0.0193884, 0.1695065, 0.120765],
        'night1': [0.1614353, 0.1459599, 0.1072688, 0.1946245, 0.0757379],
        'day2': [0.1104305, 0.1658895, 0.123723, 0.1723483, 0.1154916],
    }
    df_px = pd.DataFrame(data_px)
    df_px.to_csv(os.path.join(directory, "activityTotalPx_ExpA.csv"), index=False)

    # --- File 2: sleepHours_ExpA.csv ---
    # Note: The core function must be able to handle 'night 1' (with space) and 'night1' (no space).
    data_sleep = {
        'parameter': ['sleepHours'] * 5,
        'date': [250110] * 5,
        'box': [10] * 5,
        'fish': ['f1', 'f2', 'f3', 'f4', 'f5'],
        'grp': ['wt', 'wt', 'wt', 'mutant', 'mutant'],
        'night 1': [6.394268, 0.2501076, 2.7502932, 2.2321074, 7.3647121], # Note the space
        'day1': [6.7669949, 8.9217957, 0.8693883, 4.2192182, 0.2979722],
        'night1': [2.1863797, 5.0535529, 0.2653597, 1.9883765, 6.4988444],
        'day2': [5.4494148, 2.2044062, 5.8926568, 8.0943046, 0.0649876],
    }
    df_sleep = pd.DataFrame(data_sleep)
    df_sleep.to_csv(os.path.join(directory, "sleepHours_ExpA.csv"), index=False)

def create_normalisation_test_data(experiment_dir: str,
                                   parameter_name: str,
                                   date: int,
                                   box: int,
                                   grp_data: List[str],
                                   data_range: tuple):
    """
    Generates a specific DataFrame for the normalise_data test based on user specification.
    """

    # Ensure the experiment directory exists
    os.makedirs(experiment_dir, exist_ok=True)

    min_val, max_val = data_range
    num_rows = len(grp_data)

    # Generate random data points for the two time windows (key normalization columns)
    night0_data = [round(random.uniform(min_val, max_val), 7) for _ in range(num_rows)]
    day1_data = [round(random.uniform(min_val, max_val), 7) for _ in range(num_rows)]

    # Generate random data points for the average columns (to simulate complete input data)
    avgDay_data = [round(random.uniform(min_val, max_val), 7) for _ in range(num_rows)]
    avgNight_data = [round(random.uniform(min_val, max_val), 7) for _ in range(num_rows)]

    # Data columns (Note: 'night0' and 'day1' are the new time windows)
    data = {
        'parameter': [parameter_name] * num_rows,
        'date': [date] * num_rows,
        'box': [box] * num_rows,
        'fish': [f'f{i+1}' for i in range(num_rows)], # f1, f2, f3, f4, f5
        'grp': grp_data,
        'night0': night0_data,
        'day1': day1_data,
        # Populating 'avgDay' and 'avgNight' with generated random data
        'avgDay': avgDay_data,
        'avgNight': avgNight_data,
    }
    df = pd.DataFrame(data)

    # Save the file with the specified naming convention
    filename = f"{parameter_name}_{os.path.basename(experiment_dir)}.csv"
    df.to_csv(os.path.join(experiment_dir, filename), index=False)

    # Return the DataFrame for verification/pre-calculation in the test function
    return df


def setup_test_environment(dirs_to_create: List[str]):
    """Cleans and creates required directories."""
    if os.path.exists(TEST_INPUT_DIR):
        shutil.rmtree(TEST_INPUT_DIR)
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)

    os.makedirs(TEST_INPUT_DIR, exist_ok=True)
    os.makedirs(TEST_OUTPUT_DIR, exist_ok=True)

    for d in dirs_to_create:
        os.makedirs(d, exist_ok=True)

def cleanup_test_environment():
    """Removes all test directories."""
    if os.path.exists(TEST_INPUT_DIR):
        shutil.rmtree(TEST_INPUT_DIR)
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)

# =================================================================
# 3. Unit Test Functions
# =================================================================

def test_avgDayNight_creation():
    """Tests if the avgDayNight function runs and creates new average columns for two files."""

    # --- SETUP ---
    setup_test_environment(dirs_to_create=[])
    # FIX: Use the correct function name and the dedicated input directory constant
    create_avgDayNight_test_data_files(AVG_DAY_NIGHT_INPUT_DIR)

    # --- EXECUTION ---
    avgDayNight(input_dir=AVG_DAY_NIGHT_INPUT_DIR, output_dir=TEST_OUTPUT_DIR)

    # --- ASSERTION for activityTotalPx_ExpA.csv ---
    output_file_px = os.path.join(TEST_OUTPUT_DIR, "activityTotalPx_ExpA.csv")
    assert os.path.exists(output_file_px), "The activityTotalPx output file was not created."

    df_px_out = pd.read_csv(output_file_px)
    assert 'avgDay' in df_px_out.columns, "The 'avgDay' column is missing in activityTotalPx."
    assert 'avgNight' in df_px_out.columns, "The 'avgNight' column is missing in activityTotalPx."

    # Check numerical accuracy for Row 0 (Fish f1) in activityTotalPx_ExpA.csv:
    # avgDay = (day1 + day2) / 2 = (0.0673521 + 0.1104305) / 2 = 0.0888913
    expected_avg_day_px = 0.0888913
    # avgNight = (night0 + night1) / 2 = (0.1611736 + 0.1614353) / 2 = 0.16130445
    expected_avg_night_px = 0.16130445

    # Use close assertion for floating point numbers
    assert abs(df_px_out['avgDay'][0] - expected_avg_day_px) < 1e-6, \
        f"activityTotalPx avgDay check failed. Expected {expected_avg_day_px}, got {df_px_out['avgDay'][0]}"
    assert abs(df_px_out['avgNight'][0] - expected_avg_night_px) < 1e-6, \
        f"activityTotalPx avgNight check failed. Expected {expected_avg_night_px}, got {df_px_out['avgNight'][0]}"


    # --- ASSERTION for sleepHours_ExpA.csv ---
    output_file_sleep = os.path.join(TEST_OUTPUT_DIR, "sleepHours_ExpA.csv")
    assert os.path.exists(output_file_sleep), "The sleepHours output file was not created."

    df_sleep_out = pd.read_csv(output_file_sleep)
    assert 'avgDay' in df_sleep_out.columns, "The 'avgDay' column is missing in sleepHours."
    assert 'avgNight' in df_sleep_out.columns, "The 'avgNight' column is missing in sleepHours."

    # Check numerical accuracy for Row 0 (Fish f1) in sleepHours_ExpA.csv:
    # avgDay = (day1 + day2) / 2 = (6.7669949 + 5.4494148) / 2 = 6.10820485
    expected_avg_day_sleep = 6.10820485
    # avgNight = (night 1 + night1) / 2 = (6.394268 + 2.1863797) / 2 = 4.29032385
    expected_avg_night_sleep = 4.29032385

    assert abs(df_sleep_out['avgDay'][0] - expected_avg_day_sleep) < 1e-6, \
        f"sleepHours avgDay check failed. Expected {expected_avg_day_sleep}, got {df_sleep_out['avgDay'][0]}"
    assert abs(df_sleep_out['avgNight'][0] - expected_avg_night_sleep) < 1e-6, \
        f"sleepHours avgNight check failed. Expected {expected_avg_night_sleep}, got {df_sleep_out['avgNight'][0]}"

    # --- CLEANUP (Re-enabled for this specific test) ---
    cleanup_test_environment()


def test_normalise_data_full_process():
    """
    Tests the normalise_data function for file creation, data merging,
    and calculation accuracy across two experiments with two parameters each.
    """
    # --- SETUP ---
    exp_a_dir = os.path.join(TEST_INPUT_DIR, 'ExpA')
    exp_b_dir = os.path.join(TEST_INPUT_DIR, 'ExpB')

    setup_test_environment(dirs_to_create=[exp_a_dir, exp_b_dir])

    # Define common parameters
    TIME_WINDOWS = ['night0', 'day1']
    PARAMETERS = ['sleepHours', 'activityTotalPx']
    BASELINE_GROUP = 'wt'
    NON_BASELINE_GROUP_A = 'mutant'

    # --- EXP A: 3 WT, 2 Mutant ---
    grp_a = [BASELINE_GROUP] * 3 + [NON_BASELINE_GROUP_A] * 2

    # 1. sleepHours_ExpA.csv (Range 0-10)
    df_sleep_a = create_normalisation_test_data(
        experiment_dir=exp_a_dir,
        parameter_name=PARAMETERS[0],
        date=250110,
        box=10,
        grp_data=grp_a,
        data_range=(0, 10)
    )

    # 2. activityTotalPx_ExpA.csv (Range 0.00005-0.2)
    df_activity_a = create_normalisation_test_data(
        experiment_dir=exp_a_dir,
        parameter_name=PARAMETERS[1],
        date=250110,
        box=10,
        grp_data=grp_a,
        data_range=(0.00005, 0.2)
    )

    # --- EXP B: 2 WT, 3 Mutant ---
    grp_b = [BASELINE_GROUP] * 2 + [NON_BASELINE_GROUP_A] * 3

    # 3. sleepHours_ExpB.csv (Range 0-10)
    df_sleep_b = create_normalisation_test_data(
        experiment_dir=exp_b_dir,
        parameter_name=PARAMETERS[0],
        date=250120,
        box=12,
        grp_data=grp_b,
        data_range=(0, 10)
    )

    # 4. activityTotalPx_ExpB.csv (Range 0.00005-0.2)
    df_activity_b = create_normalisation_test_data(
        experiment_dir=exp_b_dir,
        parameter_name=PARAMETERS[1],
        date=250120,
        box=12,
        grp_data=grp_b,
        data_range=(0.00005, 0.2)
    )

    # --- PRE-CALCULATE BASELINES FOR ASSERTION ---
    # ExpA Baselines (mean of first 3 rows, where grp == 'wt')
    sleep_a_baseline_night0 = df_sleep_a[df_sleep_a['grp'] == 'wt']['night0'].mean()
    sleep_a_baseline_day1 = df_sleep_a[df_sleep_a['grp'] == 'wt']['day1'].mean()

    activity_a_baseline_day1 = df_activity_a[df_activity_a['grp'] == 'wt']['day1'].mean()

    # ExpB Baselines (mean of first 2 rows, where grp == 'wt')
    sleep_b_baseline_night0 = df_sleep_b[df_sleep_b['grp'] == 'wt']['night0'].mean()
    activity_b_baseline_day1 = df_activity_b[df_activity_b['grp'] == 'wt']['day1'].mean()


    # Input parameters for normalise_data
    input_dirs = [exp_a_dir, exp_b_dir]
    export_master_path = os.path.join(TEST_OUTPUT_DIR, 'master_normalised.csv')

    # Baseline map: All groups are normalised against the 'wt' baseline group
    baseline_map = {
        'wt': 'wt',
        'mutant': 'wt',
    }

    # --- EXECUTION ---
    normalise_data(
        input_dirs=input_dirs,
        export_path=export_master_path,
        control_grouping=baseline_map,
        time_windows=TIME_WINDOWS,
        parameters=PARAMETERS
    )

    # --- ASSERTION ---

    # 1. Check for required file creation
    info_csv_path = os.path.join(TEST_OUTPUT_DIR, 'normalisation_info.csv')
    metadata_json_path = os.path.join(TEST_OUTPUT_DIR, 'normalisation_metadata.json')

    assert os.path.exists(export_master_path), "Master CSV was not created."
    assert os.path.exists(info_csv_path), "Normalisation info CSV (audit log) was not created."
    assert os.path.exists(metadata_json_path), "Normalisation metadata JSON was not created."

    # 2. Check content of Master CSV
    df_master = pd.read_csv(export_master_path)
    # Check total rows (5 in ExpA + 5 in ExpB = 10)
    assert len(df_master) == 10, f"Expected 10 rows in master CSV, got {len(df_master)}"

    # Check column names (5 ID + 4 normalised columns = 9)
    expected_cols = set(['date', 'box', 'fish', 'grp', 'experiment',
                         'sleepHours_night0_normalised', 'sleepHours_day1_normalised',
                         'activityTotalPx_night0_normalised', 'activityTotalPx_day1_normalised'])
    assert expected_cols.issubset(df_master.columns), "Missing normalised columns in master CSV."

    # --- VERIFY CALCULATIONS ---

    # A. ExpA: sleepHours (Mutant Fish 1, night0)
    # Raw value is the 4th row (index 3) of df_sleep_a
    raw_val_sleep_a = df_sleep_a['night0'].iloc[3]
    expected_val_sleep_a = raw_val_sleep_a - sleep_a_baseline_night0

    master_val_sleep_a = df_master[
        (df_master['experiment'] == 'experiment_1') & (df_master['grp'] == 'mutant')
    ].iloc[0]['sleepHours_night0_normalised']

    assert abs(master_val_sleep_a - expected_val_sleep_a) < 1e-6, \
        f"ExpA sleepHours night0 check failed. Expected {expected_val_sleep_a}, got {master_val_sleep_a}"

    # B. ExpB: activityTotalPx (Mutant Fish 2, day1)
    # Raw value is the 4th row (index 3) of df_activity_b
    raw_val_activity_b = df_activity_b['day1'].iloc[3]
    expected_val_activity_b = raw_val_activity_b - activity_b_baseline_day1

    master_val_activity_b = df_master[
        (df_master['experiment'] == 'experiment_2') & (df_master['grp'] == 'mutant')
    ].iloc[1]['activityTotalPx_day1_normalised'] # Second mutant row (index 1 of the filtered df)

    assert abs(master_val_activity_b - expected_val_activity_b) < 1e-6, \
        f"ExpB activityTotalPx day1 check failed. Expected {expected_val_activity_b}, got {master_val_activity_b}"

    # C. ExpA: Check a WT fish (should be close to zero, or exactly zero if the raw WT value was the mean)
    # Raw value is the 3rd row (index 2) of df_sleep_a (last WT fish)
    raw_val_sleep_a_wt = df_sleep_a['day1'].iloc[2]
    expected_val_sleep_a_wt = raw_val_sleep_a_wt - sleep_a_baseline_day1

    master_val_sleep_a_wt = df_master[
        (df_master['experiment'] == 'experiment_1') & (df_master['grp'] == 'wt')
    ].iloc[2]['sleepHours_day1_normalised']

    assert abs(master_val_sleep_a_wt - expected_val_sleep_a_wt) < 1e-6, \
        f"ExpA sleepHours WT day1 check failed. Expected {expected_val_sleep_a_wt}, got {master_val_sleep_a_wt}"

    # 3. Check content of Normalisation Info CSV (Audit Log)
    df_audit = pd.read_csv(info_csv_path)
    # 2 experiments * 2 parameters * 2 time windows = 8 audit entries
    assert len(df_audit) == 8, f"Expected 8 audit entries, got {len(df_audit)}"

    # Verify the ExpA sleepHours baseline mean recorded in the audit log
    expa_audit_mean_sleep = df_audit[
        (df_audit['experiment_id'] == 'experiment_1') &
        (df_audit['time_window'] == 'night0') &
        (df_audit['parameter'] == 'sleepHours')
    ]['calculated_mean'].iloc[0]

    assert abs(expa_audit_mean_sleep - sleep_a_baseline_night0) < 1e-6, \
        f"Audit log recorded incorrect mean for ExpA sleepHours: {expa_audit_mean_sleep}"

    # Verify the ExpB activityTotalPx baseline mean recorded in the audit log
    expb_audit_mean_activity = df_audit[
        (df_audit['experiment_id'] == 'experiment_2') &
        (df_audit['time_window'] == 'day1') &
        (df_audit['parameter'] == 'activityTotalPx')
    ]['calculated_mean'].iloc[0]

    assert abs(expb_audit_mean_activity - activity_b_baseline_day1) < 1e-6, \
        f"Audit log recorded incorrect mean for ExpB activityTotalPx: {expb_audit_mean_activity}"

    # 4. Check content of Metadata JSON (Archival Context)
    with open(metadata_json_path, 'r') as f:
        metadata = json.load(f)

    assert len(metadata) == 2, "Metadata should contain 2 experiment entries."
    assert metadata[0]['experiment_id'] == 'experiment_1'
    # Check baseline map structure is archived correctly
    assert metadata[0]['baseline_map']['mutant'] == 'wt'
    # Check calculated baselines structure is archived and contains both parameters
    assert 'sleepHours' in metadata[0]['calculated_baselines']
    assert 'activityTotalPx' in metadata[0]['calculated_baselines']

    # --- CLEANUP (Removed as requested by user) ---
    cleanup_test_environment()
