# tests/test_core.py

import pandas as pd
import os
import shutil
from zf_bhv_suite.core import avgDayNight

# Define the input and output directories for testing
TEST_INPUT_DIR = "tests/test_input"
TEST_OUTPUT_DIR = "tests/test_output"

# Define a function to create fake data for testing
def create_test_data(file_path):
    data = {
        'Z_Day1': [10, 20, 30],
        'Z_Night1': [5, 15, 25],
        'Z_Day2': [12, 22, 32],
        'Z_Night2': [7, 17, 27],
        'other_data': [1, 2, 3]
    }
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)

def test_avgDayNight_creation():
    """Tests if the function runs and creates new average columns."""

    # Setup: Create necessary directories and test data
    if os.path.exists(TEST_INPUT_DIR):
        shutil.rmtree(TEST_INPUT_DIR)
    if os.path.exists(TEST_OUTPUT_DIR):
        shutil.rmtree(TEST_OUTPUT_DIR)

    os.makedirs(TEST_INPUT_DIR)
    os.makedirs(TEST_OUTPUT_DIR)

    input_file = os.path.join(TEST_INPUT_DIR, "test_bhv.csv")
    create_test_data(input_file)

    # Run the function
    avgDayNight(input_dir=TEST_INPUT_DIR, output_dir=TEST_OUTPUT_DIR)

    # Assertion: Check if the output file was created
    output_file = os.path.join(TEST_OUTPUT_DIR, "test_bhv.csv")
    assert os.path.exists(output_file)

    # Assertion: Check if the calculated columns are correct
    df_out = pd.read_csv(output_file)
    assert 'avgDay' in df_out.columns
    assert 'avgNight' in df_out.columns

    # Assertion: Check a single calculated value for accuracy (average of day 10, 12 is 11)
    assert df_out['avgDay'][0] == 11.0

    # Cleanup: Remove test directories
    shutil.rmtree(TEST_INPUT_DIR)
    shutil.rmtree(TEST_OUTPUT_DIR)
