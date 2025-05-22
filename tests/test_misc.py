# tests/test_misc.py

import os

# import sys # sys is not used, can be removed
import pytest
from pathlib import Path  # For robust path handling

# from toolbox import * # TODO: Refactor this wildcard import later
from ldrawpy import clean_file  # Import specific function needed

# Get the directory of the current test file (tests/)
TEST_FILE_DIR = Path(__file__).parent

# Construct the path to the input test file relative to this test script's directory
# tests/test_files/testfile2.ldr
FIN_PATH = TEST_FILE_DIR / "test_files" / "testfile2.ldr"


def test_cleanup():
    # Convert Path objects to strings, as clean_file likely expects string paths
    fin_str = str(FIN_PATH)
    fno_str = fin_str.replace(".ldr", "_clean.ldr")

    # Ensure the input file actually exists before trying to clean it
    assert FIN_PATH.is_file(), f"Test input file not found: {FIN_PATH}"

    clean_file(fin_str, fno_str)

    # Assertions for the input file (after ensuring it was found and read)
    with open(fin_str, "r") as f:
        fl_in = f.read()
        # These assertions might need to be verified if the file was not correctly read before.
        # For now, we keep them as they were in the original test.
        assert len(fl_in) == 1284, "Input file length mismatch"
        assert "-59.999975" in fl_in, "Expected content not in input file"

    # Assertions for the output file
    assert Path(fno_str).is_file(), f"Cleaned output file not found: {fno_str}"
    with open(fno_str, "r") as f:
        fl_out = f.read()
        # These assertions might need to be verified.
        assert len(fl_out) == 1101, "Cleaned file length mismatch"
        assert "-60" in fl_out, "Expected content not in cleaned file"

    # Optional: Clean up the generated _clean.ldr file after the test
    if Path(fno_str).exists():
        os.remove(fno_str)
