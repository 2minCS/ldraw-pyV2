#!/usr/bin/env python3

import sys
import argparse
from pathlib import Path  # ADDED
from typing import List, Optional

# Explicit imports from the ldrawpy package
# Assuming ldrawpy is installed or in PYTHONPATH
from ldrawpy.ldrhelpers import clean_file
from ldrawpy.ldrpprint import pprint_line


def main() -> None:
    """
    Main function for the ldrcat script.
    Parses command-line arguments and displays/cleans an LDraw file.
    """
    parser = argparse.ArgumentParser(
        description="Display the contents of a LDraw file.",
        formatter_class=argparse.RawTextHelpFormatter,  # Preserves formatting in help text
    )
    parser.add_argument(
        "filename",
        metavar="filename",
        type=str,
        nargs="?",  # Optional positional argument
        help="LDraw filename to process",
    )
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",  # Sets args.clean to True if present
        default=False,
        help="Clean up coordinate values in the output. This reads the file, cleans the content "
        "in memory, and prints the cleaned version. It does not modify the original file unless "
        "the output of this script is redirected to the original file.",
    )
    parser.add_argument(
        "-n",
        "--nocolour",
        action="store_true",
        default=False,
        help="Do not show file with colour syntax highlighting (requires 'rich' package).",
    )
    parser.add_argument(
        "-l", "--lineno", action="store_true", default=False, help="Show line numbers."
    )
    args = parser.parse_args()

    if args.filename is None:
        parser.print_help()
        sys.exit(1)

    # Convert the filename string from argparse to a Path object
    # .expanduser() handles paths like ~/file.ldr
    input_filepath = Path(args.filename).expanduser()

    lines_to_print: List[str] = []
    try:
        # Resolve the path to an absolute path.
        # strict=True will raise FileNotFoundError if the path doesn't exist.
        # This is a good check before attempting to open or clean.
        resolved_input_filepath = input_filepath.resolve(strict=True)

        if args.clean:
            # clean_file has been refactored to accept Path objects.
            # It returns a list of strings when as_str=True.
            cleaned_lines_list = clean_file(resolved_input_filepath, as_str=True)

            if (
                cleaned_lines_list is None
            ):  # clean_file returns None on error when not as_str,
                # but should return list or raise error if as_str.
                # Adding a check just in case.
                print(
                    f"Error: Could not clean file {str(resolved_input_filepath)}",
                    file=sys.stderr,
                )
                sys.exit(1)
            lines_to_print = cleaned_lines_list
        else:
            # open() works directly with Path objects
            with open(resolved_input_filepath, "r", encoding="utf-8") as f_handle:
                lines_to_print = f_handle.readlines()

    except FileNotFoundError:
        # This handles the case where input_filepath.resolve(strict=True) fails
        # or if the file is deleted between resolve and open (less likely).
        print(
            f"Error: File not found - {str(input_filepath)}", file=sys.stderr
        )  # Print user-provided path
        sys.exit(1)
    except Exception as e:
        # Catch other potential errors (e.g., permission issues during open/read)
        print(
            f"An error occurred while processing '{str(input_filepath)}': {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Print the (potentially cleaned) lines
    for i, line_content_str in enumerate(lines_to_print):
        line_to_pass_to_pprint = line_content_str.rstrip(
            "\r\n"
        )  # Remove original newlines

        line_num_to_display: Optional[int] = None
        if args.lineno:
            line_num_to_display = i + 1  # Line numbers are 1-indexed

        pprint_line(
            line_to_pass_to_pprint, lineno=line_num_to_display, nocolour=args.nocolour
        )


if __name__ == "__main__":
    main()
