#!/usr/bin/env python3

# import math # Not used
# import os.path # Not used
import sys
import argparse

# Explicit imports from the ldrawpy package
from ldrawpy.ldrhelpers import clean_file
from ldrawpy.ldrpprint import pprint_line

# If any constants were used, they would be imported from ldrawpy.constants


def main():
    parser = argparse.ArgumentParser(
        description="Display the contents of a LDraw file.",
        formatter_class=argparse.RawTextHelpFormatter,  # For better help text formatting
    )
    parser.add_argument(
        "filename", metavar="filename", type=str, nargs="?", help="LDraw filename"
    )
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        default=False,
        help="Clean up coordinate values in the output (does not modify the file).",
    )
    parser.add_argument(
        "-n",
        "--nocolour",  # Changed from --nocolor for consistency with British spelling in package
        action="store_true",
        default=False,
        help="Do not show file with colour syntax highlighting (requires 'rich' package).",
    )
    parser.add_argument(
        "-l",
        "--lineno",
        action="store_true",
        default=False,
        help="Show line numbers.",
    )
    args = parser.parse_args()
    # argsd = vars(args) # Not strictly needed if accessing args directly

    if args.filename is None:
        parser.print_help()
        sys.exit(1)  # Exit with error code if no filename

    lines_to_print: list[str] = []
    try:
        if args.clean:
            # clean_file with as_str=True returns a list of cleaned lines
            # It expects a filename as input.
            cleaned_lines_list = clean_file(args.filename, as_str=True)
            if cleaned_lines_list is None:  # clean_file might return None on error
                print(f"Error: Could not clean file {args.filename}", file=sys.stderr)
                sys.exit(1)
            lines_to_print = cleaned_lines_list
        else:
            with open(args.filename, "r", encoding="utf-8") as f:  # Added encoding
                lines_to_print = f.readlines()  # readlines() includes newlines
    except FileNotFoundError:
        print(f"Error: File not found - {args.filename}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    for i, line_content in enumerate(lines_to_print):
        # pprint_line expects a single line string, typically without trailing newline
        # if it handles its own printing. If it returns a string, then print() is needed.
        # Assuming pprint_line prints directly.
        line_to_pass = line_content.rstrip("\r\n")  # Remove EOL for pprint_line

        line_num_to_display = (i + 1) if args.lineno else None

        # pprint_line uses rich for coloring.
        # The --nocolour flag should be passed to it.
        pprint_line(line_to_pass, lineno=line_num_to_display, nocolour=args.nocolour)


if __name__ == "__main__":
    main()
