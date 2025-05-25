#!/usr/bin/env python3

import sys
import argparse
from typing import List, Optional  # Added for type hints if needed later

# Explicit imports from the ldrawpy package
from ldrawpy.ldrhelpers import clean_file
from ldrawpy.ldrpprint import pprint_line


def main() -> None:  # ADDED return type hint
    parser = argparse.ArgumentParser(
        description="Display the contents of a LDraw file.",
        formatter_class=argparse.RawTextHelpFormatter,
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

    lines_to_print: List[str] = []
    try:
        if args.clean:
            cleaned_lines_list = clean_file(args.filename, as_str=True)
            if cleaned_lines_list is None:
                print(f"Error: Could not clean file {args.filename}", file=sys.stderr)
                sys.exit(1)
            lines_to_print = cleaned_lines_list
        else:
            with open(args.filename, "r", encoding="utf-8") as f:
                lines_to_print = f.readlines()
    except FileNotFoundError:
        print(f"Error: File not found - {args.filename}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)
        sys.exit(1)

    for i, line_content in enumerate(lines_to_print):
        line_to_pass = line_content.rstrip("\r\n")
        line_num_to_display: Optional[int] = (i + 1) if args.lineno else None
        pprint_line(line_to_pass, lineno=line_num_to_display, nocolour=args.nocolour)


if __name__ == "__main__":
    main()
