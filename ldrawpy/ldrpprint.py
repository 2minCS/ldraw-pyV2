#! /usr/bin/env python3
#
# Copyright (C) 2020  Michael Gale
# This file is part of the legocad python module.
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge,
# publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
"""
ldrpprint.py: LDraw pretty printer helper functions for console output.

This module provides functions to format LDraw lines with syntax highlighting
using the 'rich' library for console output. It helps in visualizing LDraw
data directly in the terminal with colours and styles.
"""

import string
from typing import List, Optional

from rich import print as rich_print  # type: ignore # For styled console output

# Assuming ldrawpy is installed or in PYTHONPATH for relative imports
from .ldrcolour import LDRColour
from .constants import LDRAW_TOKENS, META_TOKENS


def is_hex_colour(text: str) -> bool:
    """
    Checks if a given string represents a valid hex colour code (e.g., "#RRGGBB").
    Optionally strips surrounding quotes from the text before checking.

    Args:
        text: The string to check.

    Returns:
        True if the text is a valid 7-character hex colour string, False otherwise.
    """
    text_cleaned = text.strip('"')  # Remove potential quotes
    if not len(text_cleaned) == 7:  # Must be 7 characters (e.g., #RRGGBB)
        return False
    if not text_cleaned.startswith("#"):  # Must start with '#'
        return False
    # All characters after '#' must be hexadecimal digits
    return all(c in string.hexdigits for c in text_cleaned.lstrip("#"))


def pprint_ldr_colour(code_str: str) -> str:
    """
    Formats an LDraw colour code string with Rich library style tags for console output.
    Known LDraw colours are displayed with their actual colour as background (reversed).
    Special colours like 16 (default) and 24 (edge) get specific styles.

    Args:
        code_str: The LDraw colour code as a string.

    Returns:
        A Rich-formatted string for the colour code.
    """
    # Special styling for default (16) and edge/outline (24) colours
    if code_str == "16" or code_str == "24":
        return f"[bold navajo_white1]{code_str}[/bold navajo_white1]"
    if code_str == "0":  # Black
        return f"[bold default on white]{code_str}[/default on white]"  # Ensure black is visible on dark terms

    try:
        # Attempt to parse the code string as an integer LDraw colour code
        colour_code_int = int(code_str)
        colour_obj = LDRColour(colour_code_int)
        # Use the colour's hex value for Rich styling (reversed for visibility)
        return f"[#{colour_obj.as_hex()} reverse]{code_str}[/]"
    except ValueError:  # If code_str is not a valid integer
        return f"[white]{code_str}[/white]"  # Default to white text


def pprint_coord_str(v_coords_str_list: List[str], colour_tag: str = "[white]") -> str:
    """
    Formats a list of three coordinate strings (x, y, z) with a Rich colour tag.

    Args:
        v_coords_str_list: A list of three strings representing x, y, z coordinates.
        colour_tag: The Rich style tag to apply (e.g., "[white]", "[#91E3FF]").

    Returns:
        A Rich-formatted string for the coordinates, or an empty string if
        input is not a list of three strings.
    """
    if len(v_coords_str_list) == 3:
        # Join coordinates with spaces, apply colour tag, and make not bold
        return f"[not bold]{colour_tag}{v_coords_str_list[0]} {v_coords_str_list[1]} {v_coords_str_list[2]}[/not bold]"
    return ""  # Return empty if not 3 coordinates


def pprint_line1(line_str: str) -> str:
    """
    Pretty-prints an LDraw type 1 line (part/submodel reference) using Rich styling.
    Formats line type, colour code, coordinates, matrix, and filename with different styles.

    Args:
        line_str: The raw LDraw type 1 line string.

    Returns:
        A Rich-formatted string for the type 1 line.
    """
    s_parts_formatted: List[str] = []
    line_str_cleaned = line_str.rstrip()  # Remove trailing newline/whitespace
    tokens = line_str_cleaned.split()

    if len(tokens) < 14:  # Minimum tokens for a valid type 1 line
        return f"[white]{line_str_cleaned}[/white]"  # Default formatting if malformed

    # Line type (1)
    s_parts_formatted.append(f"[bold white]{tokens[0]}[/bold white]")
    # Colour code
    s_parts_formatted.append(pprint_ldr_colour(tokens[1]))
    # Location coordinates (x, y, z)
    s_parts_formatted.append(pprint_coord_str(tokens[2:5]))
    # Transformation matrix (3x3, row-major) - styled in groups of 3
    s_parts_formatted.append(
        pprint_coord_str(tokens[5:8], colour_tag="[#91E3FF]")
    )  # Matrix row 1
    s_parts_formatted.append(
        pprint_coord_str(tokens[8:11], colour_tag="[#FFF3AF]")
    )  # Matrix row 2
    s_parts_formatted.append(
        pprint_coord_str(tokens[11:14], colour_tag="[#91E3FF]")
    )  # Matrix row 3

    # Filename (can contain spaces)
    filename_part_str = " ".join(tokens[14:])
    # Style filename based on extension (.ldr for submodels, .dat for parts)
    file_colour_tag = (
        "#B7E67A" if filename_part_str.lower().endswith(".ldr") else "#F27759"
    )
    s_parts_formatted.append(
        f"[bold {file_colour_tag}]{filename_part_str}[/bold {file_colour_tag}]"
    )

    return " ".join(s_parts_formatted)


def pprint_line2345(line_str: str) -> str:
    """
    Pretty-prints LDraw type 2 (line), 3 (triangle), 4 (quadrilateral),
    or 5 (optional line) using Rich styling.
    Formats line type, colour code, and coordinate sets.

    Args:
        line_str: The raw LDraw type 2, 3, 4, or 5 line string.

    Returns:
        A Rich-formatted string for the line.
    """
    s_parts_formatted: List[str] = []
    line_str_cleaned = line_str.rstrip()
    tokens = line_str_cleaned.split()

    if len(tokens) < 2:  # Must have at least line type and colour
        return f"[white]{line_str_cleaned}[/white]"

    # Line type (2, 3, 4, or 5)
    s_parts_formatted.append(f"[bold white]{tokens[0]}[/bold white]")
    # Colour code
    s_parts_formatted.append(pprint_ldr_colour(tokens[1]))

    coord_start_index = 2
    try:
        num_points_expected = int(tokens[0]) if tokens[0] in "2345" else 0
    except ValueError:
        num_points_expected = 0  # Should not happen if tokens[0] is one of "2345"

    # Define a cycle of colours for multiple coordinate sets
    point_colour_tags = ["[white]", "[#91E3FF]", "[#FFF3AF]", "[#91E3FF]"]

    for i in range(num_points_expected):
        if coord_start_index + 3 <= len(
            tokens
        ):  # Check if enough tokens remain for 3 coordinates
            s_parts_formatted.append(
                pprint_coord_str(
                    tokens[coord_start_index : coord_start_index + 3],
                    colour_tag=point_colour_tags[
                        i % len(point_colour_tags)
                    ],  # Cycle through colours
                )
            )
            coord_start_index += 3
        else:  # Not enough tokens for a full coordinate set, stop processing
            break

    # Append any remaining tokens (e.g., for type 5 optional lines control points) as plain white
    if coord_start_index < len(tokens):
        s_parts_formatted.append(
            f"[white]{' '.join(tokens[coord_start_index:])}[/white]"
        )

    return " ".join(s_parts_formatted)


def pprint_line0(line_str: str) -> str:
    """
    Pretty-prints an LDraw type 0 line (meta-command or comment) using Rich styling.
    Identifies known LDraw commands, meta-tags, filenames, and hex colours for styling.
    General comments are dimmed.

    Args:
        line_str: The raw LDraw type 0 line string.

    Returns:
        A Rich-formatted string for the type 0 line.
    """
    s_parts_formatted: List[str] = []
    line_str_cleaned = line_str.rstrip()
    tokens = line_str_cleaned.split()

    if len(tokens) < 1:  # Should at least have "0"
        return f"[dim]{line_str_cleaned}[/dim]"  # Treat as simple comment

    # Line type (0)
    s_parts_formatted.append(f"[bold white]{tokens[0]}[/bold white]")

    if len(tokens) < 2:  # Just "0", treat as empty comment line
        return " ".join(s_parts_formatted)  # Will just be "0" styled

    command_token = tokens[1]
    command_token_upper = command_token.upper()

    # Check if it's a known LDraw command, meta-tag, or starts with '!' (LPUB etc.)
    if (
        command_token_upper in LDRAW_TOKENS
        or command_token.startswith("!")  # Common for extensions like LPUB, !PY
        or command_token_upper in META_TOKENS
    ):
        # Determine style based on command type
        tag_colour_for_command = "#7096FF"  # Default for general LDraw commands
        if "FILE" in command_token_upper:  # "0 FILE ..."
            tag_colour_for_command = "#B7E67A"
        elif command_token.startswith("!"):  # LPUB, LPEdit, !PY etc.
            tag_colour_for_command = "#78D4FE"
        elif command_token_upper in META_TOKENS:  # BFC, ROTSTEP, etc.
            tag_colour_for_command = "#BA7AE4"

        s_parts_formatted.append(
            f"[bold {tag_colour_for_command}]{command_token}[/bold {tag_colour_for_command}]"
        )

        # Process remaining tokens on the line
        if len(tokens) > 2:
            remaining_text_str = " ".join(tokens[2:])
            # Special handling if remaining text looks like a filename
            if remaining_text_str.lower().endswith((".ldr", ".dat")):
                file_ext_colour_tag = (
                    "#B7E67A"
                    if remaining_text_str.lower().endswith(".ldr")
                    else "#F27759"
                )
                s_parts_formatted.append(
                    f"[{file_ext_colour_tag}]{remaining_text_str}[/{file_ext_colour_tag}]"
                )
            else:  # Process other tokens individually for potential styling
                formatted_remaining_parts = []
                for token_item in tokens[2:]:
                    if token_item.upper() in META_TOKENS:  # Style nested meta tokens
                        formatted_remaining_parts.append(
                            f"[bold #BA7AE4]{token_item}[/bold #BA7AE4]"
                        )
                    elif is_hex_colour(token_item):  # Style hex colour codes
                        cleaned_hex_str = token_item.strip('"')
                        formatted_remaining_parts.append(
                            f"[{cleaned_hex_str} reverse]{token_item}[/]"
                        )
                    else:  # Default styling for other tokens
                        formatted_remaining_parts.append(f"[white]{token_item}[/white]")
                s_parts_formatted.append(" ".join(formatted_remaining_parts))
    else:
        # If not a recognized command, treat the rest of the line as a general comment (dimmed)
        s_parts_formatted.append(f"[dim]{' '.join(tokens[1:])}[/dim]")

    return " ".join(s_parts_formatted)


def pprint_line(line: str, lineno: Optional[int] = None, nocolour: bool = False):
    """
    Pretty-prints a single LDraw line to the console using Rich styling.
    Dispatches to specific formatting functions based on the LDraw line type.

    Args:
        line: The LDraw line string to print.
        lineno: Optional line number to prefix the output.
        nocolour: If True, prints the line without Rich styling (plain white).
    """
    line_content_stripped = line.strip()  # Remove leading/trailing whitespace
    if not line_content_stripped:  # Handle empty lines
        rich_print()  # Print a blank line via Rich
        return

    tokens = line_content_stripped.split()

    # Prepare line number prefix if provided
    s_prefix_lineno = f"[#404040]{lineno:4d} | [/]" if lineno is not None else ""

    formatted_line_output_str = (
        f"[white]{line_content_stripped}[/white]"  # Default if no styling
    )

    if (
        not nocolour and tokens
    ):  # Only apply styling if not nocolour and line has tokens
        line_type_str = tokens[0]
        if line_type_str == "1":
            formatted_line_output_str = pprint_line1(line_content_stripped)
        elif line_type_str == "0":
            formatted_line_output_str = pprint_line0(line_content_stripped)
        elif (
            line_type_str in "2345"
        ):  # Check if it's one of the geometric primitive types
            formatted_line_output_str = pprint_line2345(line_content_stripped)
        # else: line_type is unknown or not handled, uses default white formatting

    rich_print(f"{s_prefix_lineno}{formatted_line_output_str}")
