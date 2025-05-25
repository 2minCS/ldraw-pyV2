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
# LDraw pretty printer helper functions

import string
from typing import List, Optional  # For type hints

from rich import print as rich_print  # Use rich.print

# Assuming LDRColour and constants are needed
from .ldrcolour import LDRColour
from .constants import (
    LDRAW_TOKENS,
    META_TOKENS,
)  # If used by is_hex_colour or other logic


def is_hex_colour(text: str) -> bool:
    text = text.strip('"')  # Remove quotes if present
    if not len(text) == 7:
        return False
    if not text.startswith("#"):  # Use startswith for clarity
        return False
    hs = text.lstrip("#")
    return all(c in string.hexdigits for c in hs)


def pprint_ldr_colour(code_str: str) -> str:  # Parameter is string code
    # CONVERTED TO F-STRINGS
    if code_str == "16" or code_str == "24":  # Compare with strings
        return f"[bold navajo_white1]{code_str}[/bold navajo_white1]"  # Ensure closing tag matches opening
    if code_str == "0":
        return f"[bold]{code_str}[/bold]"  # Simpler bold for 0
        # return f"[bold yellow reverse]{code_str}[not reverse][/bold yellow reverse]" # Alternative if specific style needed

    try:
        code_int = int(code_str)
        colour = LDRColour(code_int)
        # Use colour.as_hex() which should return RRGGBB
        return f"[#{colour.as_hex()} reverse]{code_str}[/]"  # Removed [not reverse], Rich handles nesting
    except ValueError:  # If code_str is not a valid int for LDRColour
        return f"[white]{code_str}[/white]"  # Default for unparseable codes


def pprint_coord_str(
    v_coords: List[str], colour_tag: str = "[white]"
) -> str:  # v_coords is list of 3 strings
    # CONVERTED TO F-STRING
    if len(v_coords) == 3:
        return f"[not bold]{colour_tag}{v_coords[0]} {v_coords[1]} {v_coords[2]}[/not bold]"
    return ""  # Or handle error for incorrect v_coords length


def pprint_line1(line: str) -> str:
    s_parts: List[str] = []
    line = line.rstrip()
    ls = line.split()

    if len(ls) < 14:
        return f"[white]{line}[/white]"  # Not a valid line 1

    s_parts.append(f"[bold white]{ls[0]}[/bold white]")
    s_parts.append(pprint_ldr_colour(ls[1]))
    s_parts.append(pprint_coord_str(ls[2:5]))
    s_parts.append(pprint_coord_str(ls[5:8], colour_tag="[#91E3FF]"))
    s_parts.append(pprint_coord_str(ls[8:11], colour_tag="[#FFF3AF]"))
    s_parts.append(pprint_coord_str(ls[11:14], colour_tag="[#91E3FF]"))

    filename_part = " ".join(ls[14:])
    if filename_part.lower().endswith(".ldr"):
        s_parts.append(f"[bold #B7E67A]{filename_part}[/bold #B7E67A]")
    else:
        s_parts.append(f"[bold #F27759]{filename_part}[/bold #F27759]")
    return " ".join(s_parts)


def pprint_line2345(line: str) -> str:  # For lines 2, 3, 4, 5
    s_parts: List[str] = []
    line = line.rstrip()
    ls = line.split()

    if len(ls) < 2:
        return f"[white]{line}[/white]"  # Not enough parts

    s_parts.append(f"[bold white]{ls[0]}[/bold white]")
    s_parts.append(pprint_ldr_colour(ls[1]))

    coord_idx = 2
    # Line 2 (2 points, 6 coords)
    # Line 3 (3 points, 9 coords)
    # Line 4 (4 points, 12 coords)
    # Line 5 (optional line, 2 points, 6 coords)
    num_points = int(ls[0]) if ls[0] in "2345" else 0

    # Define colors for points if desired, or use default
    point_colours = [
        "[white]",
        "[#91E3FF]",
        "[#FFF3AF]",
        "[#91E3FF]",
    ]  # Cycle through for points

    for i in range(num_points):
        if coord_idx + 3 <= len(ls):
            color_tag_for_point = point_colours[i % len(point_colours)]
            s_parts.append(
                pprint_coord_str(
                    ls[coord_idx : coord_idx + 3], colour_tag=color_tag_for_point
                )
            )
            coord_idx += 3
        else:
            break  # Not enough coordinates left for a full point

    return " ".join(s_parts)


def pprint_line0(line: str) -> str:
    s_parts: List[str] = []
    line = line.rstrip()
    ls = line.split()

    if len(ls) < 2:
        return f"[dim]{line}[/dim]"  # Default for short/unknown 0-lines

    s_parts.append(f"[bold white]{ls[0]}[/bold white]")  # The '0'

    command = ls[1]
    command_upper = command.upper()

    if (
        command_upper in LDRAW_TOKENS
        or command_upper.startswith("!")
        or command_upper in META_TOKENS
    ):
        if "FILE" in command_upper:  # "0 FILE model.ldr"
            s_parts.append(f"[bold #B7E67A]{command}[/bold #B7E67A]")
        elif command.startswith("!"):  # E.g. "!LPUB", "!PY"
            s_parts.append(f"[bold #78D4FE]{command}[/bold #78D4FE]")
        elif command_upper in META_TOKENS:  # E.g. "ROTSTEP", "COLOR"
            s_parts.append(f"[bold #BA7AE4]{command}[/bold #BA7AE4]")
        else:  # Other LDRAW_TOKENS like "STEP", "NOFILE"
            s_parts.append(f"[bold #7096FF]{command}[/bold #7096FF]")

        # Remaining parts of the line
        if len(ls) > 2:
            remaining_text = " ".join(ls[2:])
            # Simple heuristic for filenames, could be more robust
            if remaining_text.lower().endswith(".ldr"):
                s_parts.append(f"[#B7E67A]{remaining_text}[/#B7E67A]")
            elif remaining_text.lower().endswith(".dat"):
                s_parts.append(f"[#F27759]{remaining_text}[/#F27759]")
            else:
                # For other arguments, check for special formatting like hex colors
                formatted_remaining_parts = []
                for token in ls[2:]:
                    if token.upper() in META_TOKENS:
                        formatted_remaining_parts.append(
                            f"[bold #BA7AE4]{token}[/bold #BA7AE4]"
                        )
                    elif is_hex_colour(token):
                        # Use the hex value directly for Rich color styling
                        # Ensure no extra quotes around token for Rich color tag
                        clean_hex = token.strip('"')
                        formatted_remaining_parts.append(
                            f"[{clean_hex} reverse]{token}[/]"
                        )
                    else:
                        formatted_remaining_parts.append(f"[white]{token}[/white]")
                s_parts.append(" ".join(formatted_remaining_parts))
    else:
        # General comment line, print dimmed
        s_parts.append(f"[dim]{' '.join(ls[1:])}[/dim]")

    return " ".join(s_parts)


def pprint_line(line: str, lineno: Optional[int] = None, nocolour: bool = False):
    # Ensure line is a string and strip any leading/trailing whitespace that's not part of content
    line_content = line.strip()
    if not line_content:  # Skip empty lines
        rich_print()  # Print a blank line for spacing if desired, or just return
        return

    ls = line_content.split()
    s_prefix = ""
    if lineno is not None:
        s_prefix = f"[#404040]{lineno:4d} | [/]"  # Use f-string for line number

    formatted_line_str = ""
    if nocolour or not ls:  # If no-color mode or empty line after strip
        formatted_line_str = f"[white]{line_content}[/white]"
    else:
        line_type = ls[0]
        if line_type == "1":
            formatted_line_str = pprint_line1(line_content)
        elif line_type == "0":
            formatted_line_str = pprint_line0(line_content)
        elif line_type in "2345":  # Check if it's one of these characters
            formatted_line_str = pprint_line2345(line_content)
        else:  # Unknown line type or malformed
            formatted_line_str = f"[white]{line_content}[/white]"

    rich_print(f"{s_prefix}{formatted_line_str}")
