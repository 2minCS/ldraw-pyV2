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
from typing import List, Optional

from rich import print as rich_print  # type: ignore

from .ldrcolour import LDRColour
from .constants import LDRAW_TOKENS, META_TOKENS


def is_hex_colour(text: str) -> bool:
    text = text.strip('"')
    if not len(text) == 7:
        return False
    if not text.startswith("#"):
        return False
    return all(c in string.hexdigits for c in text.lstrip("#"))


def pprint_ldr_colour(code_str: str) -> str:
    if code_str == "16" or code_str == "24":
        return f"[bold navajo_white1]{code_str}[/bold navajo_white1]"
    if code_str == "0":
        return f"[bold]{code_str}[/bold]"
    try:
        colour = LDRColour(int(code_str))
        return f"[#{colour.as_hex()} reverse]{code_str}[/]"
    except ValueError:
        return f"[white]{code_str}[/white]"


def pprint_coord_str(v_coords: List[str], colour_tag: str = "[white]") -> str:
    if len(v_coords) == 3:
        return f"[not bold]{colour_tag}{v_coords[0]} {v_coords[1]} {v_coords[2]}[/not bold]"
    return ""


def pprint_line1(line: str) -> str:
    s_parts: List[str] = []
    line = line.rstrip()
    ls = line.split()
    if len(ls) < 14:
        return f"[white]{line}[/white]"
    s_parts.append(f"[bold white]{ls[0]}[/bold white]")
    s_parts.append(pprint_ldr_colour(ls[1]))
    s_parts.append(pprint_coord_str(ls[2:5]))
    s_parts.append(pprint_coord_str(ls[5:8], colour_tag="[#91E3FF]"))
    s_parts.append(pprint_coord_str(ls[8:11], colour_tag="[#FFF3AF]"))
    s_parts.append(pprint_coord_str(ls[11:14], colour_tag="[#91E3FF]"))
    filename_part = " ".join(ls[14:])
    tag = "#B7E67A" if filename_part.lower().endswith(".ldr") else "#F27759"
    s_parts.append(f"[bold {tag}]{filename_part}[/bold {tag}]")
    return " ".join(s_parts)


def pprint_line2345(line: str) -> str:
    s_parts: List[str] = []
    line = line.rstrip()
    ls = line.split()
    if len(ls) < 2:
        return f"[white]{line}[/white]"
    s_parts.append(f"[bold white]{ls[0]}[/bold white]")
    s_parts.append(pprint_ldr_colour(ls[1]))
    coord_idx = 2
    num_points = int(ls[0]) if ls[0] in "2345" else 0
    point_colours = ["[white]", "[#91E3FF]", "[#FFF3AF]", "[#91E3FF]"]
    for i in range(num_points):
        if coord_idx + 3 <= len(ls):
            s_parts.append(
                pprint_coord_str(
                    ls[coord_idx : coord_idx + 3],
                    colour_tag=point_colours[i % len(point_colours)],
                )
            )
            coord_idx += 3
        else:
            break
    return " ".join(s_parts)


def pprint_line0(line: str) -> str:
    s_parts: List[str] = []
    line = line.rstrip()
    ls = line.split()
    if len(ls) < 2:
        return f"[dim]{line}[/dim]"
    s_parts.append(f"[bold white]{ls[0]}[/bold white]")
    command, cmd_upper = ls[1], ls[1].upper()
    if (
        cmd_upper in LDRAW_TOKENS
        or cmd_upper.startswith("!")
        or cmd_upper in META_TOKENS
    ):
        tag = (
            "#B7E67A"
            if "FILE" in cmd_upper
            else (
                "#78D4FE"
                if command.startswith("!")
                else "#BA7AE4" if cmd_upper in META_TOKENS else "#7096FF"
            )
        )
        s_parts.append(f"[bold {tag}]{command}[/bold {tag}]")
        if len(ls) > 2:
            rem_text = " ".join(ls[2:])
            if rem_text.lower().endswith((".ldr", ".dat")):
                tag_file = "#B7E67A" if rem_text.lower().endswith(".ldr") else "#F27759"
                s_parts.append(f"[{tag_file}]{rem_text}[/{tag_file}]")
            else:
                fmt_rem_parts = []
                for token in ls[2:]:
                    if token.upper() in META_TOKENS:
                        fmt_rem_parts.append(f"[bold #BA7AE4]{token}[/bold #BA7AE4]")
                    elif is_hex_colour(token):
                        clean_hex = token.strip('"')
                        fmt_rem_parts.append(f"[{clean_hex} reverse]{token}[/]")
                    else:
                        fmt_rem_parts.append(f"[white]{token}[/white]")
                s_parts.append(" ".join(fmt_rem_parts))
    else:
        s_parts.append(f"[dim]{' '.join(ls[1:])}[/dim]")
    return " ".join(s_parts)


def pprint_line(line: str, lineno: Optional[int] = None, nocolour: bool = False):
    line_content = line.strip()
    if not line_content:
        rich_print()
        return
    ls = line_content.split()
    s_prefix = f"[#404040]{lineno:4d} | [/]" if lineno is not None else ""
    formatted_line_str = f"[white]{line_content}[/white]"
    if not nocolour and ls:
        line_type = ls[0]
        if line_type == "1":
            formatted_line_str = pprint_line1(line_content)
        elif line_type == "0":
            formatted_line_str = pprint_line0(line_content)
        elif line_type in "2345":
            formatted_line_str = pprint_line2345(line_content)
    rich_print(f"{s_prefix}{formatted_line_str}")
