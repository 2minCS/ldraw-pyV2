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
# LDraw related helper functions

import decimal
from math import pi, cos, sin
from typing import List, Union, Any, TYPE_CHECKING, Tuple, Optional

# Explicit imports from toolbox
from toolbox import (
    Vector,
    Matrix,
    Identity,
)  # Assuming these are used or were from wildcard

# Explicit relative imports from ldrawpy package
from .constants import LDR_OPT_COLOUR, ASPECT_DICT, FLIP_DICT, LDRAW_TOKENS, META_TOKENS

# Forward declaration for LDRPart and LDRLine to avoid circular import
if TYPE_CHECKING:
    from .ldrprimitives import LDRPart, LDRAttrib, LDRLine


def quantize(x: Union[str, float, decimal.Decimal]) -> float:
    """Quantizes a string, float, or Decimal LDraw value to 4 decimal places."""
    try:
        v_str = str(x).strip()
        v = decimal.Decimal(v_str).quantize(
            decimal.Decimal("0.0001"), rounding=decimal.ROUND_HALF_UP
        )
        return float(v)
    except decimal.InvalidOperation:
        raise ValueError(f"Cannot quantize non-numeric value: {x}")


def MM2LDU(x: float) -> float:
    return x * 2.5


def LDU2MM(x: float) -> float:
    return x * 0.4


def val_units(value: float, units: str = "ldu") -> str:
    """
    Writes a floating point value in units of either mm or ldu.
    It restricts the number of decimal places to 4 and minimizes
    redundant trailing zeros (as recommended by ldraw.org)
    """
    x = value * 2.5 if units == "mm" else value
    quantized_x = quantize(x)

    # CONVERTED TO F-STRING for initial formatting
    s = f"{quantized_x:.4f}"
    s = s.rstrip("0").rstrip(".")
    if s == "-0":
        return "0 "
    return f"{s} "  # Use f-string for final space addition


def mat_str(m: Union[Tuple[float, ...], List[float]]) -> str:
    """
    Writes the values of a matrix (assumed to be a flat list/tuple of 9 elements)
    formatted by val_units.
    """
    if len(m) != 9:
        return "".join([val_units(float(v), "ldu") for v in m])
    return "".join([val_units(float(v), "ldu") for v in m])


def vector_str(p: Vector, attrib: "LDRAttrib") -> str:
    return (
        val_units(p.x, attrib.units)
        + val_units(p.y, attrib.units)
        + val_units(p.z, attrib.units)
    )


def GetCircleSegments(
    radius: float, segments: int, attrib: "LDRAttrib"
) -> List["LDRLine"]:
    from .ldrprimitives import LDRLine

    lines: List[LDRLine] = []
    if segments <= 0:
        return lines

    for seg in range(segments):
        p1 = Vector(0, 0, 0)
        p2 = Vector(0, 0, 0)
        a1 = (seg / segments) * 2.0 * pi
        a2 = ((seg + 1) / segments) * 2.0 * pi
        p1.x = radius * cos(a1)
        p1.z = radius * sin(a1)
        p2.x = radius * cos(a2)
        p2.z = radius * sin(a2)

        l = LDRLine(attrib.colour, attrib.units)
        l.p1 = p1
        l.p2 = p2
        lines.append(l)
    return lines


def ldrlist_from_parts(
    parts: Union[str, List[Union[str, "LDRPart"]]],
) -> List["LDRPart"]:
    from .ldrprimitives import LDRPart

    p_list: List[LDRPart] = []
    input_list: List[Union[str, "LDRPart"]]
    if isinstance(parts, str):
        input_list = parts.splitlines()  # type: ignore
    elif isinstance(parts, list):
        input_list = parts
    else:
        return p_list

    for item in input_list:
        if isinstance(item, LDRPart):
            p_list.append(item)
        elif isinstance(item, str):
            part_obj = LDRPart()
            if part_obj.from_str(item):
                p_list.append(part_obj)
    return p_list


def ldrstring_from_list(parts: List[Any]) -> str:
    from .ldrprimitives import LDRPart

    s_list: List[str] = []
    for p in parts:
        if isinstance(p, LDRPart):
            s_list.append(str(p))
        elif isinstance(p, str):
            s_list.append(p if p.endswith("\n") else p + "\n")
    return "".join(s_list)


def merge_same_parts(
    parts: List[Union[str, "LDRPart"]],
    other: List[Union[str, "LDRPart"]],
    ignore_colour: bool = False,
    as_str: bool = False,
) -> Union[List["LDRPart"], str]:
    op_ldr = ldrlist_from_parts(other)
    p_ldr = ldrlist_from_parts(parts)
    result_parts: List["LDRPart"] = ldrlist_from_parts(other)
    for n_part in p_ldr:
        is_already_present = False
        for o_part in op_ldr:
            if n_part.is_same(
                o_part, ignore_location=False, ignore_colour=ignore_colour
            ):
                is_already_present = True
                break
        if not is_already_present:
            result_parts.append(n_part)
    return ldrstring_from_list(result_parts) if as_str else result_parts


def remove_parts_from_list(
    parts: List[Union[str, "LDRPart"]],
    other: List[Union[str, "LDRPart"]],
    ignore_colour: bool = True,
    ignore_location: bool = True,
    exact: bool = False,
    as_str: bool = False,
) -> Union[List["LDRPart"], str]:
    pp_ldr = ldrlist_from_parts(parts)
    op_ldr = ldrlist_from_parts(other)
    np_kept: List["LDRPart"] = []
    for p_item in pp_ldr:
        should_remove = False
        for o_item in op_ldr:
            if exact:
                if p_item.is_identical(o_item):
                    should_remove = True
                    break
            elif ignore_colour and ignore_location:
                if p_item.name == o_item.name:
                    should_remove = True
                    break
            else:
                if p_item.is_same(
                    o_item, ignore_location=ignore_location, ignore_colour=ignore_colour
                ):
                    should_remove = True
                    break
        if not should_remove:
            np_kept.append(p_item)
    return ldrstring_from_list(np_kept) if as_str else np_kept


def norm_angle(a: float) -> float:
    a %= 360.0
    if a > 180.0:
        a -= 360.0
    elif a < -180.0:
        a += 360.0
    return a


def norm_aspect(a: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return tuple(norm_angle(v) for v in a)  # type: ignore


def _flip_x(a: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (-a[0], a[1], a[2])


def _add_aspect(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    return norm_aspect((a[0] + b[0], a[1] + b[1], a[2] + b[2]))


def preset_aspect(
    current_aspect: Tuple[float, float, float], aspect_changes: Union[str, List[str]]
) -> Tuple[float, float, float]:
    changes_list = (
        [aspect_changes.lower()]
        if isinstance(aspect_changes, str)
        else [c.lower() for c in aspect_changes]
    )
    new_aspect_list = list(current_aspect)
    for aspect_change_key in changes_list:
        if aspect_change_key in ASPECT_DICT:
            new_aspect_list = list(ASPECT_DICT[aspect_change_key])  # type: ignore
        elif aspect_change_key in FLIP_DICT:
            rot_to_add = FLIP_DICT[aspect_change_key]  # type: ignore
            new_aspect_list[0] += rot_to_add[0]
            new_aspect_list[1] += rot_to_add[1]
            new_aspect_list[2] += rot_to_add[2]
        elif aspect_change_key == "down":
            if new_aspect_list[0] < 0:
                new_aspect_list[0] = 35.0
        elif aspect_change_key == "up":
            if new_aspect_list[0] > 0:
                new_aspect_list[0] = -35.0
    return norm_aspect(tuple(new_aspect_list))  # type: ignore


def clean_line(line: str) -> str:
    sl = line.split()
    nl_parts: List[str] = []
    for i, s_token in enumerate(sl):
        xs = s_token
        is_potentially_numeric = True
        if i == 0 and s_token.isdigit():
            is_potentially_numeric = False
        if "." in s_token and any(c.isalpha() for c in s_token):
            is_potentially_numeric = False
        if s_token.upper() in LDRAW_TOKENS or s_token.upper() in META_TOKENS:
            is_potentially_numeric = False
        if s_token.startswith("!"):
            is_potentially_numeric = False

        if i > 0 and is_potentially_numeric:
            try:
                float_val = float(s_token)
                xs = val_units(float_val).strip()
            except ValueError:
                pass
        nl_parts.append(xs)
    return " ".join(nl_parts)


def clean_file(
    fn: str, fno: Optional[str] = None, verbose: bool = False, as_str: bool = False
) -> Union[None, List[str]]:
    output_filename = fno if fno is not None else fn.replace(".ldr", "_clean.ldr")
    if output_filename == fn and not as_str:
        # CONVERTED TO F-STRING
        print(
            f"Error: Cleaned output filename '{output_filename}' is same as input '{fn}'. Aborting to prevent overwrite."
        )
        print("Specify a different output filename (fno) or use as_str=True.")
        return None if not as_str else []

    cleaned_lines: List[str] = []
    bytes_in = 0
    bytes_out = 0
    try:
        with open(fn, "r", encoding="utf-8") as f_in:
            for line_content in f_in:
                bytes_in += len(line_content.encode("utf-8"))
                cleaned_line_content = clean_line(line_content.rstrip("\r\n"))
                bytes_out += len(cleaned_line_content.encode("utf-8"))
                cleaned_lines.append(cleaned_line_content)
    except FileNotFoundError:
        # CONVERTED TO F-STRING
        print(f"Error: Input file '{fn}' not found for cleaning.")
        return None if not as_str else []
    except Exception as e:
        # CONVERTED TO F-STRING
        print(f"Error reading file '{fn}': {e}")
        return None if not as_str else []

    if verbose:
        savings_percent = (
            ((bytes_in - bytes_out) / bytes_in * 100.0) if bytes_in > 0 else 0
        )
        # CONVERTED TO F-STRING
        print(
            f"{fn} : {bytes_in} bytes in / {bytes_out} bytes out ({savings_percent:.1f}% saved)"
        )

    if as_str:
        return cleaned_lines
    else:
        try:
            with open(output_filename, "w", encoding="utf-8") as f_out:
                f_out.write("\n".join(cleaned_lines) + "\n")
            return None
        except Exception as e:
            # CONVERTED TO F-STRING
            print(f"Error writing cleaned file '{output_filename}': {e}")
            return None
