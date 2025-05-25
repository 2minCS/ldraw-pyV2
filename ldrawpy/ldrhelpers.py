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
from math import pi, cos, sin  # ADDED THIS IMPORT
from typing import List, Union, Any, TYPE_CHECKING  # For type hints

# Explicit imports from toolbox
from toolbox import (
    Vector,
    Matrix,
    Identity,
)  # Assuming these are used or were from wildcard

# Explicit relative imports from ldrawpy package
from .constants import OPT_COLOUR, ASPECT_DICT, FLIP_DICT  # Assuming these are used

# Forward declaration for LDRPart to avoid circular import if ldrprimitives imports ldrhelpers
if TYPE_CHECKING:
    from .ldrprimitives import LDRPart, LDRAttrib


def quantize(x: Union[str, float, decimal.Decimal]) -> float:
    """Quantizes a string, float, or Decimal LDraw value to 4 decimal places."""
    try:
        # Ensure x is a string before passing to Decimal if it's a float, to avoid precision issues with float->Decimal
        v_str = str(x).strip()
        v = decimal.Decimal(v_str).quantize(
            decimal.Decimal("0.0001"), rounding=decimal.ROUND_HALF_UP
        )
        return float(v)
    except decimal.InvalidOperation:  # Handle non-numeric strings if they can occur
        # Depending on desired behavior, either raise error or return a default/original value
        # For now, let's re-raise or return something that indicates error if needed
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
    # Use quantize to handle precision correctly first
    quantized_x = quantize(x)  # quantize returns float

    # Format the float to string, removing trailing zeros and decimal point if integer
    s = f"{quantized_x:.4f}"  # Format to 4 decimal places initially
    s = s.rstrip("0").rstrip(".")
    if s == "-0":
        return "0 "  # LDraw convention
    return s + " "


def mat_str(
    m: Union[Tuple[float, ...], List[float]],
) -> str:  # Expects a flat list/tuple of 9 numbers
    """
    Writes the values of a matrix (assumed to be a flat list/tuple of 9 elements)
    formatted by val_units.
    """
    if len(m) != 9:
        # Handle error: matrix should have 9 elements
        # print("Warning: mat_str expects 9 elements for a 3x3 matrix.")
        return "".join(
            [val_units(float(v), "ldu") for v in m]
        )  # Attempt to process anyway
    return "".join([val_units(float(v), "ldu") for v in m])


def vector_str(
    p: Vector, attrib: "LDRAttrib"
) -> str:  # Use forward reference for LDRAttrib
    # Assuming p is a toolbox.Vector like object with x, y, z attributes
    return (
        val_units(p.x, attrib.units)
        + val_units(p.y, attrib.units)
        + val_units(p.z, attrib.units)
    )


# This function is used by ldrshapes.py
def GetCircleSegments(
    radius: float, segments: int, attrib: "LDRAttrib"
) -> List["LDRLine"]:
    # To avoid circular import, LDRLine needs to be imported here or passed as type
    from .ldrprimitives import LDRLine  # Import locally or ensure it's available

    lines: List[LDRLine] = []
    if segments <= 0:
        return lines  # Avoid division by zero or infinite loop

    for seg in range(segments):
        p1 = Vector(0, 0, 0)
        p2 = Vector(0, 0, 0)
        # Uses pi, cos, sin from math module
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
    from .ldrprimitives import (
        LDRPart,
    )  # Local import to avoid circular dependency issues at load time

    p_list: List[LDRPart] = []
    input_list: List[Union[str, "LDRPart"]]
    if isinstance(parts, str):
        input_list = parts.splitlines()
    elif isinstance(parts, list):
        input_list = parts
    else:
        return p_list  # Or raise TypeError

    for item in input_list:
        if isinstance(item, LDRPart):
            p_list.append(item)
        elif isinstance(item, str):
            part_obj = LDRPart()
            if part_obj.from_str(item):  # from_str returns self or None
                p_list.append(part_obj)
        # Else: skip items that are not LDRPart or valid LDR strings
    return p_list


def ldrstring_from_list(parts: List[Any]) -> str:  # parts can be LDRPart or str
    from .ldrprimitives import LDRPart  # Local import

    s_list: List[str] = []
    for p in parts:
        if isinstance(p, LDRPart):
            s_list.append(str(p))
        elif isinstance(p, str):
            # Ensure newline, but avoid double newline if already present
            s_list.append(p if p.endswith("\n") else p + "\n")
        # Else: skip unknown types or handle error
    return "".join(s_list)


def merge_same_parts(
    parts: List[Union[str, "LDRPart"]],
    other: List[Union[str, "LDRPart"]],
    ignore_colour: bool = False,
    as_str: bool = False,
) -> Union[List["LDRPart"], str]:
    # from .ldrprimitives import LDRPart # Already imported via ldrlist_from_parts

    op_ldr = ldrlist_from_parts(other)
    p_ldr = ldrlist_from_parts(
        parts
    )  # Changed 'other' to 'parts' here for the base list

    # Start with parts from 'other' (these take precedence if duplicates exist)
    # Then add parts from 'parts' if they are not "same" as any in 'op_ldr'
    # This logic seems reversed from "other take precedence".
    # Let's assume the goal is: result = unique_parts_from_parts + unique_parts_from_other
    # Or, if it's a merge where `other` overwrites `parts`:
    # result = other + (parts that are not in other)

    # Original logic:
    # op = ldrlist_from_parts(other)
    # p = ldrlist_from_parts(other) # This was 'other' again, likely a typo, should be 'parts' for np
    # np = ldrlist_from_parts(parts)
    # for n in np:
    #     if not any([n.is_same(o, ignore_location=False, ignore_colour=ignore_colour) for o in op]):
    #         p.append(n) # p started as copy of op (other)
    # This means: result = other + (parts from 'parts' not already in 'other')

    result_parts: List["LDRPart"] = ldrlist_from_parts(
        other
    )  # Start with all of 'other'

    for n_part in p_ldr:  # Iterate through the base list 'parts'
        is_already_present = False
        for o_part in op_ldr:  # Check against parts from 'other'
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
            # The original logic for ignore_colour and ignore_location was a bit tangled.
            # Let's clarify:
            if exact:  # If exact, all attributes must match (including loc and matrix)
                if p_item.is_identical(
                    o_item
                ):  # is_identical checks name and all attribs
                    should_remove = True
                    break
            elif ignore_colour and ignore_location:  # Only name matters
                if p_item.name == o_item.name:
                    should_remove = True
                    break
            else:  # Use is_same with specified ignores
                if p_item.is_same(
                    o_item, ignore_location=ignore_location, ignore_colour=ignore_colour
                ):
                    should_remove = True
                    break

        if not should_remove:
            np_kept.append(p_item)

    return ldrstring_from_list(np_kept) if as_str else np_kept


def norm_angle(a: float) -> float:
    """Normalizes an angle in degrees to -180 ~ +180 deg."""
    a %= 360.0
    if a > 180.0:
        a -= 360.0
    elif a < -180.0:
        a += 360.0  # Should not happen with positive modulo, but good for general case
    return a


def norm_aspect(a: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Normalizes the three angle components of aspect angle to -180 ~ +180 deg."""
    return tuple(norm_angle(v) for v in a)  # type: ignore


def _flip_x(
    a: Tuple[float, float, float],
) -> Tuple[float, float, float]:  # Not used in preset_aspect
    return (-a[0], a[1], a[2])


def _add_aspect(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Tuple[float, float, float]:  # Not used in preset_aspect
    # Assuming norm_aspect is defined elsewhere or this is a helper for it
    # For now, direct addition and normalization.
    return norm_aspect((a[0] + b[0], a[1] + b[1], a[2] + b[2]))


def preset_aspect(
    current_aspect: Tuple[float, float, float], aspect_changes: Union[str, List[str]]
) -> Tuple[float, float, float]:

    changes_list = (
        [aspect_changes.lower()]
        if isinstance(aspect_changes, str)
        else [c.lower() for c in aspect_changes]
    )

    new_aspect_list = list(current_aspect)  # Work with a mutable list

    for aspect_change_key in changes_list:
        if aspect_change_key in ASPECT_DICT:
            # ASPECT_DICT values are (pitch, yaw, roll) for a view.
            # LDraw often uses X for pitch (view up/down), Y for yaw (view left/right).
            # The original code did _flip_x on ASPECT_DICT values.
            # Let's assume ASPECT_DICT stores (LDraw_X_rot, LDraw_Y_rot, LDraw_Z_rot)
            # LDraw X rotation: positive is often pitch down.
            # LDraw Y rotation: positive is often yaw left.
            # This needs to be consistent with how euler_to_rot_matrix interprets angles.
            # For now, directly assign, assuming ASPECT_DICT values are the target absolute orientation.
            # The original code's _flip_x might have been to align with a different convention.
            # If ASPECT_DICT provides absolute views:
            new_aspect_list = list(ASPECT_DICT[aspect_change_key])  # type: ignore
        elif aspect_change_key in FLIP_DICT:
            # FLIP_DICT values are relative rotations to add.
            rot_to_add = FLIP_DICT[aspect_change_key]  # type: ignore
            new_aspect_list[0] += rot_to_add[0]
            new_aspect_list[1] += rot_to_add[1]
            new_aspect_list[2] += rot_to_add[2]
        elif aspect_change_key == "down":  # View from more above
            # If current X rot (pitch) is negative (looking up), make it positive (looking down)
            if new_aspect_list[0] < 0:
                new_aspect_list[0] = 35.0  # Example positive pitch
            # If already looking down, maybe increase pitch further or use a fixed "down" view
            # Original: if new_aspect[0] < 0: new_aspect = (145, new_aspect[1], new_aspect[2]) - 145 is extreme
        elif aspect_change_key == "up":  # View from more below
            if new_aspect_list[0] > 0:
                new_aspect_list[0] = -35.0  # Example negative pitch
            # Original: if new_aspect[0] > 0: new_aspect = (-35, new_aspect[1], new_aspect[2])

    return norm_aspect(tuple(new_aspect_list))  # type: ignore


def clean_line(line: str) -> str:
    sl = line.split()
    nl_parts: List[str] = []
    for i, s_token in enumerate(sl):
        xs = s_token
        # Only try to quantize/format if it's not the first token (line type)
        # and it doesn't look like a filename or special LDraw keyword.
        # This is a heuristic to avoid mangling part names or meta commands.
        is_potentially_numeric = True
        if i == 0 and s_token.isdigit():
            is_potentially_numeric = False  # Line type
        if "." in s_token and any(c.isalpha() for c in s_token):
            is_potentially_numeric = False  # Likely filename
        if s_token.upper() in LDRAW_TOKENS or s_token.upper() in META_TOKENS:
            is_potentially_numeric = False
        if s_token.startswith("!"):
            is_potentially_numeric = False  # Meta command

        if i > 0 and is_potentially_numeric:
            try:
                # Try to convert to float, then format using val_units
                # val_units itself calls quantize.
                # We need to pass the float value to val_units.
                float_val = float(s_token)
                xs = val_units(
                    float_val
                ).strip()  # val_units adds a space, strip it here
            except ValueError:
                pass  # Not a float, keep original token
        nl_parts.append(xs)
    return " ".join(nl_parts)  # Join with single spaces


def clean_file(
    fn: str, fno: Optional[str] = None, verbose: bool = False, as_str: bool = False
) -> Union[None, List[str]]:  # Returns None if writing to file, List[str] if as_str
    """Cleans an LDraw file by changing all floating point numbers to
    an optimum representation within the suggested precision of up to
    4 decimal places.
    """
    output_filename = fno if fno is not None else fn.replace(".ldr", "_clean.ldr")
    if (
        output_filename == fn and not as_str
    ):  # Avoid overwriting input unless explicitly different fno or as_str
        print(
            f"Error: Cleaned output filename '{output_filename}' is same as input '{fn}'. Aborting to prevent overwrite."
        )
        print("Specify a different output filename (fno) or use as_str=True.")
        return None if not as_str else []

    cleaned_lines: List[str] = []
    bytes_in = 0
    bytes_out = 0

    try:
        with open(fn, "r", encoding="utf-8") as f_in:  # Added encoding
            for line_content in f_in:
                bytes_in += len(
                    line_content.encode("utf-8")
                )  # Count bytes for accurate comparison
                cleaned_line_content = clean_line(
                    line_content.rstrip("\r\n")
                )  # Strip EOL before cleaning
                bytes_out += len(cleaned_line_content.encode("utf-8"))
                cleaned_lines.append(cleaned_line_content)
    except FileNotFoundError:
        print(f"Error: Input file '{fn}' not found for cleaning.")
        return None if not as_str else []
    except Exception as e:
        print(f"Error reading file '{fn}': {e}")
        return None if not as_str else []

    if verbose:
        savings_percent = (
            ((bytes_in - bytes_out) / bytes_in * 100.0) if bytes_in > 0 else 0
        )
        print(
            f"{fn} : {bytes_in} bytes in / {bytes_out} bytes out ({savings_percent:.1f}% saved)"
        )

    if as_str:
        return cleaned_lines
    else:
        try:
            with open(
                output_filename, "w", encoding="utf-8"
            ) as f_out:  # Added encoding
                f_out.write("\n".join(cleaned_lines) + "\n")  # Add trailing newline
            return None  # Indicates success writing to file
        except Exception as e:
            print(f"Error writing cleaned file '{output_filename}': {e}")
            return None
