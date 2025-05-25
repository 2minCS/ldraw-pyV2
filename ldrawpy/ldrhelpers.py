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
from typing import (
    List,
    Union,
    Any,
    TYPE_CHECKING,
    Tuple,
    Optional,
    Sequence,
    cast,
)  # ADDED cast

# Explicit imports from toolbox
from toolbox import Vector, Matrix, Identity  # type: ignore

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
    x: float = value * 2.5 if units == "mm" else value
    quantized_x: float = quantize(x)
    s: str = f"{quantized_x:.4f}"
    s = s.rstrip("0").rstrip(".")
    if s == "-0":
        return "0 "
    return f"{s} "


def mat_str(m: Sequence[float]) -> str:
    if len(m) != 9:
        return "".join([val_units(float(v), "ldu") for v in m])
    return "".join([val_units(float(v), "ldu") for v in m])


def vector_str(p: Vector, attrib: "LDRAttrib") -> str:  # type: ignore
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

    for seg_idx in range(segments):
        p1: Vector = Vector(0, 0, 0)  # type: ignore
        p2: Vector = Vector(0, 0, 0)  # type: ignore
        a1: float = (seg_idx / segments) * 2.0 * pi
        a2: float = ((seg_idx + 1) / segments) * 2.0 * pi
        p1.x = radius * cos(a1)
        p1.z = radius * sin(a1)
        p2.x = radius * cos(a2)
        p2.z = radius * sin(a2)

        line_obj: "LDRLine" = LDRLine(attrib.colour, attrib.units)
        line_obj.p1 = p1
        line_obj.p2 = p2
        lines.append(line_obj)
    return lines


def ldrlist_from_parts(
    parts: Union[str, Sequence[Union[str, "LDRPart"]]],
) -> List["LDRPart"]:
    from .ldrprimitives import LDRPart

    p_list: List[LDRPart] = []
    input_items: Sequence[Union[str, "LDRPart"]]

    if isinstance(parts, str):
        input_items = parts.splitlines()
    elif isinstance(parts, (list, tuple, Sequence)):  # type: ignore
        input_items = parts
    else:
        return p_list

    for item in input_items:
        if isinstance(item, LDRPart):
            p_list.append(item)
        elif isinstance(item, str):
            part_obj = LDRPart()
            if part_obj.from_str(item):
                p_list.append(part_obj)
    return p_list


def ldrstring_from_list(parts_list: Sequence[Union["LDRPart", str]]) -> str:
    from .ldrprimitives import LDRPart

    s_list: List[str] = []
    for p_item in parts_list:
        item_str = str(p_item)
        s_list.append(item_str if item_str.endswith("\n") else item_str + "\n")
    return "".join(s_list)


def merge_same_parts(
    parts1: Sequence[Union[str, "LDRPart"]],
    parts2: Sequence[Union[str, "LDRPart"]],
    ignore_colour: bool = False,
    as_str: bool = False,
) -> Union[List["LDRPart"], str]:

    list1_ldr: List["LDRPart"] = ldrlist_from_parts(parts1)
    list2_ldr: List["LDRPart"] = ldrlist_from_parts(parts2)
    result_parts: List["LDRPart"] = list(list2_ldr)

    for p1_item in list1_ldr:
        is_already_present = False
        for res_item in result_parts:
            if p1_item.is_same(
                res_item, ignore_location=False, ignore_colour=ignore_colour
            ):
                is_already_present = True
                break
        if not is_already_present:
            result_parts.append(p1_item)

    return ldrstring_from_list(result_parts) if as_str else result_parts


def remove_parts_from_list(
    main_parts_list: Sequence[Union[str, "LDRPart"]],
    parts_to_remove: Sequence[Union[str, "LDRPart"]],
    ignore_colour: bool = True,
    ignore_location: bool = True,
    exact: bool = False,
    as_str: bool = False,
) -> Union[List["LDRPart"], str]:

    pp_ldr: List["LDRPart"] = ldrlist_from_parts(main_parts_list)
    op_ldr: List["LDRPart"] = ldrlist_from_parts(parts_to_remove)
    kept_parts: List["LDRPart"] = []

    for p_item in pp_ldr:
        should_remove_p_item = False
        for o_item_to_remove in op_ldr:
            if exact:
                if p_item.is_identical(o_item_to_remove):
                    should_remove_p_item = True
                    break
            elif ignore_colour and ignore_location:
                if p_item.name == o_item_to_remove.name:
                    should_remove_p_item = True
                    break
            else:
                if p_item.is_same(
                    o_item_to_remove,
                    ignore_location=ignore_location,
                    ignore_colour=ignore_colour,
                ):
                    should_remove_p_item = True
                    break
        if not should_remove_p_item:
            kept_parts.append(p_item)

    return ldrstring_from_list(kept_parts) if as_str else kept_parts


def norm_angle(a: float) -> float:
    a %= 360.0
    if a > 180.0:
        a -= 360.0
    elif a < -180.0:
        a += 360.0
    if a == 180.0:
        a = -180.0
    return a


def norm_aspect(a: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (norm_angle(a[0]), norm_angle(a[1]), norm_angle(a[2]))


def _flip_x(a: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return (-a[0], a[1], a[2])


def _add_aspect(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    # Explicitly create a 3-tuple for MyPy and cast it
    result_tuple = (a[0] + b[0], a[1] + b[1], a[2] + b[2])
    return norm_aspect(cast(Tuple[float, float, float], result_tuple))  # CORRECTED


def preset_aspect(
    current_aspect: Tuple[float, float, float], aspect_changes: Union[str, List[str]]
) -> Tuple[float, float, float]:
    changes_list_lower: List[str] = (
        [aspect_changes.lower()]
        if isinstance(aspect_changes, str)
        else [c.lower() for c in aspect_changes]
    )
    new_aspect_list: List[float] = list(current_aspect)
    for aspect_key in changes_list_lower:
        if aspect_key in ASPECT_DICT:
            new_aspect_list = list(ASPECT_DICT[aspect_key])
        elif aspect_key in FLIP_DICT:
            rot_to_add: Tuple[float, float, float] = FLIP_DICT[aspect_key]
            new_aspect_list[0] += rot_to_add[0]
            new_aspect_list[1] += rot_to_add[1]
            new_aspect_list[2] += rot_to_add[2]
        elif aspect_key == "down":
            if new_aspect_list[0] < 0:
                new_aspect_list[0] = 35.0
        elif aspect_key == "up":
            if new_aspect_list[0] > 0:
                new_aspect_list[0] = -35.0
    final_tuple = cast(Tuple[float, float, float], tuple(new_aspect_list))
    return norm_aspect(final_tuple)


def clean_line(line: str) -> str:
    sl: List[str] = line.split()
    nl_parts: List[str] = []
    for i, s_token in enumerate(sl):
        xs: str = s_token
        is_potentially_numeric: bool = True
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
                float_val: float = float(s_token)
                xs = val_units(float_val).strip()
            except ValueError:
                pass
        nl_parts.append(xs)
    return " ".join(nl_parts)


def clean_file(
    fn: str, fno: Optional[str] = None, verbose: bool = False, as_str: bool = False
) -> Union[None, List[str]]:
    output_filename: str = fno if fno is not None else fn.replace(".ldr", "_clean.ldr")
    if output_filename == fn and not as_str:
        print(
            f"Error: Cleaned output filename '{output_filename}' is same as input '{fn}'. Aborting to prevent overwrite."
        )
        print("Specify a different output filename (fno) or use as_str=True.")
        return None if not as_str else []
    cleaned_lines: List[str] = []
    bytes_in: int = 0
    bytes_out: int = 0
    try:
        with open(fn, "r", encoding="utf-8") as f_in:
            for line_content in f_in:
                bytes_in += len(line_content.encode("utf-8"))
                cleaned_line_content: str = clean_line(line_content.rstrip("\r\n"))
                bytes_out += len(cleaned_line_content.encode("utf-8"))
                cleaned_lines.append(cleaned_line_content)
    except FileNotFoundError:
        print(f"Error: Input file '{fn}' not found for cleaning.")
        return None if not as_str else []
    except Exception as e:
        print(f"Error reading file '{fn}': {e}")
        return None if not as_str else []
    if verbose:
        savings_percent: float = (
            ((bytes_in - bytes_out) / bytes_in * 100.0) if bytes_in > 0 else 0.0
        )
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
            print(f"Error writing cleaned file '{output_filename}': {e}")
            return None
