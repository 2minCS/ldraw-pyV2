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
from pathlib import Path  # ADDED
from typing import (
    List,
    Union,
    Any,
    TYPE_CHECKING,
    Tuple,
    Optional,
    Sequence,
    cast,
)

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
        # Quantize to 4 decimal places, rounding half up
        v = decimal.Decimal(v_str).quantize(
            decimal.Decimal("0.0001"), rounding=decimal.ROUND_HALF_UP
        )
        return float(v)
    except decimal.InvalidOperation:  # Handle cases where conversion to Decimal fails
        raise ValueError(f"Cannot quantize non-numeric value: {x}")


def MM2LDU(x: float) -> float:
    """Converts millimeters to LDraw Units (LDU). 1 LDU = 0.4mm."""
    return x * 2.5


def LDU2MM(x: float) -> float:
    """Converts LDraw Units (LDU) to millimeters."""
    return x * 0.4


def val_units(value: float, units: str = "ldu") -> str:
    """
    Formats a numeric value for LDraw output, converting to LDU if units are 'mm',
    quantizing, and stripping trailing zeros.
    """
    x: float = value * 2.5 if units == "mm" else value
    quantized_x: float = quantize(x)
    # Format to ensure decimal point for non-integers, then strip
    s: str = f"{quantized_x:.4f}"  # Ensure enough precision for stripping
    s = s.rstrip("0").rstrip(".")
    if s == "-0":  # Handle negative zero case
        return "0 "
    return f"{s} "  # Add trailing space as per LDraw convention


def mat_str(m: Sequence[float]) -> str:
    """
    Converts a sequence of 9 floats (representing a 3x3 matrix)
    into an LDraw matrix string.
    """
    # LDraw matrix string consists of 9 space-separated values
    if len(m) != 9:
        # Original code processed non-9-length sequences; retaining this for now,
        # but ideally, this should raise an error or be handled more strictly.
        return "".join([val_units(float(v), "ldu") for v in m])
    return "".join([val_units(float(v), "ldu") for v in m])


def vector_str(p: Vector, attrib: "LDRAttrib") -> str:  # type: ignore
    """
    Converts a Vector object into an LDraw coordinate string (x y z),
    respecting units specified in LDRAttrib.
    """
    return (
        val_units(p.x, attrib.units)
        + val_units(p.y, attrib.units)
        + val_units(p.z, attrib.units)
    )


def GetCircleSegments(
    radius: float, segments: int, attrib: "LDRAttrib"
) -> List["LDRLine"]:
    """
    Generates a list of LDRLine objects representing segments of a circle.
    """
    from .ldrprimitives import (
        LDRLine,
    )  # Local import to prevent circular dependency at module level

    lines: List[LDRLine] = []
    if segments <= 0:  # Cannot create a circle with no segments
        return lines

    for seg_idx in range(segments):
        p1: Vector = Vector(0, 0, 0)  # type: ignore # Start point of the segment
        p2: Vector = Vector(0, 0, 0)  # type: ignore # End point of the segment
        # Calculate angles for start and end points of the segment
        a1: float = (seg_idx / segments) * 2.0 * pi
        a2: float = ((seg_idx + 1) / segments) * 2.0 * pi
        # Calculate coordinates on the circle (in XZ plane, Y is 0)
        p1.x = radius * cos(a1)
        p1.z = radius * sin(a1)
        p2.x = radius * cos(a2)
        p2.z = radius * sin(a2)

        # Create LDRLine object for this segment
        line_obj: "LDRLine" = LDRLine(attrib.colour, attrib.units)
        line_obj.p1 = p1
        line_obj.p2 = p2
        lines.append(line_obj)
    return lines


def ldrlist_from_parts(
    parts_input: Union[str, Sequence[Union[str, "LDRPart"]]],
) -> List["LDRPart"]:
    """
    Converts a string of LDraw lines or a sequence of LDRPart/string objects
    into a list of LDRPart objects.
    """
    from .ldrprimitives import LDRPart  # Local import

    p_list: List[LDRPart] = []
    input_items_to_process: Sequence[Union[str, "LDRPart"]]

    if isinstance(parts_input, str):
        input_items_to_process = parts_input.splitlines()
    # Ensure that if it's a sequence, it's a list or tuple for MyPy
    elif isinstance(parts_input, (list, tuple)):
        input_items_to_process = parts_input  # type: ignore
    else:
        # This case should ideally not be reached if type hints are followed
        # Consider raising a TypeError or returning empty list as per original
        return p_list

    for item_data in input_items_to_process:
        if isinstance(item_data, LDRPart):
            p_list.append(item_data)
        elif isinstance(item_data, str):
            part_obj = LDRPart()
            # from_str attempts to parse the string into the LDRPart object
            if part_obj.from_str(item_data):
                p_list.append(part_obj)
            # else: line was not a valid LDRPart string, so it's skipped
    return p_list


def ldrstring_from_list(parts_list: Sequence[Union["LDRPart", str]]) -> str:
    """
    Converts a sequence of LDRPart objects or LDraw strings into a single
    LDraw formatted string.
    """
    # from .ldrprimitives import LDRPart # Not strictly needed if only using str()

    s_list: List[str] = []
    for p_item_data in parts_list:
        item_str_representation = str(
            p_item_data
        )  # Relies on LDRPart.__str__ or item itself if string
        # Ensure each line ends with a newline
        s_list.append(
            item_str_representation
            if item_str_representation.endswith("\n")
            else item_str_representation + "\n"
        )
    return "".join(s_list)


def merge_same_parts(
    parts_list1: Sequence[Union[str, "LDRPart"]],
    parts_list2: Sequence[Union[str, "LDRPart"]],
    ignore_colour: bool = False,
    as_str: bool = False,
) -> Union[List["LDRPart"], str]:
    """
    Merges two lists of parts, adding parts from parts_list1 to parts_list2
    only if they are not already present (based on name, location, and optionally colour).
    """
    list1_ldr_objects: List["LDRPart"] = ldrlist_from_parts(parts_list1)
    list2_ldr_objects: List["LDRPart"] = ldrlist_from_parts(parts_list2)

    # Start with a copy of the second list; parts from the first list will be added if unique.
    result_parts_list: List["LDRPart"] = list(list2_ldr_objects)

    for p1_item_obj in list1_ldr_objects:
        is_already_present_in_result = False
        for res_item_obj in result_parts_list:
            # LDRPart.is_same handles comparison of name, location, matrix, and optionally colour.
            if p1_item_obj.is_same(
                res_item_obj, ignore_location=False, ignore_colour=ignore_colour
            ):
                is_already_present_in_result = True
                break
        if not is_already_present_in_result:
            result_parts_list.append(p1_item_obj)

    return ldrstring_from_list(result_parts_list) if as_str else result_parts_list


def remove_parts_from_list(
    main_parts_list_input: Sequence[Union[str, "LDRPart"]],
    parts_to_remove_input: Sequence[Union[str, "LDRPart"]],
    ignore_colour: bool = True,
    ignore_location: bool = True,
    exact_match_required: bool = False,  # Renamed from 'exact' for clarity
    as_str: bool = False,
) -> Union[List["LDRPart"], str]:
    """
    Removes parts specified in parts_to_remove_input from main_parts_list_input.
    Comparison logic depends on ignore_colour, ignore_location, and exact_match_required flags.
    """
    main_ldr_objects: List["LDRPart"] = ldrlist_from_parts(main_parts_list_input)
    remove_ldr_objects: List["LDRPart"] = ldrlist_from_parts(parts_to_remove_input)
    kept_parts_list: List["LDRPart"] = []

    for p_item_main in main_ldr_objects:
        should_this_part_be_removed = False
        for o_item_to_remove in remove_ldr_objects:
            if exact_match_required:  # Strictest comparison
                if p_item_main.is_identical(o_item_to_remove):
                    should_this_part_be_removed = True
                    break
            elif ignore_colour and ignore_location:  # Simplest match: only by part name
                if p_item_main.name == o_item_to_remove.name:
                    should_this_part_be_removed = True
                    break
            else:  # Use LDRPart.is_same with specified ignore flags
                if p_item_main.is_same(
                    o_item_to_remove,
                    ignore_location=ignore_location,
                    ignore_colour=ignore_colour,
                ):
                    should_this_part_be_removed = True
                    break
        if not should_this_part_be_removed:
            kept_parts_list.append(p_item_main)

    return ldrstring_from_list(kept_parts_list) if as_str else kept_parts_list


def norm_angle(a: float) -> float:
    """Normalizes an angle to the range (-180, 180]."""
    a %= 360.0  # Ensure angle is within [0, 360) or (-360, 0) after modulo
    if a > 180.0:
        a -= 360.0
    elif a < -180.0:  # Use elif for mutually exclusive condition
        a += 360.0
    # Handle the 180 degree case to be -180 for consistency (common convention)
    if abs(a - 180.0) < 1e-9:  # Check for floating point equality
        a = -180.0
    return a


def norm_aspect(a: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Normalizes a 3-tuple of Euler angles (aspect)."""
    return (norm_angle(a[0]), norm_angle(a[1]), norm_angle(a[2]))


def _flip_x(a: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Flips the X component of an aspect tuple. (Seems unused currently)"""
    # This function seems unused based on current file content search.
    # If it's indeed unused, it could be removed. For now, kept.
    return (-a[0], a[1], a[2])


def _add_aspect(
    a: Tuple[float, float, float], b: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """Adds two aspect tuples and normalizes the result. (Seems unused currently)"""
    # This function also seems unused.
    result_tuple = (a[0] + b[0], a[1] + b[1], a[2] + b[2])
    # Ensure the result is correctly cast for norm_aspect if types are ambiguous
    return norm_aspect(cast(Tuple[float, float, float], result_tuple))


def preset_aspect(
    current_aspect: Tuple[float, float, float],
    aspect_changes_input: Union[str, List[str]],
) -> Tuple[float, float, float]:
    """
    Applies preset aspect changes (like "front", "iso45", "flipx") to a current aspect.
    """
    # Normalize input to a list of lowercase strings
    changes_list_lower: List[str] = (
        [aspect_changes_input.lower()]
        if isinstance(aspect_changes_input, str)
        else [c.lower() for c in aspect_changes_input]
    )

    # Start with current_aspect and modify it based on changes
    new_aspect_list: List[float] = list(current_aspect)

    for aspect_key_change in changes_list_lower:
        if aspect_key_change in ASPECT_DICT:
            # If key is a direct aspect name (e.g., "front", "iso45"), set to its defined tuple
            new_aspect_list = list(ASPECT_DICT[aspect_key_change])
        elif aspect_key_change in FLIP_DICT:
            # If key is a flip operation (e.g., "flipx"), add the flip rotation
            rotation_to_add: Tuple[float, float, float] = FLIP_DICT[aspect_key_change]
            new_aspect_list[0] += rotation_to_add[0]
            new_aspect_list[1] += rotation_to_add[1]
            new_aspect_list[2] += rotation_to_add[2]
        # Special handling for "up" and "down" - this logic might need context or review
        # The original logic:
        elif aspect_key_change == "down":
            if new_aspect_list[0] < 0:  # If looking somewhat up (negative X rotation)
                new_aspect_list[0] = (
                    35.0  # Set to a specific "downward" angle (positive X)
                )
        elif aspect_key_change == "up":
            if new_aspect_list[0] > 0:  # If looking somewhat down (positive X rotation)
                new_aspect_list[0] = (
                    -35.0
                )  # Set to a specific "upward" angle (negative X)

    # Normalize the final aspect angles after all changes
    final_aspect_tuple = cast(Tuple[float, float, float], tuple(new_aspect_list))
    return norm_aspect(final_aspect_tuple)


def clean_line(line: str) -> str:
    """
    Cleans an LDraw line by reformatting numeric coordinate values
    to a standard precision and stripping unnecessary trailing zeros.
    """
    sl_tokens: List[str] = line.split()
    cleaned_line_parts: List[str] = []
    for i, token_str in enumerate(sl_tokens):
        current_part_str: str = token_str
        is_potentially_numeric_for_quantization: bool = True

        # The first token (line type) is not considered numeric for quantization here
        if i == 0 and token_str.isdigit():
            is_potentially_numeric_for_quantization = False
        # If token contains both '.' and letters, it's likely not a simple number (e.g., "file.ldr", "stud.dat")
        if "." in token_str and any(c.isalpha() for c in token_str):
            is_potentially_numeric_for_quantization = False
        # If it's a known LDraw or Meta token, don't try to quantize
        if token_str.upper() in LDRAW_TOKENS or token_str.upper() in META_TOKENS:
            is_potentially_numeric_for_quantization = False
        # Tokens starting with '!' are often special commands (e.g., !LPUB, !PY)
        if token_str.startswith("!"):
            is_potentially_numeric_for_quantization = False

        # Attempt to quantize tokens after the line type if they seem numeric
        if i > 0 and is_potentially_numeric_for_quantization:
            try:
                float_value: float = float(token_str)
                # val_units includes quantization and standard LDraw formatting (trailing space)
                current_part_str = val_units(
                    float_value
                ).strip()  # Strip trailing space for re-joining
            except ValueError:
                # Not a float, keep the original token string
                pass
        cleaned_line_parts.append(current_part_str)
    return " ".join(cleaned_line_parts)


def clean_file(
    fn_in: Union[str, Path],
    fno_in: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    as_str: bool = False,
) -> Union[None, List[str]]:
    """
    Cleans an LDraw file by reformatting coordinate values in each line.
    Writes the cleaned content to a new file or returns it as a list of strings.

    Args:
        fn_in: Path to the input LDraw file (string or Path object).
        fno_in: Optional path for the output cleaned LDraw file (string or Path object).
                If None, defaults to "<input_stem>_clean<input_suffix>" in the same
                directory as the input file.
        verbose: If True, prints statistics about the cleaning process.
        as_str: If True, the function returns a list of cleaned lines
                instead of writing to a file.

    Returns:
        A list of cleaned lines if as_str is True.
        None if as_str is False (indicates file write operation was attempted).
    """
    # Convert input filename to a Path object and expand user tilde (~)
    input_path_obj = Path(fn_in).expanduser()

    output_filepath_obj: Path
    if fno_in is not None:
        # If output filename is provided, convert it to a Path object, expand, and resolve
        output_filepath_obj = Path(fno_in).expanduser().resolve()
    else:
        # Default output: "<input_stem>_clean<input_suffix>"
        # e.g., "model.ldr" -> "model_clean.ldr"
        # This places the output in the same directory as input if input_path_obj was relative.
        # .with_name() constructs a new path in the same directory as input_path_obj.
        output_filepath_obj = input_path_obj.with_name(
            f"{input_path_obj.stem}_clean{input_path_obj.suffix}"
        )
        # Resolve to make it absolute. If input_path_obj was absolute, this keeps it in the same directory.
        output_filepath_obj = output_filepath_obj.resolve()

    # Resolve the input path to an absolute path for reading
    input_filepath_resolved_obj = input_path_obj.resolve()

    # Safety check: prevent overwriting the input file if no output name was specified
    # and the operation is to write to a file (as_str is False).
    if output_filepath_obj == input_filepath_resolved_obj and not as_str:
        print(
            f"Error: Cleaned output filepath '{str(output_filepath_obj)}' is the same as input "
            f"'{str(input_filepath_resolved_obj)}'. Aborting to prevent overwrite."
        )
        print("Specify a different output filename (fno_in) or use as_str=True.")
        # Return empty list if as_str for consistency, though this path shouldn't be hit if as_str is True
        return None if not as_str else []

    cleaned_lines_list: List[str] = []
    bytes_in_count: int = 0
    bytes_out_count: int = 0

    try:
        # Open and read the input file
        with open(input_filepath_resolved_obj, "r", encoding="utf-8") as f_in_handle:
            for line_content_str in f_in_handle:
                bytes_in_count += len(line_content_str.encode("utf-8"))
                # Clean each line (rstrip to remove original newline before cleaning)
                cleaned_single_line: str = clean_line(line_content_str.rstrip("\r\n"))
                bytes_out_count += len(cleaned_single_line.encode("utf-8"))
                cleaned_lines_list.append(cleaned_single_line)
    except FileNotFoundError:
        print(
            f"Error: Input file '{str(input_filepath_resolved_obj)}' not found for cleaning."
        )
        return None if not as_str else []
    except (
        Exception
    ) as e:  # Catch other potential IOErrors or exceptions during reading
        print(f"Error reading file '{str(input_filepath_resolved_obj)}': {e}")
        return None if not as_str else []

    if verbose:
        savings_percentage: float = (
            ((bytes_in_count - bytes_out_count) / bytes_in_count * 100.0)
            if bytes_in_count > 0
            else 0.0
        )
        # Use .name for display if paths are long, for better readability
        print(
            f"Cleaned '{input_filepath_resolved_obj.name}': "
            f"{bytes_in_count} bytes in / {bytes_out_count} bytes out "
            f"({savings_percentage:.1f}% saved)"
        )

    if as_str:
        # If as_str is True, return the list of cleaned lines
        return cleaned_lines_list
    else:
        # If as_str is False, write the cleaned lines to the output file
        try:
            # Ensure parent directory exists for the output file
            output_filepath_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(output_filepath_obj, "w", encoding="utf-8") as f_out_handle:
                # Join cleaned lines with newline and add a final newline
                f_out_handle.write("\n".join(cleaned_lines_list) + "\n")
            return None  # Indicates success writing to file (consistent with original return for file mode)
        except Exception as e:  # Catch potential IOErrors or exceptions during writing
            print(f"Error writing cleaned file '{str(output_filepath_obj)}': {e}")
            return None  # Indicates failure to write
