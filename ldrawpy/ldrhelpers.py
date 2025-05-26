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
ldrhelpers.py: LDraw related helper functions.

This module provides various utility functions for working with LDraw data,
including unit conversions, string formatting for LDraw values, geometric
calculations (like circle segments), list manipulation for LDraw parts,
angle normalization, and file cleaning utilities.
"""

import decimal
from math import pi, cos, sin
from pathlib import Path
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

# Forward declaration for LDRPart and LDRLine to avoid circular import at runtime
# For type checking, these are resolved by MyPy if ldrprimitives is in the path.
if TYPE_CHECKING:
    from .ldrprimitives import LDRPart, LDRAttrib, LDRLine


def quantize(x: Union[str, float, decimal.Decimal]) -> float:
    """
    Quantizes a numeric LDraw value (string, float, or Decimal) to 4 decimal places.
    Uses ROUND_HALF_UP rounding.

    Args:
        x: The numeric value to quantize.

    Returns:
        The quantized value as a float.

    Raises:
        ValueError: If the input cannot be converted to a Decimal.
    """
    try:
        v_str = str(x).strip()
        # Quantize to 4 decimal places, rounding half up
        v_decimal = decimal.Decimal(v_str).quantize(
            decimal.Decimal("0.0001"), rounding=decimal.ROUND_HALF_UP
        )
        return float(v_decimal)
    except decimal.InvalidOperation:
        raise ValueError(f"Cannot quantize non-numeric value: {x}")


def MM2LDU(x: float) -> float:
    """Converts millimeters to LDraw Units (LDU). 1 LDU = 0.4mm, so 1mm = 2.5 LDU."""
    return x * 2.5


def LDU2MM(x: float) -> float:
    """Converts LDraw Units (LDU) to millimeters. 1 LDU = 0.4mm."""
    return x * 0.4


def val_units(value: float, units: str = "ldu") -> str:
    """
    Formats a numeric value for LDraw output.
    If units are "mm", converts to LDU. Then quantizes the value and formats
    it as a string, stripping trailing zeros and decimal points if unnecessary,
    and ensuring a trailing space as per LDraw convention.

    Args:
        value: The numeric value.
        units: The units of the input value ("ldu" or "mm"). Defaults to "ldu".

    Returns:
        A string formatted for LDraw output (e.g., "20 ", "-0.5 ", "0 ").
    """
    # Convert to LDU if input is in millimeters
    value_in_ldu: float = value * 2.5 if units.lower() == "mm" else value
    quantized_value: float = quantize(value_in_ldu)

    # Format to ensure enough precision for stripping, then strip trailing zeros and decimal point
    s: str = f"{quantized_value:.4f}"
    s = s.rstrip("0").rstrip(".")

    if s == "-0":  # Handle negative zero case, LDraw uses "0"
        return "0 "
    return f"{s} "  # Add trailing space as per LDraw convention


def mat_str(m_elements: Sequence[float]) -> str:
    """
    Converts a sequence of 9 floats (representing a 3x3 matrix in row-major order)
    into an LDraw matrix string. Each element is formatted using `val_units`.

    Args:
        m_elements: A sequence (e.g., tuple or list) of 9 float values.

    Returns:
        An LDraw formatted matrix string.
        If input sequence does not have 9 elements, it processes them as is,
        which might lead to malformed LDraw if not handled by caller.
    """
    # LDraw matrix string consists of 9 space-separated values.
    # No path operations here.
    if len(m_elements) != 9:
        # This case should ideally be an error or handled more robustly.
        # Retaining original behavior of processing non-9-element sequences.
        return "".join([val_units(float(v_elem), "ldu") for v_elem in m_elements])
    return "".join([val_units(float(v_elem), "ldu") for v_elem in m_elements])


def vector_str(p_vector: Vector, attrib: "LDRAttrib") -> str:  # type: ignore
    """
    Converts a toolbox.Vector object into an LDraw coordinate string (x y z),
    respecting the units specified in the LDRAttrib object.

    Args:
        p_vector: The toolbox.Vector object.
        attrib: The LDRAttrib object containing unit information.

    Returns:
        An LDraw formatted coordinate string.
    """
    return (
        val_units(p_vector.x, attrib.units)
        + val_units(p_vector.y, attrib.units)
        + val_units(p_vector.z, attrib.units)
    )


def GetCircleSegments(
    radius: float, segments: int, attrib: "LDRAttrib"
) -> List["LDRLine"]:
    """
    Generates a list of LDRLine objects representing the segments of a circle
    in the XZ plane, centered at the origin.

    Args:
        radius: The radius of the circle.
        segments: The number of line segments to approximate the circle.
        attrib: The LDRAttrib to apply to each line segment (for colour and units).

    Returns:
        A list of LDRLine objects.
    """
    from .ldrprimitives import (
        LDRLine,
    )  # Local import to prevent circular dependency at module level

    lines_list: List[LDRLine] = []
    if segments <= 0:  # Cannot create a circle with zero or negative segments
        return lines_list

    for seg_idx in range(segments):
        p1_vec: Vector = Vector(0, 0, 0)  # type: ignore # Start point of the current segment
        p2_vec: Vector = Vector(0, 0, 0)  # type: ignore # End point of the current segment

        # Calculate angles for the start and end points of the segment
        angle1: float = (seg_idx / segments) * 2.0 * pi
        angle2: float = ((seg_idx + 1) / segments) * 2.0 * pi

        # Calculate coordinates on the circle (in XZ plane, Y is assumed to be 0)
        p1_vec.x = radius * cos(angle1)
        p1_vec.z = radius * sin(angle1)
        p2_vec.x = radius * cos(angle2)
        p2_vec.z = radius * sin(angle2)

        # Create LDRLine object for this segment
        line_segment_obj: "LDRLine" = LDRLine(attrib.colour, attrib.units)
        line_segment_obj.p1 = p1_vec
        line_segment_obj.p2 = p2_vec
        lines_list.append(line_segment_obj)
    return lines_list


def ldrlist_from_parts(
    parts_input_source: Union[str, Sequence[Union[str, "LDRPart"]]],
) -> List["LDRPart"]:
    """
    Converts various forms of LDraw part data into a standardized list of LDRPart objects.
    Input can be a single multi-line LDraw string or a sequence (list/tuple)
    containing LDRPart objects and/or LDraw string lines.

    Args:
        parts_input_source: The source of LDraw part data.

    Returns:
        A list of LDRPart objects. Malformed string lines are skipped.
    """
    from .ldrprimitives import LDRPart  # Local import for LDRPart class

    ldr_part_objects_list: List[LDRPart] = []
    items_to_process: Sequence[Union[str, "LDRPart"]]

    if isinstance(parts_input_source, str):
        # If input is a single string, split it into lines
        items_to_process = parts_input_source.splitlines()
    elif isinstance(parts_input_source, (list, tuple)):
        # If input is already a list or tuple
        items_to_process = parts_input_source  # type: ignore
    else:
        # Invalid input type, return empty list
        return ldr_part_objects_list

    for item_data_entry in items_to_process:
        if isinstance(item_data_entry, LDRPart):
            ldr_part_objects_list.append(item_data_entry)
        elif isinstance(item_data_entry, str):
            # Attempt to parse the string as an LDRPart
            part_obj_from_string = LDRPart()
            if part_obj_from_string.from_str(item_data_entry):
                ldr_part_objects_list.append(part_obj_from_string)
            # else: line was not a valid LDRPart string, so it's skipped
    return ldr_part_objects_list


def ldrstring_from_list(parts_sequence: Sequence[Union["LDRPart", str]]) -> str:
    """
    Converts a sequence of LDRPart objects or LDraw strings into a single
    LDraw formatted string, ensuring each entry ends with a newline.

    Args:
        parts_sequence: A sequence (list, tuple) of LDRPart objects or strings.

    Returns:
        A single string representing the LDraw content.
    """
    string_lines_list: List[str] = []
    for p_item_entry in parts_sequence:
        item_as_string = str(
            p_item_entry
        )  # Uses LDRPart.__str__ or converts string to string
        # Ensure each line ends with a newline character
        string_lines_list.append(
            item_as_string if item_as_string.endswith("\n") else item_as_string + "\n"
        )
    return "".join(string_lines_list)


def merge_same_parts(
    parts_list1: Sequence[Union[str, "LDRPart"]],
    parts_list2: Sequence[Union[str, "LDRPart"]],
    ignore_colour: bool = False,
    as_str: bool = False,
) -> Union[List["LDRPart"], str]:
    """
    Merges two lists of parts. Parts from `parts_list1` are added to `parts_list2`
    only if an "same" part (based on name, location, and optionally colour)
    is not already present in `parts_list2`.

    Args:
        parts_list1: The first list of parts (source).
        parts_list2: The second list of parts (target/base for merging).
        ignore_colour: If True, colour is ignored during comparison.
        as_str: If True, returns the merged list as an LDraw string;
                otherwise, returns a list of LDRPart objects.

    Returns:
        A list of LDRPart objects or an LDraw string.
    """
    list1_ldr_objects: List["LDRPart"] = ldrlist_from_parts(parts_list1)
    list2_ldr_objects: List["LDRPart"] = ldrlist_from_parts(parts_list2)

    # Start with a copy of the second list; parts from the first list will be added if unique.
    merged_result_parts: List["LDRPart"] = list(list2_ldr_objects)

    for p1_item_object in list1_ldr_objects:
        is_already_present = False
        for result_item_object in merged_result_parts:
            # LDRPart.is_same handles comparison of name, location, matrix, and optionally colour.
            # ignore_location=False means location IS considered for sameness.
            if p1_item_object.is_same(
                result_item_object, ignore_location=False, ignore_colour=ignore_colour
            ):
                is_already_present = True
                break
        if not is_already_present:
            merged_result_parts.append(p1_item_object)

    return ldrstring_from_list(merged_result_parts) if as_str else merged_result_parts


def remove_parts_from_list(
    main_parts_list_source: Sequence[Union[str, "LDRPart"]],
    parts_to_remove_source: Sequence[Union[str, "LDRPart"]],
    ignore_colour: bool = True,
    ignore_location: bool = True,
    exact_match_required: bool = False,
    as_str: bool = False,
) -> Union[List["LDRPart"], str]:
    """
    Removes parts specified in `parts_to_remove_source` from `main_parts_list_source`.
    Comparison logic depends on `ignore_colour`, `ignore_location`, and
    `exact_match_required` flags.

    Args:
        main_parts_list_source: The list from which parts will be removed.
        parts_to_remove_source: The list of parts to remove.
        ignore_colour: If True, colour is ignored during comparison.
        ignore_location: If True, location is ignored.
        exact_match_required: If True, uses `LDRPart.is_identical` for comparison
                              (name, colour, location, matrix must all match).
        as_str: If True, returns the filtered list as an LDraw string.

    Returns:
        A list of LDRPart objects or an LDraw string.
    """
    main_ldr_part_objects: List["LDRPart"] = ldrlist_from_parts(main_parts_list_source)
    remove_ldr_part_objects: List["LDRPart"] = ldrlist_from_parts(
        parts_to_remove_source
    )
    kept_ldr_parts: List["LDRPart"] = []

    for p_item_main_obj in main_ldr_part_objects:
        should_remove_this_part = False
        for o_item_to_remove_obj in remove_ldr_part_objects:
            if exact_match_required:  # Strictest comparison: all attributes must match
                if p_item_main_obj.is_identical(o_item_to_remove_obj):
                    should_remove_this_part = True
                    break
            elif ignore_colour and ignore_location:  # Simplest match: only by part name
                if p_item_main_obj.name == o_item_to_remove_obj.name:
                    should_remove_this_part = True
                    break
            else:  # Use LDRPart.is_same with specified ignore flags
                if p_item_main_obj.is_same(
                    o_item_to_remove_obj,
                    ignore_location=ignore_location,
                    ignore_colour=ignore_colour,
                ):
                    should_remove_this_part = True
                    break
        if not should_remove_this_part:
            kept_ldr_parts.append(p_item_main_obj)

    return ldrstring_from_list(kept_ldr_parts) if as_str else kept_ldr_parts


def norm_angle(angle_degrees: float) -> float:
    """Normalizes an angle in degrees to the range (-180, 180]."""
    angle_degrees %= 360.0
    if angle_degrees > 180.0:
        angle_degrees -= 360.0
    elif angle_degrees <= -180.0:  # Ensure -180 is chosen over +180
        angle_degrees += 360.0
    # The original had `if a == 180.0: a = -180.0`.
    # The above logic should handle this correctly by mapping 180 to -180 via the modulo and subtraction.
    # For example, 180 % 360 = 180. 180 is not > 180.
    # Let's ensure 180 becomes -180 for consistency if that's the desired LDraw standard.
    # A common convention is that -180 and 180 are equivalent, often preferring -180.
    # The previous logic `elif a < -180.0` would make -180 stay -180.
    # `if a == 180.0: a = -180.0` was a direct assignment.
    # Let's refine:
    if abs(angle_degrees - 180.0) < 1e-9:  # If it's effectively 180
        return -180.0
    if abs(angle_degrees + 180.0) < 1e-9:  # If it's effectively -180
        return -180.0  # Ensure it stays -180
    return angle_degrees


def norm_aspect(
    aspect_angles: Tuple[float, float, float],
) -> Tuple[float, float, float]:
    """Normalizes a 3-tuple of Euler angles (aspect) using `norm_angle` for each component."""
    return (
        norm_angle(aspect_angles[0]),
        norm_angle(aspect_angles[1]),
        norm_angle(aspect_angles[2]),
    )


def _flip_x(aspect_angles: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Flips the X component of an aspect tuple. (Seems unused currently).
    Note: This flips the sign of the X-axis rotation.
    """
    # This function seems unused based on a search of the provided codebase.
    # If confirmed unused, it could be removed. Retaining for now.
    return (-aspect_angles[0], aspect_angles[1], aspect_angles[2])


def _add_aspect(
    aspect1: Tuple[float, float, float], aspect2: Tuple[float, float, float]
) -> Tuple[float, float, float]:
    """
    Adds two aspect tuples component-wise and normalizes the result. (Seems unused currently).
    """
    # This function also seems unused in the current codebase. Retaining for now.
    result_tuple = (
        aspect1[0] + aspect2[0],
        aspect1[1] + aspect2[1],
        aspect1[2] + aspect2[2],
    )
    return norm_aspect(cast(Tuple[float, float, float], result_tuple))


def preset_aspect(
    current_aspect_tuple: Tuple[float, float, float],
    aspect_changes_input: Union[str, List[str]],
) -> Tuple[float, float, float]:
    """
    Applies preset aspect changes (e.g., "front", "iso45", "flipx") to a current aspect tuple.
    Changes can be a single string command or a list of commands applied sequentially.

    Args:
        current_aspect_tuple: The starting aspect (Euler angles as a 3-tuple).
        aspect_changes_input: A string or list of strings representing aspect changes.

    Returns:
        A new 3-tuple representing the modified and normalized aspect.
    """
    # Normalize input to a list of lowercase strings for consistent key matching
    changes_to_apply_lower: List[str] = (
        [aspect_changes_input.lower()]
        if isinstance(aspect_changes_input, str)
        else [change_cmd.lower() for change_cmd in aspect_changes_input]
    )

    # Start with current_aspect and modify it based on the list of changes
    modified_aspect_list: List[float] = list(current_aspect_tuple)

    for aspect_key_command in changes_to_apply_lower:
        if aspect_key_command in ASPECT_DICT:
            # If key is a direct aspect name (e.g., "front", "iso45"), set to its defined tuple
            modified_aspect_list = list(ASPECT_DICT[aspect_key_command])
        elif aspect_key_command in FLIP_DICT:
            # If key is a flip operation (e.g., "flipx"), add the flip rotation components
            rotation_to_add_for_flip: Tuple[float, float, float] = FLIP_DICT[
                aspect_key_command
            ]
            modified_aspect_list[0] += rotation_to_add_for_flip[0]
            modified_aspect_list[1] += rotation_to_add_for_flip[1]
            modified_aspect_list[2] += rotation_to_add_for_flip[2]
        # Special handling for "up" and "down" - this logic might need context or review for specific intent
        elif aspect_key_command == "down":
            if (
                modified_aspect_list[0] < 0
            ):  # If looking somewhat up (negative X rotation)
                modified_aspect_list[0] = (
                    35.0  # Set to a specific "downward" angle (positive X)
                )
        elif aspect_key_command == "up":
            if (
                modified_aspect_list[0] > 0
            ):  # If looking somewhat down (positive X rotation)
                modified_aspect_list[0] = (
                    -35.0
                )  # Set to a specific "upward" angle (negative X)
        # Unrecognized commands are ignored

    # Normalize the final aspect angles after all changes have been applied
    final_normalized_aspect_tuple = cast(
        Tuple[float, float, float], tuple(modified_aspect_list)
    )
    return norm_aspect(final_normalized_aspect_tuple)


def clean_line(line_str: str) -> str:
    """
    Cleans a single LDraw line by reformatting its numeric coordinate values
    to a standard precision and stripping unnecessary trailing zeros/decimal points.
    Non-numeric tokens, command keywords, and the line type token are preserved.

    Args:
        line_str: The LDraw line string to clean.

    Returns:
        The cleaned LDraw line string.
    """
    tokens_list: List[str] = line_str.split()
    cleaned_tokens_list: List[str] = []

    for i, current_token_str in enumerate(tokens_list):
        output_token_str: str = current_token_str  # Default to original token

        # Determine if the token should be considered for numeric quantization
        is_eligible_for_quantization: bool = True
        if (
            i == 0 and current_token_str.isdigit()
        ):  # First token (line type) is not quantized
            is_eligible_for_quantization = False
        elif "." in current_token_str and any(
            c.isalpha() for c in current_token_str
        ):  # Likely a filename
            is_eligible_for_quantization = False
        elif (
            current_token_str.upper() in LDRAW_TOKENS
            or current_token_str.upper() in META_TOKENS
        ):  # Known command/meta keywords
            is_eligible_for_quantization = False
        elif current_token_str.startswith("!"):  # Special LPUB/other meta commands
            is_eligible_for_quantization = False

        # Attempt to quantize if eligible (and not the line type token)
        if i > 0 and is_eligible_for_quantization:
            try:
                float_value_of_token: float = float(current_token_str)
                # val_units handles quantization and LDraw formatting (includes trailing space)
                output_token_str = val_units(
                    float_value_of_token
                ).strip()  # Strip space for re-joining
            except ValueError:
                # Not a float, keep the original token string
                pass
        cleaned_tokens_list.append(output_token_str)
    return " ".join(cleaned_tokens_list)


def clean_file(
    fn_in_path_or_str: Union[str, Path],
    fno_out_path_or_str: Optional[Union[str, Path]] = None,
    verbose: bool = False,
    as_str: bool = False,
) -> Union[None, List[str]]:
    """
    Cleans an LDraw file by reformatting coordinate values in each line.
    Writes the cleaned content to a new file or returns it as a list of strings.

    Args:
        fn_in_path_or_str: Path to the input LDraw file (string or Path object).
        fno_out_path_or_str: Optional path for the output cleaned LDraw file.
                             If None, defaults to "<input_stem>_clean<input_suffix>"
                             in the same directory as the input file.
        verbose: If True, prints statistics about the cleaning process.
        as_str: If True, returns a list of cleaned lines instead of writing to a file.

    Returns:
        A list of cleaned lines if `as_str` is True.
        None if `as_str` is False (indicates file write operation was attempted,
        success or failure is printed to console).
    """
    input_path_obj = Path(fn_in_path_or_str).expanduser()

    output_filepath_obj: Path
    if fno_out_path_or_str is not None:
        output_filepath_obj = Path(fno_out_path_or_str).expanduser().resolve()
    else:
        output_filepath_obj = input_path_obj.with_name(
            f"{input_path_obj.stem}_clean{input_path_obj.suffix}"
        ).resolve()

    input_filepath_resolved_obj = input_path_obj.resolve()

    if output_filepath_obj == input_filepath_resolved_obj and not as_str:
        print(
            f"Error: Cleaned output filepath '{str(output_filepath_obj)}' is the same as input "
            f"'{str(input_filepath_resolved_obj)}'. Aborting to prevent overwrite."
        )
        print(
            "Specify a different output filename (fno_out_path_or_str) or use as_str=True."
        )
        return None if not as_str else []

    cleaned_lines_list: List[str] = []
    bytes_in_count: int = 0
    bytes_out_count: int = 0

    try:
        with open(input_filepath_resolved_obj, "r", encoding="utf-8") as f_in_handle:
            for line_content_str in f_in_handle:
                bytes_in_count += len(line_content_str.encode("utf-8"))
                cleaned_single_line: str = clean_line(line_content_str.rstrip("\r\n"))
                bytes_out_count += len(cleaned_single_line.encode("utf-8"))
                cleaned_lines_list.append(cleaned_single_line)
    except FileNotFoundError:
        print(
            f"Error: Input file '{str(input_filepath_resolved_obj)}' not found for cleaning."
        )
        return None if not as_str else []
    except Exception as e:
        print(f"Error reading file '{str(input_filepath_resolved_obj)}': {e}")
        return None if not as_str else []

    if verbose:
        savings_percentage: float = (
            ((bytes_in_count - bytes_out_count) / bytes_in_count * 100.0)
            if bytes_in_count > 0
            else 0.0
        )
        print(
            f"Cleaned '{input_filepath_resolved_obj.name}': "
            f"{bytes_in_count} bytes in / {bytes_out_count} bytes out "
            f"({savings_percentage:.1f}% saved)"
        )

    if as_str:
        return cleaned_lines_list
    else:
        try:
            output_filepath_obj.parent.mkdir(parents=True, exist_ok=True)
            with open(output_filepath_obj, "w", encoding="utf-8") as f_out_handle:
                f_out_handle.write("\n".join(cleaned_lines_list) + "\n")
            # print(f"Cleaned file written to: {str(output_filepath_obj)}") # Optional success message
            return None
        except Exception as e:
            print(f"Error writing cleaned file '{str(output_filepath_obj)}': {e}")
            return None
