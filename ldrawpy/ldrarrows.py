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
# LDraw arrow callout utilties

import copy
from math import sin, cos, pi
from typing import (
    List,
    Tuple,
    Union,
    Optional,
    Any,
    Dict,
    Set,
    TypedDict,
    TypeVar,
    Callable,
    cast,
)

from toolbox import Vector, Matrix, Identity, ZAxis, safe_vector  # type: ignore
from .ldrprimitives import LDRPart

# Constants defining LDraw comment lines for LPUB arrow generation
ARROW_PREFIX = """0 BUFEXCHG A STORE"""
ARROW_PLI = """0 !LPUB PLI BEGIN IGN"""
ARROW_SUFFIX = """0 !LPUB PLI END
0 STEP
0 BUFEXCHG A RETRIEVE"""
ARROW_PLI_SUFFIX = """0 !LPUB PLI END"""

ARROW_PARTS = ["hashl2.dat", "hashl3.dat", "hashl4.dat", "hashl5.dat", "hashl6.dat"]

ARROW_MZ = Matrix([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # type: ignore
ARROW_PZ = Matrix([[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])  # type: ignore
ARROW_MX = Matrix([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # type: ignore
ARROW_PX = Matrix([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # type: ignore
ARROW_MY = Matrix([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])  # type: ignore
ARROW_PY = Matrix([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])  # type: ignore

# Define a TypeVar for generic type usage in value_after_token
_VT = TypeVar("_VT")


# Define a TypedDict for the structure of arrow parameter dictionaries
class ArrowParameters(TypedDict):
    line: str
    colour: int
    length: int
    offset: List[Vector]  # type: ignore
    invert: bool
    ratio: float
    tilt: float


def value_after_token(
    tokens: List[str],
    value_token: str,
    default_val: _VT,  # Default value of type _VT
    xtype: Callable[[str], _VT],  # Conversion function from str to _VT
) -> _VT:  # Returns type _VT
    """
    Finds a specific token in a list of string tokens and returns the next token,
    converted to the specified type (xtype).
    If the token is not found, or the next token doesn't exist, or conversion fails,
    it returns default_val.
    """
    try:
        idx = tokens.index(value_token)
        if idx + 1 < len(tokens):
            return xtype(tokens[idx + 1])
    except (ValueError, TypeError):
        pass
    return default_val


def norm_angle_arrow(angle: float) -> float:
    """
    Normalizes an angle, likely for arrow rotation steps.
    """
    return angle % 45.0


def vectorize_arrow(s_coords: List[str]) -> Optional[Vector]:  # type: ignore
    """
    Converts a list of 3 string coordinates (e.g., ["10", "0", "-20"])
    into a toolbox.Vector object. Returns None if input is not 3 strings
    or if conversion to float fails.
    """
    if len(s_coords) == 3:
        try:
            return Vector(*(float(x) for x in s_coords))  # type: ignore
        except ValueError:
            return None
    return None


class ArrowContext:
    """
    Holds context and settings for generating LDraw arrows, such as default
    colour, length, scale, and current processing state for arrow commands.
    """

    colour: int
    length: int
    scale: float
    yscale: float
    offset: List[Vector]  # type: ignore
    rotstep: Vector  # type: ignore
    ratio: float
    outline_colour: int

    def __init__(self, colour: int = 804, length: int = 2):
        self.colour, self.length = colour, length
        self.scale, self.yscale, self.ratio = 25.0, 20.0, 0.5
        self.offset = []
        self.rotstep = Vector(0, 0, 0)  # type: ignore
        self.outline_colour = 804

    def part_for_length(self, length_val: int) -> str:
        """Selects an LDraw arrow part name (e.g., "hashl2.dat") based on desired length category."""
        if length_val <= 2:
            return "hashl2.dat"
        if length_val == 3:
            return "hashl3.dat"
        if length_val == 4:
            return "hashl4.dat"
        if length_val == 5:
            return "hashl5.dat"
        if length_val >= 6:
            return "hashl6.dat"
        return "hashl5.dat"

    def matrix_for_offset(self, offset_vector: Vector, axis_mask: str = "", invert_direction: bool = False, tilt_angle: float = 0.0) -> Matrix:  # type: ignore
        """
        Determines the rotation matrix for an arrow based on an offset vector,
        an axis mask (which axes are NOT primary directions), and inversion/tilt flags.
        """
        abs_x, abs_y, abs_z = (
            abs(offset_vector.x),
            abs(offset_vector.y),
            abs(offset_vector.z),
        )
        base_rotation_matrix = Identity()  # type: ignore

        if "x" not in axis_mask and abs_x > max(abs_y, abs_z, 1e-9):
            base_rotation_matrix = (
                (ARROW_PX if offset_vector.x < 0 else ARROW_MX)
                if not invert_direction
                else (ARROW_MX if offset_vector.x < 0 else ARROW_PX)
            )
        elif "y" not in axis_mask and abs_y > max(abs_x, abs_z, 1e-9):
            base_rotation_matrix = (
                (ARROW_MY if offset_vector.y < 0 else ARROW_PY)
                if not invert_direction
                else (ARROW_PY if offset_vector.y < 0 else ARROW_MY)
            )
        elif "z" not in axis_mask and abs_z > max(abs_x, abs_y, 1e-9):
            base_rotation_matrix = (
                (ARROW_PZ if offset_vector.z < 0 else ARROW_MZ)
                if not invert_direction
                else (ARROW_MZ if offset_vector.z < 0 else ARROW_PZ)
            )

        return base_rotation_matrix.rotate(tilt_angle, ZAxis) if tilt_angle != 0.0 and ZAxis else base_rotation_matrix  # type: ignore

    def loc_for_offset(self, offset_vector: Vector, arrow_length_category: int, axis_mask: str = "", position_ratio: float = 0.5) -> Vector:  # type: ignore
        """
        Calculates the location for an arrow head based on the target offset_vector,
        arrow_length_category, axis_mask, and position_ratio.
        """
        arrow_loc_vector = Vector(0, 0, 0)  # type: ignore

        scaled_len_x_adj = float(arrow_length_category) * position_ratio * self.scale
        scaled_len_y_adj = float(arrow_length_category) * position_ratio * self.yscale
        scaled_len_z_adj = float(arrow_length_category) * position_ratio * self.scale

        if "x" not in axis_mask:
            direction_x_sign = (
                1 if offset_vector.x > 1e-9 else (-1 if offset_vector.x < -1e-9 else 0)
            )
            arrow_loc_vector.x = (offset_vector.x / 2.0) - (
                direction_x_sign * scaled_len_x_adj
            )
        if "y" not in axis_mask:
            direction_y_sign = (
                1 if offset_vector.y > 1e-9 else (-1 if offset_vector.y < -1e-9 else 0)
            )
            arrow_loc_vector.y = (offset_vector.y / 2.0) - (
                direction_y_sign * scaled_len_y_adj
            )
        if "z" not in axis_mask:
            direction_z_sign = (
                1 if offset_vector.z > 1e-9 else (-1 if offset_vector.z < -1e-9 else 0)
            )
            arrow_loc_vector.z = (offset_vector.z / 2.0) - (
                direction_z_sign * scaled_len_z_adj
            )

        if "x" in axis_mask:
            arrow_loc_vector.x += offset_vector.x
        if "y" in axis_mask:
            arrow_loc_vector.y += offset_vector.y
        if "z" in axis_mask:
            arrow_loc_vector.z += offset_vector.z
        return arrow_loc_vector

    def part_loc_for_offset(self, offset_vector: Vector, axis_mask: str = "") -> Vector:  # type: ignore
        """
        Determines the effective location of the part being pointed to by the arrow.
        """
        part_effective_location = Vector(0, 0, 0)  # type: ignore
        if "x" not in axis_mask:
            part_effective_location.x = offset_vector.x
        if "y" not in axis_mask:
            part_effective_location.y = offset_vector.y
        if "z" not in axis_mask:
            part_effective_location.z = offset_vector.z
        return part_effective_location

    def _mask_axis(self, offset_vectors_list: List[Vector]) -> str:  # type: ignore
        """
        Analyzes a list of offset vectors to determine which axes show variation.
        Returns a string containing 'x', 'y', or 'z' if the respective coordinates vary.
        """
        if not offset_vectors_list or len(offset_vectors_list) <= 1:
            return ""

        min_coords = Vector(offset_vectors_list[0].x, offset_vectors_list[0].y, offset_vectors_list[0].z)  # type: ignore
        max_coords = Vector(offset_vectors_list[0].x, offset_vectors_list[0].y, offset_vectors_list[0].z)  # type: ignore

        for ov_item in offset_vectors_list[1:]:
            min_coords.x = min(min_coords.x, ov_item.x)
            min_coords.y = min(min_coords.y, ov_item.y)
            min_coords.z = min(min_coords.z, ov_item.z)
            max_coords.x = max(max_coords.x, ov_item.x)
            max_coords.y = max(max_coords.y, ov_item.y)
            max_coords.z = max(max_coords.z, ov_item.z)

        mask_str = ""
        tolerance = 1e-6
        if abs(max_coords.x - min_coords.x) > tolerance:
            mask_str += "x"
        if abs(max_coords.y - min_coords.y) > tolerance:
            mask_str += "y"
        if abs(max_coords.z - min_coords.z) > tolerance:
            mask_str += "z"
        return mask_str

    def arrow_from_dict(
        self, arrow_data_dict: ArrowParameters
    ) -> str:  # UPDATED type hint
        """Generates LDraw string for arrows based on a dictionary of parameters."""
        arrows_str_list: List[str] = []
        offset_vectors_from_dict: List[Vector] = arrow_data_dict["offset"]  # type: ignore

        if not (
            isinstance(offset_vectors_from_dict, list)
            and all(isinstance(v, Vector) for v in offset_vectors_from_dict)
        ):  # type: ignore
            return ""

        axis_variation_mask = self._mask_axis(offset_vectors_from_dict)

        for single_offset_vec in offset_vectors_from_dict:
            original_part_ref = LDRPart()
            if original_part_ref.from_str(arrow_data_dict["line"]) is None:
                continue

            arrow_part_obj = LDRPart()
            arrow_part_obj.name = self.part_for_length(arrow_data_dict["length"])

            arrow_base_offset_loc = self.loc_for_offset(
                single_offset_vec,
                arrow_data_dict["length"],
                axis_variation_mask,
                position_ratio=arrow_data_dict["ratio"],
            )
            arrow_part_obj.attrib.loc = original_part_ref.attrib.loc + arrow_base_offset_loc  # type: ignore

            arrow_part_obj.attrib.matrix = self.matrix_for_offset(
                single_offset_vec,
                axis_variation_mask,
                invert_direction=arrow_data_dict["invert"],
                tilt_angle=arrow_data_dict["tilt"],
            )
            arrow_part_obj.attrib.colour = arrow_data_dict["colour"]
            arrows_str_list.append(str(arrow_part_obj))

        return "".join(arrows_str_list)

    def dict_for_line(
        self,
        ldraw_line_str: str,
        invert_arrow_direction: bool,
        arrow_pos_ratio: float,
        arrow_colour_override: Optional[int] = None,
        arrow_tilt_angle: float = 0.0,
    ) -> ArrowParameters:  # UPDATED return type hint
        """Creates a dictionary of parameters for generating an arrow for a given LDraw line."""
        params: ArrowParameters = {
            "line": ldraw_line_str,
            "colour": (
                arrow_colour_override
                if arrow_colour_override is not None
                else self.colour
            ),
            "length": self.length,
            "offset": copy.deepcopy(self.offset),
            "invert": invert_arrow_direction,
            "ratio": arrow_pos_ratio,
            "tilt": arrow_tilt_angle,
        }
        return params


def arrows_for_step(
    arrow_ctx: ArrowContext,
    step_content_str: str,
    generate_lpub_meta: bool = True,
    output_only_arrows: bool = False,
    return_as_dict_list: bool = False,
) -> Union[str, List[ArrowParameters]]:  # UPDATED return type hint
    """
    Processes an LDraw step's content to generate arrow annotations based on
    !PY ARROW meta commands found within the step.
    """
    processed_lines_for_output: List[str] = []
    arrow_data_to_generate: List[ArrowParameters] = []  # UPDATED type hint
    original_part_lines_from_arrow_blocks: List[str] = []

    is_inside_arrow_meta_block = False
    current_block_offset_vectors: List[Vector] = []  # type: ignore

    current_block_effective_arrow_colour = arrow_ctx.colour
    current_block_effective_arrow_length = arrow_ctx.length
    current_block_effective_arrow_ratio = arrow_ctx.ratio
    current_block_effective_arrow_tilt = 0.0

    for line_str_from_step in step_content_str.splitlines():
        stripped_line_content = line_str_from_step.lstrip()
        line_type_char = stripped_line_content[0] if stripped_line_content else ""

        if line_type_char == "0":
            tokens_in_line_upper = line_str_from_step.upper().split()
            is_py_arrow_command = (
                "!PY" in tokens_in_line_upper and "ARROW" in tokens_in_line_upper
            )

            if is_py_arrow_command:
                original_case_tokens = (
                    line_str_from_step.split()
                )  # Use original case for parsing values
                if "BEGIN" in tokens_in_line_upper:
                    is_inside_arrow_meta_block = True
                    current_block_offset_vectors = []

                    coord_candidate_tokens = [
                        tkn
                        for tkn in original_case_tokens
                        if tkn.upper()
                        not in {
                            "0",
                            "!PY",
                            "ARROW",
                            "BEGIN",
                            "COLOUR",
                            "LENGTH",
                            "RATIO",
                            "TILT",
                        }
                        and not tkn.upper().isalpha()
                    ]

                    idx = 0
                    while idx + 2 < len(coord_candidate_tokens):
                        vec = vectorize_arrow(coord_candidate_tokens[idx : idx + 3])
                        if vec:
                            current_block_offset_vectors.append(vec)
                        idx += 3
                    arrow_ctx.offset = current_block_offset_vectors

                    current_block_effective_arrow_colour = value_after_token(
                        original_case_tokens, "COLOUR", arrow_ctx.colour, int
                    )
                    current_block_effective_arrow_length = value_after_token(
                        original_case_tokens, "LENGTH", arrow_ctx.length, int
                    )
                    current_block_effective_arrow_ratio = value_after_token(
                        original_case_tokens, "RATIO", arrow_ctx.ratio, float
                    )
                    current_block_effective_arrow_tilt = value_after_token(
                        original_case_tokens, "TILT", 0.0, float
                    )
                    arrow_ctx.colour = current_block_effective_arrow_colour
                    arrow_ctx.length = current_block_effective_arrow_length

                elif "END" in tokens_in_line_upper and is_inside_arrow_meta_block:
                    is_inside_arrow_meta_block = False
                    arrow_ctx.offset = []

                elif not is_inside_arrow_meta_block:
                    arrow_ctx.colour = value_after_token(
                        original_case_tokens, "COLOUR", arrow_ctx.colour, int
                    )
                    arrow_ctx.length = value_after_token(
                        original_case_tokens, "LENGTH", arrow_ctx.length, int
                    )

            if not output_only_arrows:
                if not (generate_lpub_meta and is_py_arrow_command):
                    processed_lines_for_output.append(line_str_from_step)

        elif line_type_char == "1":
            if is_inside_arrow_meta_block:
                original_part_lines_from_arrow_blocks.append(line_str_from_step)
                data_normal_direction = arrow_ctx.dict_for_line(
                    line_str_from_step,
                    False,
                    current_block_effective_arrow_ratio,
                    current_block_effective_arrow_colour,
                    current_block_effective_arrow_tilt,
                )
                data_inverted_direction = arrow_ctx.dict_for_line(
                    line_str_from_step,
                    True,
                    current_block_effective_arrow_ratio,
                    current_block_effective_arrow_colour,
                    current_block_effective_arrow_tilt,
                )
                arrow_data_to_generate.extend(
                    [data_normal_direction, data_inverted_direction]
                )

            if not output_only_arrows:
                if not (generate_lpub_meta and is_inside_arrow_meta_block):
                    processed_lines_for_output.append(line_str_from_step)

        elif not output_only_arrows:
            processed_lines_for_output.append(line_str_from_step)

    if return_as_dict_list:
        return arrow_data_to_generate

    if generate_lpub_meta:
        lpub_output_lines: List[str] = []
        for ln in processed_lines_for_output:
            if not ("!PY" in ln.upper() and "ARROW" in ln.upper()):
                lpub_output_lines.append(ln)

        if original_part_lines_from_arrow_blocks or arrow_data_to_generate:
            lpub_output_lines.append(ARROW_PREFIX)
            unique_part_lines_processed: Set[str] = set()
            for arrow_dict_item in arrow_data_to_generate:
                original_line_str = arrow_dict_item["line"]
                if original_line_str not in unique_part_lines_processed:
                    part_obj_original = LDRPart()
                    if part_obj_original.from_str(original_line_str):
                        if arrow_dict_item["offset"]:
                            first_offset_vec = arrow_dict_item["offset"][0]
                            axis_var_mask = arrow_ctx._mask_axis(
                                arrow_dict_item["offset"]
                            )
                            part_eff_loc = arrow_ctx.part_loc_for_offset(
                                first_offset_vec, axis_var_mask
                            )
                            part_obj_original.attrib.loc += part_eff_loc  # type: ignore
                        lpub_output_lines.append(str(part_obj_original).strip())
                        unique_part_lines_processed.add(original_line_str)

            lpub_output_lines.append(ARROW_PLI)
            for arrow_dict_item in arrow_data_to_generate:
                arrow_ldr_str = arrow_ctx.arrow_from_dict(arrow_dict_item)
                if arrow_ldr_str:
                    lpub_output_lines.append(arrow_ldr_str.strip())
            lpub_output_lines.append(ARROW_SUFFIX)

            lpub_output_lines.append(ARROW_PLI)
            for orig_part_line in original_part_lines_from_arrow_blocks:
                lpub_output_lines.append(orig_part_line.strip())
            lpub_output_lines.append(ARROW_PLI_SUFFIX)

        return "\n".join(lpub_output_lines) + ("\n" if lpub_output_lines else "")

    else:
        final_direct_output_lines: List[str] = []
        if not output_only_arrows:
            final_direct_output_lines.extend(
                ln
                for ln in processed_lines_for_output
                if not ("!PY" in ln.upper() and "ARROW" in ln.upper())
            )
        for arrow_dict_item in arrow_data_to_generate:
            arrow_ldr_str = arrow_ctx.arrow_from_dict(arrow_dict_item)
            if arrow_ldr_str:
                final_direct_output_lines.append(arrow_ldr_str.strip())

        return "\n".join(final_direct_output_lines) + (
            "\n" if final_direct_output_lines else ""
        )


def arrows_for_lpub_file(input_filename: str, output_filename: str):
    """
    Processes an entire LDraw file (potentially MPD), finds !PY ARROW commands
    in each step of each model/submodel, and generates LPUB-compatible arrow
    meta-commands, writing the result to output_filename.
    """
    arrow_context_global = ArrowContext()
    try:
        with open(input_filename, "rt", encoding="utf-8") as fp_in_handle:
            full_file_content = fp_in_handle.read()
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
        return
    except Exception as e:
        print(f"Error reading input file '{input_filename}': {e}")
        return

    output_all_final_string_blocks: List[str] = []

    file_content_blocks = full_file_content.split("0 FILE")

    first_block_is_model_content = not full_file_content.strip().startswith(
        "0 FILE"
    ) and bool(file_content_blocks)

    start_block_processing_idx = 0
    if first_block_is_model_content:
        steps_in_first_block = file_content_blocks[0].split("0 STEP")
        for i, step_text_content in enumerate(steps_in_first_block):
            if (
                i == 0
                and not step_text_content.strip()
                and len(steps_in_first_block) > 1
            ):
                continue

            processed_step_output_fb = arrows_for_step(  # Renamed to avoid conflict
                arrow_context_global, step_text_content, generate_lpub_meta=True
            )
            processed_step_output_str_fb = cast(
                str, processed_step_output_fb
            )  # Cast to str

            step_prefix_fb = ""
            if (
                i > 0
                and processed_step_output_str_fb.strip()
                and not processed_step_output_str_fb.strip().startswith(
                    ARROW_PREFIX.strip()
                )
                and not processed_step_output_str_fb.strip().startswith("0 STEP")
            ):
                step_prefix_fb = "0 STEP\n"

            if processed_step_output_str_fb.strip():
                output_all_final_string_blocks.append(
                    step_prefix_fb + processed_step_output_str_fb
                )
        start_block_processing_idx = 1

    for i in range(start_block_processing_idx, len(file_content_blocks)):
        current_file_block_text_with_header = file_content_blocks[i]
        if not current_file_block_text_with_header.strip():
            continue

        current_model_full_str_content = (
            "0 FILE " + current_file_block_text_with_header.strip()
        )

        model_lines = current_model_full_str_content.splitlines()
        if not model_lines:
            continue

        output_all_final_string_blocks.append(model_lines[0])

        content_for_step_splitting = "\n".join(model_lines[1:])
        steps_content_in_this_block = content_for_step_splitting.split("0 STEP")

        for j, step_text_content_sub in enumerate(steps_content_in_this_block):
            if (
                j == 0
                and not step_text_content_sub.strip()
                and len(steps_content_in_this_block) > 1
            ):
                continue

            processed_step_output_sub = arrows_for_step(  # Renamed
                arrow_context_global, step_text_content_sub, generate_lpub_meta=True
            )
            processed_step_output_str_sub = cast(str, processed_step_output_sub)  # Cast

            step_prefix_sub = ""
            if (
                j > 0
                and processed_step_output_str_sub.strip()
                and not processed_step_output_str_sub.strip().startswith(
                    ARROW_PREFIX.strip()
                )
                and not processed_step_output_str_sub.strip().startswith("0 STEP")
            ):
                step_prefix_sub = "0 STEP\n"

            if processed_step_output_str_sub.strip():
                output_all_final_string_blocks.append(
                    step_prefix_sub + processed_step_output_str_sub
                )

    final_output_file_content = "\n".join(
        s_block.strip()
        for s_block in output_all_final_string_blocks
        if isinstance(s_block, str) and s_block.strip()
    )
    if final_output_file_content and not final_output_file_content.endswith("\n"):
        final_output_file_content += "\n"

    try:
        with open(output_filename, "w", encoding="utf-8") as fpo_handle:
            fpo_handle.write(final_output_file_content)
    except Exception as e:
        print(f"Error writing output file '{output_filename}': {e}")


def remove_offset_parts(
    parts_list_main: List[Union[LDRPart, str]],
    parts_list_original_positions: List[Union[LDRPart, str]],
    arrow_definitions_list: List[ArrowParameters],  # UPDATED type hint
    return_as_strings: bool = False,
) -> Union[List[LDRPart], List[str]]:
    """
    Removes parts from parts_list_main that are considered "offset versions"
    of parts in parts_list_original_positions, based on arrow definitions.
    """
    main_parts_as_objects: List[LDRPart] = []
    for item_m in parts_list_main:
        p_obj_m = LDRPart()
        if isinstance(item_m, LDRPart):
            main_parts_as_objects.append(item_m)
        elif isinstance(item_m, str) and p_obj_m.from_str(item_m):
            main_parts_as_objects.append(p_obj_m)

    original_parts_as_objects: List[LDRPart] = []
    for item_o in parts_list_original_positions:
        p_obj_o = LDRPart()
        if isinstance(item_o, LDRPart):
            original_parts_as_objects.append(item_o)
        elif isinstance(item_o, str) and p_obj_o.from_str(item_o):
            original_parts_as_objects.append(p_obj_o)

    arrow_graphic_part_names: Set[str] = set(
        ARROW_PARTS
    )  # Use defined arrow part names
    all_arrow_offset_vectors_world: List[Vector] = []  # type: ignore

    for arrow_dict_item_def in arrow_definitions_list:
        # arrow_dict_item_def is now ArrowParameters, so keys are guaranteed (by type checker)
        offsets_in_this_arrow_def = arrow_dict_item_def["offset"]  # type: ignore
        if isinstance(offsets_in_this_arrow_def, list) and all(
            isinstance(v, Vector) for v in offsets_in_this_arrow_def
        ):  # type: ignore
            all_arrow_offset_vectors_world.extend(offsets_in_this_arrow_def)  # type: ignore

    kept_parts_final: List[LDRPart] = []
    for part_candidate_to_keep in main_parts_as_objects:
        if part_candidate_to_keep.name in arrow_graphic_part_names:
            kept_parts_final.append(part_candidate_to_keep)
            continue

        is_this_an_offset_version_to_remove = False
        for original_pos_part_ref in original_parts_as_objects:
            if not (
                part_candidate_to_keep.name == original_pos_part_ref.name
                and part_candidate_to_keep.attrib.colour
                == original_pos_part_ref.attrib.colour
            ):
                continue

            candidate_loc = part_candidate_to_keep.attrib.loc
            original_loc = original_pos_part_ref.attrib.loc

            for offset_vec_world in all_arrow_offset_vectors_world:
                if candidate_loc.almost_same_as(original_loc + offset_vec_world, 0.1):  # type: ignore
                    is_this_an_offset_version_to_remove = True
                    break
            if is_this_an_offset_version_to_remove:
                break

        if not is_this_an_offset_version_to_remove:
            kept_parts_final.append(part_candidate_to_keep)

    return [str(p) for p in kept_parts_final] if return_as_strings else kept_parts_final
