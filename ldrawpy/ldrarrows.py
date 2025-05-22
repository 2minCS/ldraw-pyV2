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

import os
import copy
from math import sin, cos, pi  # Keep math imports
from functools import reduce
from typing import List, Tuple, Union, Optional, Any, Dict  # For type hints

# Explicit imports from toolbox
from toolbox import Vector, Matrix, Identity, ZAxis, safe_vector  # ADDED safe_vector

# Explicit imports from ldrawpy package
from .ldrprimitives import LDRPart


ARROW_PREFIX = """0 BUFEXCHG A STORE"""
ARROW_PLI = """0 !LPUB PLI BEGIN IGN"""
ARROW_SUFFIX = """0 !LPUB PLI END
0 STEP
0 BUFEXCHG A RETRIEVE"""
ARROW_PLI_SUFFIX = """0 !LPUB PLI END"""

ARROW_PARTS = ["hashl2", "hashl3", "hashl4", "hashl5", "hashl6"]

# Ensure Matrix is defined (it is, via toolbox import)
ARROW_MZ = Matrix([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
ARROW_PZ = Matrix([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
ARROW_MX = Matrix([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
ARROW_PX = Matrix([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
ARROW_MY = Matrix([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
ARROW_PY = Matrix([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


def value_after_token(
    tokens: List[str], value_token: str, default_val: Any, xtype: type = int
) -> Any:
    try:
        idx = tokens.index(value_token)
        if idx + 1 < len(tokens):
            return xtype(tokens[idx + 1])
    except ValueError:  # token not found or conversion error
        pass
    except TypeError:  # xtype conversion failed
        pass
    return default_val


def norm_angle_arrow(angle: float) -> float:
    return angle % 45.0  # Use float for consistency


def vectorize_arrow(s_coords: List[str]) -> Optional[Vector]:
    if len(s_coords) == 3:
        try:
            return Vector(*(float(x) for x in s_coords))
        except ValueError:
            return None
    return None


class ArrowContext:
    colour: int
    length: int
    scale: float
    yscale: float
    offset: List[Vector]  # Stores list of offset vectors for current arrow group
    rotstep: Vector
    ratio: float
    outline_colour: int

    def __init__(self, colour: int = 804, length: int = 2):
        self.colour = colour
        self.length = length
        self.scale = 25.0
        self.yscale = 20.0
        self.offset = []  # Initialize as empty list
        self.rotstep = Vector(0, 0, 0)
        self.ratio = 0.5
        self.outline_colour = 804

    def part_for_length(self, length: int) -> str:
        if length <= 2:
            return "hashl2"
        if length == 3:
            return "hashl3"
        if length == 4:
            return "hashl4"
        if length >= 5:
            return "hashl5"
        return "hashl2"

    def matrix_for_offset(
        self,
        offset_vec: Vector,
        mask: str = "",
        invert: bool = False,
        tilt: float = 0.0,
    ) -> Matrix:
        abs_x, abs_y, abs_z = abs(offset_vec.x), abs(offset_vec.y), abs(offset_vec.z)
        arrow_matrix_base = Identity()

        if "x" not in mask and abs_x > max(
            abs_y, abs_z, 1e-9
        ):  # Added tolerance for max comparison
            arrow_matrix_base = ARROW_PX if offset_vec.x < 0 else ARROW_MX
            if invert:
                arrow_matrix_base = ARROW_MX if offset_vec.x < 0 else ARROW_PX
        elif "y" not in mask and abs_y > max(abs_x, abs_z, 1e-9):
            arrow_matrix_base = ARROW_MY if offset_vec.y < 0 else ARROW_PY
            if invert:
                arrow_matrix_base = ARROW_PY if offset_vec.y < 0 else ARROW_MY
        elif "z" not in mask and abs_z > max(abs_x, abs_y, 1e-9):
            arrow_matrix_base = ARROW_PZ if offset_vec.z < 0 else ARROW_MZ
            if invert:
                arrow_matrix_base = ARROW_MZ if offset_vec.z < 0 else ARROW_PZ

        if tilt != 0.0 and ZAxis is not None:
            return arrow_matrix_base.rotate(tilt, ZAxis)  # type: ignore
        return arrow_matrix_base

    def loc_for_offset(
        self, offset_vec: Vector, length: int, mask: str = "", ratio: float = 0.5
    ) -> Vector:
        loc_offset = Vector(0, 0, 0)

        if "x" not in mask:
            scaled_len_x = float(length) * ratio * self.scale
            loc_offset.x = (offset_vec.x / 2.0) - (
                scaled_len_x
                if offset_vec.x > 1e-9
                else -scaled_len_x if offset_vec.x < -1e-9 else 0
            )
        if "y" not in mask:
            scaled_len_y = float(length) * ratio * self.yscale
            loc_offset.y = (offset_vec.y / 2.0) - (
                scaled_len_y
                if offset_vec.y > 1e-9
                else -scaled_len_y if offset_vec.y < -1e-9 else 0
            )
        if "z" not in mask:
            scaled_len_z = float(length) * ratio * self.scale
            loc_offset.z = (offset_vec.z / 2.0) - (
                scaled_len_z
                if offset_vec.z > 1e-9
                else -scaled_len_z if offset_vec.z < -1e-9 else 0
            )

        if "x" in mask:
            loc_offset.x += offset_vec.x
        if "y" in mask:
            loc_offset.y += offset_vec.y
        if "z" in mask:
            loc_offset.z += offset_vec.z

        return loc_offset

    def part_loc_for_offset(self, offset_vec: Vector, mask: str = "") -> Vector:
        part_final_loc = Vector(0, 0, 0)
        if "x" not in mask:
            part_final_loc.x = offset_vec.x
        if "y" not in mask:
            part_final_loc.y = offset_vec.y
        if "z" not in mask:
            part_final_loc.z = offset_vec.z
        return part_final_loc

    def _mask_axis(self, offsets_list: List[Vector]) -> str:
        if not offsets_list or len(offsets_list) <= 1:
            return ""
        min_coords = Vector(
            min(o.x for o in offsets_list),
            min(o.y for o in offsets_list),
            min(o.z for o in offsets_list),
        )
        max_coords = Vector(
            max(o.x for o in offsets_list),
            max(o.y for o in offsets_list),
            max(o.z for o in offsets_list),
        )
        mask = ""
        if abs(max_coords.x - min_coords.x) > 1e-6:
            mask += "x"
        if abs(max_coords.y - min_coords.y) > 1e-6:
            mask += "y"
        if abs(max_coords.z - min_coords.z) > 1e-6:
            mask += "z"
        return mask

    def arrow_from_dict(self, arrow_data_dict: Dict[str, Any]) -> str:
        arrows_str_list: List[str] = []
        offsets_for_this_arrow_group: List[Vector] = arrow_data_dict.get("offset", [])
        if not isinstance(offsets_for_this_arrow_group, list) or not all(
            isinstance(v, Vector) for v in offsets_for_this_arrow_group
        ):
            return ""

        mask = self._mask_axis(offsets_for_this_arrow_group)

        for single_offset_vec in offsets_for_this_arrow_group:
            lego_part_original = LDRPart()
            if lego_part_original.from_str(arrow_data_dict["line"]) is None:
                continue

            arrow_part = LDRPart()
            arrow_part.name = self.part_for_length(arrow_data_dict["length"])

            arrow_base_offset = self.loc_for_offset(
                single_offset_vec,
                arrow_data_dict["length"],
                mask,
                ratio=arrow_data_dict["ratio"],
            )
            arrow_part.attrib.loc = lego_part_original.attrib.loc + arrow_base_offset
            arrow_part.attrib.matrix = self.matrix_for_offset(
                single_offset_vec,
                mask,
                invert=arrow_data_dict["invert"],
                tilt=arrow_data_dict["tilt"],
            )
            arrow_part.attrib.colour = arrow_data_dict["colour"]
            arrows_str_list.append(str(arrow_part))

        return "".join(arrows_str_list)

    def dict_for_line(
        self,
        line_str: str,
        invert: bool,
        ratio: float,
        colour: Optional[int] = None,
        tilt: float = 0.0,
    ) -> Dict[str, Any]:
        return {
            "line": line_str,
            "colour": colour if colour is not None else self.colour,
            "length": self.length,
            "offset": copy.deepcopy(self.offset),
            "invert": invert,
            "ratio": ratio,
            "tilt": tilt,
        }


def arrows_for_step(
    arrow_ctx: ArrowContext,
    step_content: str,
    as_lpub: bool = True,
    only_arrows: bool = False,
    as_dict: bool = False,
) -> Union[str, List[Dict[str, Any]]]:
    processed_lines: List[str] = []
    arrow_data_collected: List[Dict[str, Any]] = []
    original_part_lines_from_arrow_blocks: List[str] = []

    in_arrow_meta_block = False
    current_block_offsets: List[Vector] = []
    current_block_arrow_colour = arrow_ctx.colour
    current_block_arrow_length = arrow_ctx.length
    current_block_arrow_ratio = arrow_ctx.ratio
    current_block_arrow_tilt = 0.0

    for line_str in step_content.splitlines():
        stripped = line_str.lstrip()
        line_type = int(stripped[0]) if stripped and stripped[0].isdigit() else -1

        if line_type == 0:
            tokens = line_str.upper().split()
            is_py_arrow_cmd = "!PY" in tokens and "ARROW" in tokens

            if is_py_arrow_cmd:
                if "BEGIN" in tokens:
                    in_arrow_meta_block = True
                    current_block_offsets = []
                    coord_tokens = [
                        t
                        for t in tokens
                        if t
                        not in {
                            "!PY",
                            "ARROW",
                            "BEGIN",
                            "COLOUR",
                            "LENGTH",
                            "RATIO",
                            "TILT",
                        }
                        and not t.isalpha()
                    ]
                    idx = 0
                    while idx + 2 < len(coord_tokens):
                        v = vectorize_arrow(coord_tokens[idx : idx + 3])
                        if v:
                            current_block_offsets.append(v)
                        idx += 3
                    arrow_ctx.offset = current_block_offsets
                    current_block_arrow_colour = value_after_token(
                        tokens, "COLOUR", arrow_ctx.colour, int
                    )
                    current_block_arrow_length = value_after_token(
                        tokens, "LENGTH", arrow_ctx.length, int
                    )
                    current_block_arrow_ratio = value_after_token(
                        tokens, "RATIO", arrow_ctx.ratio, float
                    )
                    current_block_arrow_tilt = value_after_token(
                        tokens, "TILT", 0.0, float
                    )
                    arrow_ctx.colour = (
                        current_block_arrow_colour  # Update context if on BEGIN line
                    )
                    arrow_ctx.length = current_block_arrow_length
                elif "END" in tokens and in_arrow_meta_block:
                    in_arrow_meta_block = False
                elif not in_arrow_meta_block:
                    arrow_ctx.colour = value_after_token(
                        tokens, "COLOUR", arrow_ctx.colour, int
                    )
                    arrow_ctx.length = value_after_token(
                        tokens, "LENGTH", arrow_ctx.length, int
                    )
                    arrow_ctx.ratio = value_after_token(
                        tokens, "RATIO", arrow_ctx.ratio, float
                    )
            if not as_lpub or (as_lpub and not is_py_arrow_cmd):
                if not only_arrows:
                    processed_lines.append(line_str)
        elif line_type == 1:
            if in_arrow_meta_block:
                original_part_lines_from_arrow_blocks.append(line_str)
                data_norm = arrow_ctx.dict_for_line(
                    line_str,
                    False,
                    current_block_arrow_ratio,
                    current_block_arrow_colour,
                    current_block_arrow_tilt,
                )
                data_inv = arrow_ctx.dict_for_line(
                    line_str,
                    True,
                    current_block_arrow_ratio,
                    current_block_arrow_colour,
                    current_block_arrow_tilt,
                )
                arrow_data_collected.extend([data_norm, data_inv])
            if not only_arrows and not as_lpub:
                processed_lines.append(line_str)
        else:
            if not only_arrows:
                processed_lines.append(line_str)

    if as_dict:
        return arrow_data_collected

    if as_lpub:
        lpub_result_lines: List[str] = []
        if original_part_lines_from_arrow_blocks or arrow_data_collected:
            lpub_result_lines.append(ARROW_PREFIX)
            unique_lpub_parts = set()
            for arrow_instr_dict in arrow_data_collected:
                if arrow_instr_dict["line"] not in unique_lpub_parts:
                    part_obj = LDRPart()
                    if part_obj.from_str(arrow_instr_dict["line"]):
                        if arrow_instr_dict["offset"]:
                            first_offset = arrow_instr_dict["offset"][0]
                            mask = arrow_ctx._mask_axis(arrow_instr_dict["offset"])
                            part_offset_for_bufexchg = arrow_ctx.part_loc_for_offset(
                                first_offset, mask
                            )
                            part_obj.attrib.loc += part_offset_for_bufexchg
                        lpub_result_lines.append(str(part_obj).strip())
                        unique_lpub_parts.add(arrow_instr_dict["line"])
            lpub_result_lines.append(ARROW_PLI)
            for arrow_instr_dict in arrow_data_collected:
                arrow_graphic_str = arrow_ctx.arrow_from_dict(arrow_instr_dict)
                if arrow_graphic_str:
                    lpub_result_lines.append(arrow_graphic_str.strip())
            lpub_result_lines.append(ARROW_SUFFIX)
            lpub_result_lines.append(ARROW_PLI)
            for part_line in original_part_lines_from_arrow_blocks:
                lpub_result_lines.append(part_line.strip())
            lpub_result_lines.append(ARROW_PLI_SUFFIX)
            return "\n".join(lpub_result_lines) + "\n" if lpub_result_lines else ""
        else:
            return "\n".join(processed_lines) + "\n" if processed_lines else ""

    final_non_lpub_lines: List[str] = []
    if not only_arrows:
        final_non_lpub_lines.extend(processed_lines)
    for arrow_instr_dict in arrow_data_collected:
        arrow_graphic_str = arrow_ctx.arrow_from_dict(arrow_instr_dict)
        if arrow_graphic_str:
            final_non_lpub_lines.append(arrow_graphic_str.strip())
    return "\n".join(final_non_lpub_lines) + "\n" if final_non_lpub_lines else ""


def arrows_for_lpub_file(filename: str, outfile: str):
    arrow_ctx = ArrowContext()
    try:
        with open(filename, "rt", encoding="utf-8") as fp_in:
            content = fp_in.read()
    except FileNotFoundError:
        print(f"Error: Input file {filename} not found.")
        return

    output_parts: List[str] = []
    file_blocks = content.split("0 FILE")
    first_block_is_model_content = not content.strip().startswith("0 FILE") and bool(
        file_blocks
    )

    current_block_index = 0
    if first_block_is_model_content:
        # Process file_blocks[0] as the first model content block
        steps_in_block = file_blocks[0].split("0 STEP")
        for i, step_text in enumerate(steps_in_block):
            if i == 0 and not step_text.strip() and len(steps_in_block) > 1:
                continue
            processed_step = arrows_for_step(arrow_ctx, step_text, as_lpub=True)
            if processed_step.strip():
                output_parts.append(processed_step)
        current_block_index = 1

    for i in range(current_block_index, len(file_blocks)):
        block_text_with_header = file_blocks[i]
        if not block_text_with_header.strip():
            continue

        # Reconstruct the "0 FILE ..." header for this block
        full_block_text = "0 FILE " + block_text_with_header.strip()
        block_lines = full_block_text.splitlines()
        if not block_lines:
            continue

        output_parts.append(block_lines[0])  # Add "0 FILE <name>"
        content_after_header = "\n".join(block_lines[1:])

        steps_in_block = content_after_header.split("0 STEP")
        for j, step_text in enumerate(steps_in_block):
            if j == 0 and not step_text.strip() and len(steps_in_block) > 1:
                continue
            # If it's not the very first step of this "0 FILE" block, it was preceded by "0 STEP"
            # The `arrows_for_step` with as_lpub=True handles internal "0 STEP" for arrow sequences.
            # We need to ensure "0 STEP" is added *between* outputs of `arrows_for_step` if they
            # don't already provide it.
            # A simple way: join non-empty processed steps with "0 STEP\n"
            processed_step = arrows_for_step(arrow_ctx, step_text, as_lpub=True)
            if processed_step.strip():
                # If this is not the first step *within this current FILE block*
                # and the processed_step doesn't already start with "0 STEP" (which it might if it's an arrow block)
                # This logic is complex due to ARROW_SUFFIX containing "0 STEP".
                # For now, let `arrows_for_step` manage its own "0 STEP" via ARROW_SUFFIX.
                # We will join the results of `arrows_for_step` calls.
                # If a step is not an arrow step, it won't have "0 STEP" from ARROW_SUFFIX.
                if j > 0 and not processed_step.strip().startswith(
                    "0 BUFEXCHG A STORE"
                ):  # Crude check if it's not an arrow block
                    output_parts.append(
                        "0 STEP"
                    )  # Add delimiter if it's a normal step after the first
                output_parts.append(processed_step)

    final_output_str = "\n".join(filter(None, [s.strip() for s in output_parts]))
    if final_output_str and not final_output_str.endswith("\n"):
        final_output_str += "\n"

    try:
        with open(outfile, "w", encoding="utf-8") as fpo:
            fpo.write(final_output_str)
    except Exception as e:
        print(f"Error writing output file {outfile}: {e}")


def remove_offset_parts(
    parts: List[Union[LDRPart, str]],
    oparts: List[Union[LDRPart, str]],
    arrow_dict_list: List[Dict[str, Any]],
    as_str: bool = False,
) -> Union[List[LDRPart], List[str]]:
    pp_objs: List[LDRPart] = []
    for item in parts:
        if isinstance(item, LDRPart):
            pp_objs.append(item)
        elif isinstance(item, str):
            p_obj = LDRPart()
            if p_obj.from_str(item):
                pp_objs.append(p_obj)
    op_objs: List[LDRPart] = []
    for item in oparts:
        if isinstance(item, LDRPart):
            op_objs.append(item)
        elif isinstance(item, str):
            p_obj = LDRPart()
            if p_obj.from_str(item):
                op_objs.append(p_obj)

    arrow_part_names: set[str] = set()
    arrow_offsets_world: List[Vector] = []
    for arrow_instr_dict in arrow_dict_list:
        offsets_in_instr = arrow_instr_dict.get("offset", [])
        if isinstance(offsets_in_instr, list) and all(
            isinstance(v, Vector) for v in offsets_in_instr
        ):
            arrow_offsets_world.extend(offsets_in_instr)
        arrow_graphic_str = arrow_instr_dict.get("arrow")
        if isinstance(arrow_graphic_str, str):
            temp_arrow_graphic_part = LDRPart()
            if temp_arrow_graphic_part.from_str(arrow_graphic_str):
                arrow_part_names.add(temp_arrow_graphic_part.name)

    kept_parts: List[LDRPart] = []
    for p_candidate in pp_objs:
        if p_candidate.name in arrow_part_names:
            kept_parts.append(p_candidate)
            continue
        is_an_offset_version = False
        for o_ref_part in op_objs:
            if not (
                p_candidate.name == o_ref_part.name
                and p_candidate.attrib.colour == o_ref_part.attrib.colour
            ):
                continue
            p_loc = p_candidate.attrib.loc
            o_loc = o_ref_part.attrib.loc
            for offset_v in arrow_offsets_world:
                if p_loc.almost_same_as(
                    o_loc + offset_v, tol=0.1
                ) or o_loc.almost_same_as(
                    p_loc + offset_v, tol=0.1
                ):  # type: ignore
                    is_an_offset_version = True
                    break
            if is_an_offset_version:
                break
        if not is_an_offset_version:
            kept_parts.append(p_candidate)

    if as_str:
        return [str(p) for p in kept_parts]
    return kept_parts
