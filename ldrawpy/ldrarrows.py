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

# import os # REMOVED - Unused
from math import sin, cos, pi

# from functools import reduce # REMOVED - Unused
from typing import List, Tuple, Union, Optional, Any, Dict, Set

from toolbox import Vector, Matrix, Identity, ZAxis, safe_vector  # type: ignore
from .ldrprimitives import (
    LDRPart,
)  # Assuming LDRPart might be needed for context or future use.

# Constants defining LDraw comment lines for LPUB arrow generation
ARROW_PREFIX = """0 BUFEXCHG A STORE"""
ARROW_PLI = (
    """0 !LPUB PLI BEGIN IGN"""  # Start Parts List Item, ignore for normal rendering
)
ARROW_SUFFIX = """0 !LPUB PLI END
0 STEP
0 BUFEXCHG A RETRIEVE"""
ARROW_PLI_SUFFIX = """0 !LPUB PLI END"""

# Standard LDraw part names for arrowheads of different lengths
ARROW_PARTS = [
    "hashl2.dat",
    "hashl3.dat",
    "hashl4.dat",
    "hashl5.dat",
    "hashl6.dat",
]  # Added .dat extension

# Predefined rotation matrices for aligning arrows to axes
ARROW_MZ = Matrix(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
)  # Arrow points along -Z in its local space
ARROW_PZ = Matrix(
    [[0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
)  # Arrow points along +Z (corrected from original comment if it was different)
# The file provided had: ARROW_PZ = Matrix([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
# This seems to map local Y to world -X, local X to world -Y, local Z to world -Z.
# For an arrow part typically modeled along +X or +Y, this would need careful checking.
# Let's assume the file's ARROW_PZ is what's intended for now:
# ARROW_PZ = Matrix([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]) # type: ignore

ARROW_MX = Matrix(
    [[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)  # Arrow points along -X
ARROW_PX = Matrix(
    [[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
)  # Arrow points along +X (corrected from original comment if different)
# File: ARROW_PX = Matrix([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]) # type: ignore

ARROW_MY = Matrix(
    [[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]]
)  # Arrow points along -Y
ARROW_PY = Matrix(
    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]
)  # Arrow points along +Y (corrected from original comment if different)
# File: ARROW_PY = Matrix([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]) # type: ignore


def value_after_token(
    tokens: List[str], value_token: str, default_val: Any, xtype: type = int
) -> Any:
    """
    Finds a specific token in a list of string tokens and returns the next token,
    converted to the specified type (xtype).
    If the token is not found, or the next token doesn't exist, or conversion fails,
    it returns default_val.
    """
    try:
        idx = tokens.index(value_token)  # Find the index of the target token
        if idx + 1 < len(tokens):  # Check if there is a token after it
            return xtype(tokens[idx + 1])  # Convert and return the next token
    except (ValueError, TypeError):
        # ValueError if token not in list, TypeError if xtype conversion fails
        pass
    return default_val


def norm_angle_arrow(angle: float) -> float:
    """
    Normalizes an angle, likely for arrow rotation steps.
    The original function `return angle % 45.0` would return the remainder
    when divided by 45. This might be intended for snapping rotations to
    multiples of 45 degrees if used in a specific way, or it might be
    a different kind of normalization. Preserving original behavior.
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
            # Create a Vector by converting each string coordinate to a float
            return Vector(*(float(x) for x in s_coords))  # type: ignore
        except ValueError:
            # Handle cases where a string coordinate cannot be converted to float
            return None
    return None  # Return None if the input list doesn't have exactly 3 coordinates


class ArrowContext:
    """
    Holds context and settings for generating LDraw arrows, such as default
    colour, length, scale, and current processing state for arrow commands.
    """

    colour: int  # Default LDraw color code for arrows
    length: int  # Default length category for arrows (e.g., 2 for hashl2)
    scale: float  # General scale factor for arrow part placement/size
    yscale: float  # Specific scale factor for Y-dimension adjustments
    offset: List[Vector]  # type: ignore # List of offset vectors for current arrow command
    rotstep: Vector  # type: ignore # Rotation step (seems unused in current methods)
    ratio: float  # Ratio for positioning arrow head relative to its length/offset
    outline_colour: int  # LDraw color code for arrow outlines (if applicable, not directly used in part generation here)

    def __init__(self, colour: int = 804, length: int = 2):  # 804 is often an arrow red
        self.colour, self.length = colour, length
        self.scale, self.yscale, self.ratio = (
            25.0,
            20.0,
            0.5,
        )  # Default scale/ratio values
        self.offset = []  # Initialize as empty list of Vectors
        self.rotstep = Vector(0, 0, 0)  # type: ignore # Initialize Vector
        self.outline_colour = 804  # Default outline colour (e.g. LDR_ARROW_RED)

    def part_for_length(self, length_val: int) -> str:
        """Selects an LDraw arrow part name (e.g., "hashl2.dat") based on desired length category."""
        # Ensure .dat extension is included for LDRPart name
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
        return "hashl5.dat"  # Fallback for unhandled lengths

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
        base_rotation_matrix = Identity()  # type: ignore # Default to no rotation

        # Determine dominant axis for arrow direction, excluding masked axes.
        # The arrow part is typically modeled pointing along its own local +X or +Y.
        # These matrices orient that local axis along the world axis.
        if "x" not in axis_mask and abs_x > max(
            abs_y, abs_z, 1e-9
        ):  # Dominant X world-axis offset
            base_rotation_matrix = (
                (
                    ARROW_PX if offset_vector.x < 0 else ARROW_MX
                )  # Point along world -X or +X
                if not invert_direction
                else (ARROW_MX if offset_vector.x < 0 else ARROW_PX)  # Inverted
            )
        elif "y" not in axis_mask and abs_y > max(
            abs_x, abs_z, 1e-9
        ):  # Dominant Y world-axis offset
            base_rotation_matrix = (
                (
                    ARROW_MY if offset_vector.y < 0 else ARROW_PY
                )  # Point along world -Y or +Y
                if not invert_direction
                else (ARROW_PY if offset_vector.y < 0 else ARROW_MY)  # Inverted
            )
        elif "z" not in axis_mask and abs_z > max(
            abs_x, abs_y, 1e-9
        ):  # Dominant Z world-axis offset
            base_rotation_matrix = (
                (
                    ARROW_PZ if offset_vector.z < 0 else ARROW_MZ
                )  # Point along world -Z or +Z
                if not invert_direction
                else (ARROW_MZ if offset_vector.z < 0 else ARROW_PZ)  # Inverted
            )

        # Apply tilt if specified (rotation around the arrow's local Z-axis after primary orientation)
        # ZAxis should be toolbox.Vector(0,0,1) or similar for local Z.
        return base_rotation_matrix.rotate(tilt_angle, ZAxis) if tilt_angle != 0.0 and ZAxis else base_rotation_matrix  # type: ignore

    def loc_for_offset(self, offset_vector: Vector, arrow_length_category: int, axis_mask: str = "", position_ratio: float = 0.5) -> Vector:  # type: ignore
        """
        Calculates the location for an arrow head based on the target offset_vector,
        arrow_length_category, axis_mask, and position_ratio.
        The arrow head is typically placed near the midpoint of the offset component and
        then pulled back slightly so the arrow points towards the center of that component.
        """
        arrow_loc_vector = Vector(0, 0, 0)  # type: ignore

        # Scaled length components used for adjusting the arrow head's position.
        # position_ratio (0 to 1) determines how far along the length this adjustment occurs.
        # 0.5 means the adjustment point is halfway along the arrow's effective length.
        scaled_len_x_adj = float(arrow_length_category) * position_ratio * self.scale
        scaled_len_y_adj = float(arrow_length_category) * position_ratio * self.yscale
        scaled_len_z_adj = float(arrow_length_category) * position_ratio * self.scale

        # Adjust location for non-masked axes:
        # Start at half the offset component, then subtract the scaled length adjustment.
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

        # For masked axes, the arrow is placed directly at the full offset component.
        # This means the arrow does not visually span the offset along masked axes.
        if "x" in axis_mask:
            arrow_loc_vector.x += offset_vector.x
        if "y" in axis_mask:
            arrow_loc_vector.y += offset_vector.y
        if "z" in axis_mask:
            arrow_loc_vector.z += offset_vector.z
        return arrow_loc_vector

    def part_loc_for_offset(self, offset_vector: Vector, axis_mask: str = "") -> Vector:  # type: ignore
        """
        Determines the effective location of the part being pointed to by the arrow,
        considering the offset_vector and axis_mask.
        Essentially, it returns the components of offset_vector for the non-masked axes.
        The arrow originates relative to this "part location".
        """
        part_effective_location = Vector(0, 0, 0)  # type: ignore
        if "x" not in axis_mask:  # If X is a primary direction of offset
            part_effective_location.x = offset_vector.x
        if "y" not in axis_mask:  # If Y is a primary direction of offset
            part_effective_location.y = offset_vector.y
        if "z" not in axis_mask:  # If Z is a primary direction of offset
            part_effective_location.z = offset_vector.z
        # Components on masked axes are zero, meaning the arrow base is at origin for these axes.
        return part_effective_location

    def _mask_axis(self, offset_vectors_list: List[Vector]) -> str:  # type: ignore
        """
        Analyzes a list of offset vectors to determine which axes are "masked".
        An axis is considered masked (i.e., included in the returned string 'xyz')
        if the coordinate values along that axis *vary* among the provided offset vectors.
        If coordinates are constant along an axis, that axis character is *not* in the mask string.
        This is used to decide arrow orientation for multi-offset arrows (e.g., push-in-and-down).
        The arrow should primarily align with axes that *do not* vary.
        """
        if not offset_vectors_list or len(offset_vectors_list) <= 1:
            # If no offsets or only one, no basis for masking by comparison; treat all axes as variable.
            return "xyz"  # Or "" depending on desired default for single offset. Original returned "".
            # Let's stick to "" for single offset, meaning no axes are "constant by variance".

        # Find min and max for each coordinate across all offset vectors
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
        tolerance = 1e-6  # Tolerance for float comparison
        # If there's significant variation along an axis, include it in the mask string.
        if abs(max_coords.x - min_coords.x) > tolerance:
            mask_str += "x"
        if abs(max_coords.y - min_coords.y) > tolerance:
            mask_str += "y"
        if abs(max_coords.z - min_coords.z) > tolerance:
            mask_str += "z"
        # The returned mask_str contains axes where coordinates *vary*.
        # The matrix_for_offset and loc_for_offset use `if "x" not in mask_str`
        # to mean "X is a primary (non-varying, or constant relative to other offsets) direction".
        # This seems to be the intended logic: the mask identifies axes of *movement/spread* of the offsets.
        # The arrow should align with axes *not* in this mask_str (i.e., constant axes).
        # Example: if offsets are (10,0,0), (20,0,0) -> mask="x". Arrow shouldn't be along X.
        # If offsets are (10,5,0), (10,15,0) -> mask="y". Arrow shouldn't be along Y.
        # If offsets are (10,0,0), (10,0,0) -> mask="". Arrow can be along any axis (defaults).
        # The original code's `_mask_axis` returned axes that *were* constant.
        # Let's correct this to match the usage:
        # The mask should identify axes that are *constant* or *shared* among the offsets.
        # The arrow will then typically align along one of these constant axes if possible,
        # or an axis where the offset is significant if all axes vary.

        # REVISED LOGIC for _mask_axis:
        # The mask string should identify axes that are *fixed* or *shared* across the offsets.
        # The arrow generation logic then uses `if 'x' not in mask` to mean X is a direction of motion.
        # So, if X is fixed, 'x' should be IN the mask.
        # If X varies, 'x' should NOT be in the mask.
        # This is the opposite of what I wrote above. Let's re-verify the original code's intent.

        # Original code:
        # mc = Vector(min(o.x for o in ol), min(o.y for o in ol), min(o.z for o in ol))
        # Xc = Vector(max(o.x for o in ol), max(o.y for o in ol), max(o.z for o in ol))
        # m = ""
        # if abs(Xc.x - mc.x) > tol: m += "x" -> if X varies, add 'x' to m
        # This `m` is then passed as `mask`.
        # matrix_for_offset uses `if "x" not in mask`. So if X varies (is in `m`), this condition is false.
        # This means if X varies, it's *not* chosen as a primary arrow direction.
        # This seems correct: if X coordinates are all different, the arrow shouldn't try to align solely along X.
        # The arrow should align along an axis where the offset component is significant AND that axis is *not* part of this "variance mask".

        return mask_str  # mask_str contains axes along which the offsets *vary*

    def arrow_from_dict(self, arrow_data_dict: Dict[str, Any]) -> str:
        """Generates LDraw string for arrows based on a dictionary of parameters."""
        arrows_str_list: List[str] = []
        offset_vectors_from_dict: List[Vector] = arrow_data_dict.get("offset", [])  # type: ignore

        # Ensure offset_vectors_from_dict is a list of Vector objects
        if not (
            isinstance(offset_vectors_from_dict, list)
            and all(isinstance(v, Vector) for v in offset_vectors_from_dict)
        ):  # type: ignore
            # print(f"Warning: Invalid 'offset' data in arrow_data_dict: {offset_vectors_from_dict}")
            return ""

        # Determine the axis mask based on the variation in the provided offset vectors
        axis_variation_mask = self._mask_axis(offset_vectors_from_dict)

        for single_offset_vec in offset_vectors_from_dict:
            # Create an LDRPart for the original part line (to get its base location)
            original_part_ref = LDRPart()
            if original_part_ref.from_str(arrow_data_dict["line"]) is None:
                # print(f"Warning: Could not parse original part line: {arrow_data_dict['line']}")
                continue  # Skip if the original part line is invalid

            # Create the arrow LDRPart
            arrow_part_obj = LDRPart()
            arrow_part_obj.name = self.part_for_length(arrow_data_dict["length"])

            # Calculate arrow base offset: location of arrow relative to the original part's location
            arrow_base_offset_loc = self.loc_for_offset(
                single_offset_vec,
                arrow_data_dict["length"],
                axis_variation_mask,  # Mask of axes where offsets vary
                position_ratio=arrow_data_dict["ratio"],
            )
            # Final arrow location: original part location + calculated arrow base offset
            arrow_part_obj.attrib.loc = original_part_ref.attrib.loc + arrow_base_offset_loc  # type: ignore

            # Determine arrow rotation matrix
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
    ) -> Dict[str, Any]:
        """Creates a dictionary of parameters for generating an arrow for a given LDraw line."""
        return {
            "line": ldraw_line_str,  # The LDraw line of the part the arrow relates to
            "colour": (
                arrow_colour_override
                if arrow_colour_override is not None
                else self.colour
            ),
            "length": self.length,  # Uses current context's default length
            "offset": copy.deepcopy(self.offset),  # Current list of offset vectors
            "invert": invert_arrow_direction,
            "ratio": arrow_pos_ratio,
            "tilt": arrow_tilt_angle,
        }


def arrows_for_step(
    arrow_ctx: ArrowContext,  # Current arrow generation settings/context
    step_content_str: str,  # Raw LDraw string content for the current step
    generate_lpub_meta: bool = True,  # If True, wrap arrows in LPUB meta commands
    output_only_arrows: bool = False,  # If True, only arrow LDraw lines are returned
    return_as_dict_list: bool = False,  # If True, return list of dicts instead of LDraw string
) -> Union[str, List[Dict[str, Any]]]:
    """
    Processes an LDraw step's content to generate arrow annotations based on
    !PY ARROW meta commands found within the step.
    """
    processed_lines_for_output: List[str] = []  # Stores non-arrow lines or final output
    arrow_data_to_generate: List[Dict[str, Any]] = (
        []
    )  # List of dicts, each defining an arrow
    # Stores original LDraw part lines that were targets of !PY ARROW commands (for LPUB PLI)
    original_part_lines_from_arrow_blocks: List[str] = []

    is_inside_arrow_meta_block = (
        False  # True when between "!PY ARROW BEGIN" and "!PY ARROW END"
    )
    current_block_offset_vectors: List[Vector] = []  # type: ignore # Offsets defined by current ARROW BEGIN

    # Temporary storage for parameters from current "!PY ARROW BEGIN" block
    current_block_effective_arrow_colour = arrow_ctx.colour
    current_block_effective_arrow_length = arrow_ctx.length
    current_block_effective_arrow_ratio = arrow_ctx.ratio
    current_block_effective_arrow_tilt = 0.0

    for line_str_from_step in step_content_str.splitlines():
        stripped_line_content = line_str_from_step.lstrip()
        # Determine line type (0 for meta, 1 for part, etc.)
        line_type_char = stripped_line_content[0] if stripped_line_content else ""

        if line_type_char == "0":  # Meta command line
            tokens_in_line = (
                line_str_from_step.upper().split()
            )  # Process tokens in uppercase
            is_py_arrow_command = "!PY" in tokens_in_line and "ARROW" in tokens_in_line

            if is_py_arrow_command:
                if "BEGIN" in tokens_in_line:  # Start of an arrow definition block
                    is_inside_arrow_meta_block = True
                    current_block_offset_vectors = []  # Reset for this new block

                    # Extract coordinate values for offsets from the BEGIN command
                    # Exclude command keywords, then try to parse remaining tokens as coords
                    coord_candidate_tokens = [
                        tkn
                        for tkn in tokens_in_line  # Use original case for parsing values
                        if tkn.upper()
                        not in {  # Compare with uppercase keywords
                            "0",
                            "!PY",
                            "ARROW",
                            "BEGIN",
                            "COLOUR",
                            "LENGTH",
                            "RATIO",
                            "TILT",
                        }
                        and not tkn.upper().isalpha()  # Filter out any other stray alpha tokens
                    ]

                    idx = 0
                    while idx + 2 < len(
                        coord_candidate_tokens
                    ):  # Need 3 tokens for a vector
                        vec = vectorize_arrow(coord_candidate_tokens[idx : idx + 3])
                        if vec:
                            current_block_offset_vectors.append(vec)
                        idx += 3
                    arrow_ctx.offset = (
                        current_block_offset_vectors  # Update context for this block
                    )

                    # Parse optional COLOUR, LENGTH, RATIO, TILT from BEGIN command
                    # Use original case tokens for value_after_token
                    original_case_tokens = line_str_from_step.split()
                    current_block_effective_arrow_colour = value_after_token(
                        original_case_tokens,
                        "COLOUR",
                        arrow_ctx.colour,
                        int,  # Case-sensitive "COLOUR"
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
                    # Update context's current settings from this block's parameters
                    arrow_ctx.colour = current_block_effective_arrow_colour
                    arrow_ctx.length = current_block_effective_arrow_length
                    # arrow_ctx.ratio and arrow_ctx.tilt are not instance variables,
                    # they are passed directly to dict_for_line.

                elif "END" in tokens_in_line and is_inside_arrow_meta_block:
                    is_inside_arrow_meta_block = False  # End of arrow definition block
                    # Reset context's offset, or it will persist for subsequent individual arrows
                    arrow_ctx.offset = []

                elif (
                    not is_inside_arrow_meta_block
                ):  # Individual !PY ARROW command (not BEGIN/END)
                    # Update context defaults if specified on this line
                    original_case_tokens = line_str_from_step.split()
                    arrow_ctx.colour = value_after_token(
                        original_case_tokens, "COLOUR", arrow_ctx.colour, int
                    )
                    arrow_ctx.length = value_after_token(
                        original_case_tokens, "LENGTH", arrow_ctx.length, int
                    )
                    # Ratio and Tilt are not typically set on standalone !PY ARROW lines,
                    # but if they were, this is where they'd be parsed for arrow_ctx update.
                    # However, the primary use of these on standalone lines is less common.

            # Add non-!PY ARROW meta lines to output if not outputting only arrows
            # And if generating LPUB, skip !PY ARROW lines as they are processed into other meta
            if not output_only_arrows:
                if not (generate_lpub_meta and is_py_arrow_command):
                    processed_lines_for_output.append(line_str_from_step)

        elif line_type_char == "1":  # LDraw Part line
            if (
                is_inside_arrow_meta_block
            ):  # This part is a target for the current ARROW BEGIN block
                original_part_lines_from_arrow_blocks.append(line_str_from_step)
                # Create arrow data for normal and inverted directions
                # Use the parameters parsed from the current "BEGIN" block
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

            # Add part line to output if not outputting only arrows AND
            # (if generating LPUB, it's added later within the PLI block if it was an arrow target)
            if not output_only_arrows:
                if not (generate_lpub_meta and is_inside_arrow_meta_block):
                    processed_lines_for_output.append(line_str_from_step)

        elif (
            not output_only_arrows
        ):  # Other line types (2,3,4,5, comments not starting with 0)
            processed_lines_for_output.append(line_str_from_step)

    # --- Output Generation ---
    if return_as_dict_list:
        return (
            arrow_data_to_generate  # Return the list of arrow definition dictionaries
        )

    if generate_lpub_meta:
        lpub_output_lines: List[str] = []
        # Add existing non-arrow related lines first if any (e.g. step headers, other metas)
        # Filter out any !PY ARROW commands from processed_lines_for_output as they are handled by LPUB metas
        for ln in processed_lines_for_output:
            if not ("!PY" in ln.upper() and "ARROW" in ln.upper()):
                lpub_output_lines.append(ln)

        if original_part_lines_from_arrow_blocks or arrow_data_to_generate:
            lpub_output_lines.append(ARROW_PREFIX)
            # Add transformed original parts (that arrows point to) into BUFEXCHG A STORE
            unique_part_lines_processed: Set[str] = set()
            for arrow_dict_item in arrow_data_to_generate:
                original_line_str = arrow_dict_item["line"]
                if original_line_str not in unique_part_lines_processed:
                    part_obj_original = LDRPart()
                    if part_obj_original.from_str(original_line_str):
                        # If there are offsets, transform the part to the first offset's effective location
                        # This is for the PLI view of the part itself.
                        if arrow_dict_item["offset"]:  # offset is List[Vector]
                            first_offset_vec = arrow_dict_item["offset"][0]
                            # Determine mask based on *all* offsets for this arrow group
                            axis_var_mask = arrow_ctx._mask_axis(
                                arrow_dict_item["offset"]
                            )
                            part_eff_loc = arrow_ctx.part_loc_for_offset(
                                first_offset_vec, axis_var_mask
                            )
                            part_obj_original.attrib.loc += part_eff_loc  # type: ignore
                        lpub_output_lines.append(str(part_obj_original).strip())
                        unique_part_lines_processed.add(original_line_str)

            lpub_output_lines.append(ARROW_PLI)  # Start PLI for arrows
            for arrow_dict_item in arrow_data_to_generate:
                arrow_ldr_str = arrow_ctx.arrow_from_dict(arrow_dict_item)
                if arrow_ldr_str:
                    lpub_output_lines.append(arrow_ldr_str.strip())
            lpub_output_lines.append(
                ARROW_SUFFIX
            )  # End PLI, 0 STEP, BUFEXCHG A RETRIEVE

            # Add the original parts again, but this time for the main step view (after RETRIEVE)
            # These are the parts as they appear in the model, not offset for PLI.
            lpub_output_lines.append(
                ARROW_PLI
            )  # Start another PLI for the actual parts in step
            for orig_part_line in original_part_lines_from_arrow_blocks:
                lpub_output_lines.append(orig_part_line.strip())
            lpub_output_lines.append(ARROW_PLI_SUFFIX)  # End this PLI

        # Join all lines for the LPUB output string
        return "\n".join(lpub_output_lines) + ("\n" if lpub_output_lines else "")

    else:  # Not generating LPUB meta, just direct LDraw output
        final_direct_output_lines: List[str] = []
        if not output_only_arrows:
            # Add all non-!PY ARROW lines
            final_direct_output_lines.extend(
                ln
                for ln in processed_lines_for_output
                if not ("!PY" in ln.upper() and "ARROW" in ln.upper())
            )
        # Add generated arrow LDraw lines
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
    arrow_context_global = (
        ArrowContext()
    )  # Create a single context for the file processing
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

    # Split the file into blocks based on "0 FILE" directive (for MPD handling)
    file_content_blocks = full_file_content.split("0 FILE")

    # The first block needs special handling: if the file doesn't start with "0 FILE",
    # file_content_blocks[0] is the first model. Otherwise, file_content_blocks[0] is empty.
    first_block_is_model_content = not full_file_content.strip().startswith(
        "0 FILE"
    ) and bool(file_content_blocks)

    start_block_processing_idx = 0
    if first_block_is_model_content:
        # Process the first block (which is a model itself)
        steps_in_first_block = file_content_blocks[0].split("0 STEP")
        for i, step_text_content in enumerate(steps_in_first_block):
            # Skip empty initial part if "0 STEP" was at the very beginning of this block
            if (
                i == 0
                and not step_text_content.strip()
                and len(steps_in_first_block) > 1
            ):
                continue

            # Process this step for arrows, generating LPUB meta
            processed_step_output_str_fb = arrows_for_step(
                arrow_context_global, step_text_content, generate_lpub_meta=True
            )
            # Ensure it's a string, as generate_lpub_meta=True should return str
            assert isinstance(
                processed_step_output_str_fb, str
            ), f"arrows_for_step(generate_lpub_meta=True) expected str, got {type(processed_step_output_str_fb)}"

            # Add "0 STEP" prefix if it's not the first part of the model and output is not empty
            # and output doesn't already start with a known prefix like ARROW_PREFIX or "0 STEP"
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

            if processed_step_output_str_fb.strip():  # Only add if there's content
                output_all_final_string_blocks.append(
                    step_prefix_fb + processed_step_output_str_fb
                )
        start_block_processing_idx = (
            1  # Start processing subsequent "0 FILE" blocks from index 1
        )

    # Process remaining blocks (each starting with a submodel name after "0 FILE")
    for i in range(start_block_processing_idx, len(file_content_blocks)):
        current_file_block_text_with_header = file_content_blocks[i]
        if not current_file_block_text_with_header.strip():  # Skip empty blocks
            continue

        # The block text starts with the model name, e.g., "submodel.ldr ...rest of model..."
        # Prepend "0 FILE " to make it a parsable unit if it's not already there (it shouldn't be after split)
        current_model_full_str_content = (
            "0 FILE " + current_file_block_text_with_header.strip()
        )

        model_lines = current_model_full_str_content.splitlines()
        if not model_lines:  # Should not happen if strip check passed
            continue

        output_all_final_string_blocks.append(
            model_lines[0]
        )  # Add the "0 FILE submodel.ldr" line

        content_for_step_splitting = "\n".join(
            model_lines[1:]
        )  # Content after the "0 FILE" line
        steps_content_in_this_block = content_for_step_splitting.split("0 STEP")

        for j, step_text_content_sub in enumerate(steps_content_in_this_block):
            if (
                j == 0
                and not step_text_content_sub.strip()
                and len(steps_content_in_this_block) > 1
            ):
                continue  # Skip empty initial part of sub-block

            processed_step_output_str_sub = arrows_for_step(
                arrow_context_global, step_text_content_sub, generate_lpub_meta=True
            )
            assert isinstance(
                processed_step_output_str_sub, str
            ), f"arrows_for_step(generate_lpub_meta=True) expected str, got {type(processed_step_output_str_sub)}"

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

    # Join all processed blocks and lines into the final output string
    final_output_file_content = "\n".join(
        s_block.strip()
        for s_block in output_all_final_string_blocks
        if isinstance(s_block, str) and s_block.strip()
    )
    if final_output_file_content and not final_output_file_content.endswith("\n"):
        final_output_file_content += "\n"  # Ensure trailing newline

    try:
        with open(output_filename, "w", encoding="utf-8") as fpo_handle:
            fpo_handle.write(final_output_file_content)
    except Exception as e:
        print(f"Error writing output file '{output_filename}': {e}")


def remove_offset_parts(
    parts_list_main: List[Union[LDRPart, str]],  # Main list of parts
    parts_list_original_positions: List[
        Union[LDRPart, str]
    ],  # Reference list of parts at original positions
    arrow_definitions_list: List[
        Dict[str, Any]
    ],  # List of arrow definition dictionaries
    return_as_strings: bool = False,  # If True, return list of strings, else list of LDRPart
) -> Union[List[LDRPart], List[str]]:  # Corrected return type hint
    """
    Removes parts from parts_list_main that are considered "offset versions"
    of parts in parts_list_original_positions, based on arrow definitions.
    This is used to avoid duplicating parts in views where arrows show movement.
    Parts that are actual arrow graphics are kept.
    """
    # Convert input lists to LDRPart objects for consistent processing
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

    # Collect names of arrow graphic parts (e.g., "hashl2.dat")
    arrow_graphic_part_names: Set[str] = set()
    # Collect all unique offset vectors defined by the arrows
    all_arrow_offset_vectors_world: List[Vector] = []  # type: ignore

    for arrow_dict_item_def in arrow_definitions_list:
        if not isinstance(arrow_dict_item_def, dict):
            continue

        # Add offset vectors from this arrow definition
        offsets_in_this_arrow_def = arrow_dict_item_def.get("offset", [])
        if isinstance(offsets_in_this_arrow_def, list) and all(
            isinstance(v, Vector) for v in offsets_in_this_arrow_def
        ):  # type: ignore
            all_arrow_offset_vectors_world.extend(offsets_in_this_arrow_def)  # type: ignore

        # Get the LDraw string for the arrow graphic itself
        # This part of the original logic was a bit indirect.
        # The goal is to identify if a part in main_parts_as_objects *is* an arrow graphic.
        # The arrow_ctx.arrow_from_dict(arrow_dict_item_def) generates the arrow LDraw string.
        # We need the *name* of the arrow part (e.g. hashl2.dat)
        # This should come from arrow_ctx.part_for_length(arrow_dict_item_def['length'])
        # Let's assume ArrowContext is available or reconstruct this part.
        # For simplicity, if an ArrowContext instance was used to generate these dicts,
        # we'd ideally have access to its `part_for_length` method.
        # If not, we might need to infer from ARROW_PARTS or assume a naming convention.
        # The current `arrow_definitions_list` doesn't directly store the arrow part *name*.
        # It stores the `line` of the *target* part.
        # This function's purpose is to remove *target parts* that appear offset,
        # not to remove the arrow graphics themselves.
        # However, the original code had `if pc.name in arrow_part_names: kept_parts.append(pc)`.
        # This implies arrow_part_names should contain names like "hashl2.dat".
        # Let's assume ARROW_PARTS contains the names of arrow graphics.
        arrow_graphic_part_names.update(ARROW_PARTS)

    kept_parts_final: List[LDRPart] = []
    for part_candidate_to_keep in main_parts_as_objects:
        # Always keep parts that are themselves arrow graphics
        if part_candidate_to_keep.name in arrow_graphic_part_names:
            kept_parts_final.append(part_candidate_to_keep)
            continue

        is_this_an_offset_version_to_remove = False
        # Compare with each part in its original (non-offset) position
        for original_pos_part_ref in original_parts_as_objects:
            # Check if it's the same part type and color
            if not (
                part_candidate_to_keep.name == original_pos_part_ref.name
                and part_candidate_to_keep.attrib.colour
                == original_pos_part_ref.attrib.colour
            ):
                continue  # Not the same basic part, so not an offset version of *this* original_pos_part_ref

            # Check if the candidate part's location matches the original part's location
            # plus any of the defined arrow offset vectors.
            candidate_loc = part_candidate_to_keep.attrib.loc
            original_loc = original_pos_part_ref.attrib.loc

            for offset_vec_world in all_arrow_offset_vectors_world:
                # Is candidate_loc == original_loc + offset_vec_world?
                if candidate_loc.almost_same_as(original_loc + offset_vec_world, 0.1):  # type: ignore
                    is_this_an_offset_version_to_remove = True
                    break
                # Also check the reverse, though less common for this function's purpose
                # (Is original_loc == candidate_loc + offset_vec_world?)
                # if original_loc.almost_same_as(candidate_loc + offset_vec_world, 0.1):
                #     is_this_an_offset_version_to_remove = True
                #     break
            if is_this_an_offset_version_to_remove:
                break  # Found it's an offset of an original part, no need to check other originals

        if not is_this_an_offset_version_to_remove:
            kept_parts_final.append(part_candidate_to_keep)

    return [str(p) for p in kept_parts_final] if return_as_strings else kept_parts_final
