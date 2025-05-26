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
ldrawpy.py: Main LDraw utility functions for the ldrawpy package.

This module provides core functionalities such as coordinate system transformations
for LDraw compatibility, conversion of mesh data (vertices and faces) into
LDraw format, and a utility for abbreviating LDraw part descriptions.
"""
from typing import List, Tuple, Union, Optional, Sequence

from toolbox import Vector  # type: ignore

# Assuming ldrawpy is installed or in PYTHONPATH for relative imports
from .constants import LDR_DEF_COLOUR, LDR_OPT_COLOUR
from .ldrprimitives import LDRTriangle, LDRLine

# Type alias for a 3D point, which can be a tuple of floats, list of floats, or a toolbox.Vector
Point3D = Union[Tuple[float, float, float], List[float], Vector]  # type: ignore
# Type alias for a 3D point specifically as a tuple of floats
Point3DTuple = Tuple[float, float, float]


def xyz_to_ldr(
    point: Point3D, as_tuple: bool = False
) -> Union[Vector, Point3DTuple]:  # type: ignore
    """
    Converts a 3D coordinate from a typical Cartesian system (where Y is up,
    and Z is depth) to the LDraw file format's coordinate system.

    The LDraw coordinate system is typically:
    - X-axis: Right
    - Y-axis: Up (negative in file for "down" on screen)
    - Z-axis: Towards viewer (negative in file for "into screen")

    This function specifically maps input (x, y_up, z_depth) to
    LDraw file coordinates (x, -z_depth, y_up).
    This means:
    - Input X maps to LDraw X.
    - Input Y (up) maps to LDraw Z.
    - Input Z (depth) maps to LDraw -Y.

    Args:
        point: The 3D coordinate to convert. Can be a 3-element tuple/list
               of floats, or a toolbox.Vector.
        as_tuple: If True, returns the LDraw coordinate as a 3-tuple of floats.
                  If False (default), returns a toolbox.Vector.

    Returns:
        The coordinate converted to the LDraw system, either as a toolbox.Vector
        or a 3-tuple of floats.

    Raises:
        TypeError: If the input point is not a 3-element tuple/list or a Vector,
                   or if tuple/list elements are not numeric.
    """
    v_internal_ldr: Vector  # type: ignore

    if isinstance(point, Vector):  # type: ignore
        # LDraw: X_ldr = X_in, Y_ldr = -Z_in, Z_ldr = Y_in
        v_internal_ldr = Vector(point.x, -point.z, point.y)  # type: ignore
    elif isinstance(point, (tuple, list)) and len(point) == 3:
        # Ensure elements are numbers before attempting conversion
        if not all(isinstance(coord, (int, float)) for coord in point):
            raise TypeError("Input point tuple/list elements must be numeric.")
        # LDraw: X_ldr = X_in, Y_ldr = -Z_in, Z_ldr = Y_in
        v_internal_ldr = Vector(float(point[0]), -float(point[2]), float(point[1]))  # type: ignore
    else:
        raise TypeError(
            "Input point must be a 3-element tuple/list or a toolbox.Vector object."
        )

    return v_internal_ldr.as_tuple() if as_tuple else v_internal_ldr  # type: ignore


def mesh_to_ldr(
    faces: Sequence[Tuple[int, int, int]],  # Vertex indices for each triangular face
    vertices: Sequence[Point3D],  # List of 3D points (vertices)
    mesh_colour: int = LDR_DEF_COLOUR,  # LDraw colour for the mesh faces
    edges: Optional[
        Sequence[Tuple[Point3D, Point3D]]
    ] = None,  # Optional list of edge vertex pairs
    edge_colour: Optional[
        int
    ] = None,  # LDraw colour for edges (defaults to LDR_OPT_COLOUR if edges are provided)
) -> str:
    """
    Converts a triangular mesh, defined by faces and vertices, into an LDraw formatted string.
    Optionally includes edge lines.

    The input vertices are assumed to be in a coordinate system that `xyz_to_ldr`
    can correctly convert to the LDraw standard (typically X-right, Y-up, Z-depth).
    The generated LDraw primitives will use "mm" as their units, based on the
    original context of this function.

    Args:
        faces: A sequence of 3-integer tuples. Each tuple defines a triangular face
               by listing the 0-based indices of its vertices in the `vertices` list.
        vertices: A sequence of 3D points (each can be a tuple, list, or Vector)
                  representing the mesh vertices.
        mesh_colour: The LDraw colour code for the faces (triangles).
                     Defaults to LDR_DEF_COLOUR.
        edges: An optional sequence of tuples, where each inner tuple contains two 3D points
               defining an edge. If provided, LDraw lines will be generated for these edges.
        edge_colour: The LDraw colour code for the edges. If `edges` are provided and
                     `edge_colour` is None, it defaults to LDR_OPT_COLOUR.

    Returns:
        A string containing LDraw type 3 lines (triangles) and optionally
        type 2 lines (edges).

    Raises:
        TypeError: If vertex or edge point data is not in the expected format.
    """
    ldr_lines_output_list: List[str] = []

    # 1. Convert all input vertices to LDraw coordinate system (as toolbox.Vector objects)
    ldr_coordinate_vertices: List[Vector] = []  # type: ignore
    for original_vertex_point in vertices:
        # xyz_to_ldr converts to LDraw's X, -Z, Y system and returns a Vector
        converted_ldr_vertex_union = xyz_to_ldr(original_vertex_point, as_tuple=False)
        if isinstance(converted_ldr_vertex_union, Vector):  # type: ignore
            ldr_coordinate_vertices.append(converted_ldr_vertex_union)
        else:
            # This case should ideally not be hit if xyz_to_ldr works as expected
            raise TypeError(
                f"xyz_to_ldr did not return a Vector as expected for vertex: {original_vertex_point}"
            )

    # 2. Create LDraw triangles (type 3 lines) from faces and converted vertices
    for face_vertex_indices in faces:
        if len(face_vertex_indices) == 3:  # Ensure it's a triangle
            # Validate vertex indices against the length of the ldr_coordinate_vertices list
            if not all(
                0 <= idx < len(ldr_coordinate_vertices) for idx in face_vertex_indices
            ):
                # Consider logging a warning or skipping if desired, instead of just continuing
                # print(f"Warning: Invalid face index in {face_vertex_indices} for {len(ldr_coordinate_vertices)} vertices.")
                continue  # Skip this invalid face

            triangle_primitive = LDRTriangle(
                mesh_colour, units="mm"
            )  # Assume "mm" based on original context
            triangle_primitive.p1 = ldr_coordinate_vertices[face_vertex_indices[0]]
            triangle_primitive.p2 = ldr_coordinate_vertices[face_vertex_indices[1]]
            triangle_primitive.p3 = ldr_coordinate_vertices[face_vertex_indices[2]]
            ldr_lines_output_list.append(str(triangle_primitive))
        # else: Could add handling for non-triangular faces (e.g., skip with warning, or triangulate)

    # 3. Create LDraw lines (type 2 lines) for edges if provided
    if edges is not None:
        # Determine the colour for edges: use provided edge_colour, or default to LDR_OPT_COLOUR
        final_edge_colour: int = (
            edge_colour if edge_colour is not None else LDR_OPT_COLOUR
        )
        for edge_point_pair_tuple in edges:
            if len(edge_point_pair_tuple) == 2:  # Ensure it's a pair of points
                edge_line_primitive = LDRLine(
                    final_edge_colour, units="mm"
                )  # Assume "mm"

                original_p1_for_edge, original_p2_for_edge = edge_point_pair_tuple

                # Convert edge points to LDraw coordinates
                ldr_p1_for_edge_union = xyz_to_ldr(original_p1_for_edge, as_tuple=False)
                ldr_p2_for_edge_union = xyz_to_ldr(original_p2_for_edge, as_tuple=False)

                if isinstance(ldr_p1_for_edge_union, Vector) and isinstance(ldr_p2_for_edge_union, Vector):  # type: ignore
                    edge_line_primitive.p1 = ldr_p1_for_edge_union
                    edge_line_primitive.p2 = ldr_p2_for_edge_union
                    ldr_lines_output_list.append(str(edge_line_primitive))
                else:
                    # print(f"Warning: Could not convert edge points to Vectors: {original_p1_for_edge}, {original_p2_for_edge}")
                    pass  # Or raise an error if strict conversion is required
            # else: Could add handling for malformed edges

    return "".join(
        ldr_lines_output_list
    )  # Join all generated LDraw lines into a single string


def brick_name_strip(s: str, level: int = 0) -> str:
    """
    Progressively strips and abbreviates an LDraw part description string
    based on the specified stripping `level`. Higher levels apply more
    aggressive abbreviations.

    This is useful for generating concise labels or Bill of Materials (BOM)
    entries where space might be limited.

    Args:
        s: The input part description string.
        level: The stripping level (0-6). Higher levels apply more substitutions.
               Level 0 applies the most common and general substitutions.

    Returns:
        The stripped and abbreviated part description string.
    """
    stripped_name: str = s  # Start with the original string

    # Level 0 replacements (most common/general, e.g., standardizing jumper plate names)
    if level >= 0:  # Apply level 0 for all levels >= 0
        replacements0 = {
            "  ": " ",  # Consolidate double spaces first (idempotent if run multiple times)
            "Plate 1 x 2 with Groove with 1 Centre Stud, without Understud": "Plate 1 x 2 Jumper",
            "Plate 1 x 2 without Groove with 1 Centre Stud": "Plate 1 x 2 Jumper",
            "Plate 1 x 2 with Groove with 1 Centre Stud": "Plate 1 x 2 Jumper",
            "Brick 1 x 1 with Headlight": "Brick 1 x 1 Erling",  # Common alternative name
            "with Groove": "",  # Often redundant
            "Bluish ": "Bl ",  # Abbreviation
            "Slope Brick": "Slope",  # Common shortening
            "0.667": "2/3",
            "1.667": "1-2/3",
            "1.333": "1-1/3",
            "1 And 1/3": "1-1/3",
            "1 and 1/3": "1-1/3",
            "1 & 1/3": "1-1/3",
            "with Headlight": "Erling",  # "Erling brick"
            "Angle Connector": "Conn",
            "~Plate": "Plate",  # Remove tilde prefix sometimes used for unofficial parts
        }
        for old, new in replacements0.items():
            stripped_name = stripped_name.replace(old, new)

    # Level 1 replacements (more abbreviations for common terms)
    if level >= 1:
        replacements1 = {
            "with ": "w/",
            "With ": "w/",
            "without ": "wo/",
            "Without ": "wo/",
            "Ribbed": "ribbed",  # Consistent casing
            "One": "1",
            "Two": "2",
            "Three": "3 ",
            "Four": "4",  # Digits for numbers
            " and ": " & ",
            " And ": " & ",
            "Dark": "Dk",
            "Light": "Lt",
            "Bright": "Br",
            "Reddish Brown": "Rd Brown",
            "Reddish": "Rd",
            "Yellowish": "Ylwish",
            "Medium": "Med",
            "Offset": "offs",
            "Adjacent": "adj",
            " degree": "°",  # Degree symbol
        }
        for old, new in replacements1.items():
            stripped_name = stripped_name.replace(old, new)

    # Level 2 replacements (shorter common words)
    if level >= 2:
        replacements2 = {
            "Trans": "Tr",
            " x ": "x",
            "Bl ": " ",
        }  # "Bl " from Bluish, remove if already abbreviated
        for old, new in replacements2.items():
            stripped_name = stripped_name.replace(old, new)

    # Level 3 replacements (more color and common part term abbreviations)
    if level >= 3:
        replacements_l3 = {
            "Orange": "Org",
            "Yellow": "Ylw",
            "Black": "Blk",
            "White": "Wht",
            "Green": "Grn",
            "Brown": "Brn",
            "Purple": "Prpl",
            "Violet": "Vlt",
            "Gray": "Gry",
            "Grey": "Gry",  # Alternate spellings
            "Axlehole": "axle",
            "Cylinder": "Cyl",
            "cylinder": "cyl",
            "Inverted": "Inv",
            "inverted": "inv",
            "Centre": "Ctr",
            "centre": "ctr",
            "Center": "Ctr",
            "center": "ctr",
            "Figure": "Fig",
            "figure": "fig",
            "Rounded": "Round",
            "rounded": "round",
            "Underside": "under",
            "Vertical": "vert",
            "Horizontal": "horz",
            "vertical": "vert",
            "horizontal": "horz",
            "Flex-System": "Flex",
            "Flanges": "Flange",
            "Joiner": "joiner",
            "Joint": "joint",
            "Type 1": "",
            "Type 2": "",  # Remove type indicators if not critical
        }
        for old, new in replacements_l3.items():
            stripped_name = stripped_name.replace(old, new)

    # Level 4 replacements (Technic, stud configurations)
    if level >= 4:
        replacements_l4 = {
            "Technic": "",  # Often implied by part type
            "Single": "1",
            "Dual": "2",
            "Double": "Dbl",
            "Stud on": "stud",
            "Studs on Sides": "stud sides",
            "Studs on Side": "side studs",
            "Hinge Plate": "Hinge",
        }
        for old, new in replacements_l4.items():
            stripped_name = stripped_name.replace(old, new)

    # Level 5 replacements (very short abbreviations, remove filler words)
    if level >= 5:
        replacements_l5 = {
            " on ": " ",
            " On ": " ",  # Remove "on"
            "Round": "Rnd",
            "round": "rnd",
            "Side": "Sd",
            "Groove": "Grv",
            "Minifig": "",  # Often implied or too specific for high abbreviation
            "Curved": "Curv",
            "curved": "curv",
            "Notched": "notch",
            "Friction": "fric",
            "(Complete)": "",  # Remove "(Complete)" assembly indicators
            "Cross": "X",
            "Embossed": "Emb",
            "Extension": "Ext",
            "Bottom": "Bot",
            "bottom": "bot",
            "Inside": "Insd",
            "inside": "insd",
            "Locking": "click",
            "Axleholder": "axle",
            "axleholder": "axle",
            "End": "end",
            "Open": "open",
            "Rod": "rod",
            "Hole": "hole",
            "Ball": "ball",
            "Thin": "thin",
            "Thick": "thick",
            " - ": "-",  # Consolidate spaced hyphen
        }
        for old, new in replacements_l5.items():
            stripped_name = stripped_name.replace(old, new)

    # Level 6 replacements (most aggressive abbreviations)
    if level >= 6:
        replacements_l6 = {
            "Up": "up",
            "Down": "dn",
            "Bot": "bot",
            "Rnd": "rnd",
            "Studs": "St",
            "studs": "St",
            "Stud": "St",
            "stud": "St",
            "Corners": "edge",
            "w/Curv Top": "curved",  # Specific common pattern
            "Domed": "dome",
            "Clip": "clip",
            "Clips": "clip",
            "Convex": "cvx",
        }
        for old, new in replacements_l6.items():
            stripped_name = stripped_name.replace(old, new)

    # Final pass to consolidate multiple spaces to single spaces
    stripped_name = " ".join(stripped_name.split())
    # Capitalize the first letter of the resulting string if not empty
    if stripped_name:
        stripped_name = stripped_name[0].upper() + stripped_name[1:]

    return stripped_name
