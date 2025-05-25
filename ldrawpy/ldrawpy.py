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
# LDraw python module
from typing import List, Tuple, Union, Optional, Sequence  # Added Sequence, Optional

from toolbox import Vector  # type: ignore

from .constants import LDR_DEF_COLOUR, LDR_OPT_COLOUR
from .ldrprimitives import LDRTriangle, LDRLine  # LDRPart is not directly used here

# Type alias for a 3D point representation
Point3D = Union[Tuple[float, float, float], List[float], Vector]  # type: ignore
Point3DTuple = Tuple[float, float, float]


def xyz_to_ldr(
    point: Point3D, as_tuple: bool = False
) -> Union[Vector, Point3DTuple]:  # type: ignore
    """
    Converts a typical x,y,z 3D coordinate to the LDraw file format's
    coordinate system (often X, -Z, Y if input Y is up, or X, Y, -Z if input Z is up).
    The original implementation maps (x, y_up, z_depth) to LDraw file (x, -z_depth, y_up).
    """

    v_internal: Vector  # type: ignore
    if isinstance(point, Vector):  # type: ignore
        v_internal = Vector(point.x, -point.z, point.y)  # type: ignore
    elif isinstance(point, (tuple, list)) and len(point) == 3:
        # Ensure elements are numbers
        if not all(isinstance(coord, (int, float)) for coord in point):
            raise TypeError("Point tuple/list elements must be numbers.")
        v_internal = Vector(float(point[0]), -float(point[2]), float(point[1]))  # type: ignore
    else:
        raise TypeError(
            "Input point must be a 3-element tuple/list or a Vector object."
        )

    return v_internal.as_tuple() if as_tuple else v_internal  # type: ignore


def mesh_to_ldr(
    faces: Sequence[
        Tuple[int, int, int]
    ],  # Sequence of 3-integer tuples (vertex indices)
    vertices: Sequence[Point3D],  # Sequence of 3D points
    mesh_colour: int = LDR_DEF_COLOUR,
    edges: Optional[Sequence[Tuple[Point3D, Point3D]]] = None,  # Sequence of edge pairs
    edge_colour: Optional[int] = None,
) -> str:
    """
    Converts a triangular mesh (faces and vertices) into an LDraw formatted string.
    Optionally includes edge lines.
    Assumes vertices are in a coordinate system that xyz_to_ldr correctly converts.
    Units for generated LDraw primitives are assumed to be "mm" based on original.
    """
    s_list: List[str] = []

    # Convert all input vertices to LDraw coordinate system (as Vectors)
    ldr_vertices: List[Vector] = []  # type: ignore
    for v_orig in vertices:
        ldr_v_union = xyz_to_ldr(v_orig, as_tuple=False)  # Ensure Vector is returned
        if isinstance(ldr_v_union, Vector):  # type: ignore
            ldr_vertices.append(ldr_v_union)
        else:
            # This should not happen if as_tuple=False and xyz_to_ldr is correct
            raise TypeError(
                f"xyz_to_ldr did not return a Vector as expected for vertex: {v_orig}"
            )

    # Create LDraw triangles
    for face_indices in faces:
        if len(face_indices) == 3:
            # Ensure indices are valid for ldr_vertices list
            if not all(0 <= idx < len(ldr_vertices) for idx in face_indices):
                # print(f"Warning: Invalid face index in {face_indices} for {len(ldr_vertices)} vertices.")
                continue  # Skip invalid face

            tri = LDRTriangle(mesh_colour, units="mm")  # Specify units
            tri.p1 = ldr_vertices[face_indices[0]]
            tri.p2 = ldr_vertices[face_indices[1]]
            tri.p3 = ldr_vertices[face_indices[2]]
            s_list.append(str(tri))
        # Else: skip non-triangular faces or handle error/warning

    # Create LDraw lines for edges if provided
    if edges is not None:
        actual_edge_colour: int = (
            edge_colour if edge_colour is not None else LDR_OPT_COLOUR
        )
        for edge_points_tuple in edges:
            if len(edge_points_tuple) == 2:
                line = LDRLine(actual_edge_colour, units="mm")  # Specify units
                p1_orig, p2_orig = edge_points_tuple

                ldr_p1_union = xyz_to_ldr(p1_orig, as_tuple=False)
                ldr_p2_union = xyz_to_ldr(p2_orig, as_tuple=False)

                if isinstance(ldr_p1_union, Vector) and isinstance(ldr_p2_union, Vector):  # type: ignore
                    line.p1 = ldr_p1_union
                    line.p2 = ldr_p2_union
                    s_list.append(str(line))
                else:
                    # print(f"Warning: Could not convert edge points to Vectors: {p1_orig}, {p2_orig}")
                    pass  # Or raise error
            # Else: skip malformed edges or handle error/warning

    return "".join(s_list)


def brick_name_strip(s: str, level: int = 0) -> str:
    """
    Progressively strips (with increasing levels) a part description
    by making substitutions with abbreviations, removing spaces, etc.
    This can be useful for labelling or BOM part lists where space is limited.
    """
    sn: str = s  # Ensure sn is typed as str

    # Level 0 replacements (most common/general)
    if level == 0:
        replacements0 = {
            "  ": " ",  # Consolidate double spaces first
            "Plate 1 x 2 with Groove with 1 Centre Stud, without Understud": "Plate 1 x 2  Jumper",
            "Plate 1 x 2 without Groove with 1 Centre Stud": "Plate 1 x 2  Jumper",
            "Plate 1 x 2 with Groove with 1 Centre Stud": "Plate 1 x 2  Jumper",
            "Brick 1 x 1 with Headlight": "Brick 1 x 1 Erling",
            "with Groove": "",
            "Bluish ": "Bl ",
            "Slope Brick": "Slope",
            "0.667": "2/3",
            "1.667": "1-2/3",
            "1.333": "1-1/3",
            "1 And 1/3": "1-1/3",
            "1 and 1/3": "1-1/3",
            "1 & 1/3": "1-1/3",
            "with Headlight": "Erling",  # Erling brick
            "Angle Connector": "Conn",
            "~Plate": "Plate",  # Remove tilde prefix if used for unofficial parts
        }
        for old, new in replacements0.items():
            sn = sn.replace(old, new)

    # Level 1 replacements (more abbreviations)
    elif level == 1:
        replacements1 = {
            "with ": "w/",
            "With ": "w/",
            "without ": "wo/",
            "Without ": "wo/",
            "Ribbed": "ribbed",
            "One": "1",
            "Two": "2",
            "Three": "3 ",
            "Four": "4",
            " and ": " & ",
            " And ": " & ",
            "Dark": "Dk",
            "Light": "Lt",
            "Bright": "Br",
            "Reddish Brown": "Rd Brown",
            "Reddish": "Rd",
            "Yellowish": "Ylwish",  # Shortened further
            "Medium": "Med",
            "Offset": "offs",
            "Adjacent": "adj",
            " degree": "°",
        }
        for old, new in replacements1.items():
            sn = sn.replace(old, new)

    # Level 2 replacements
    elif level == 2:
        replacements2 = {"Trans": "Tr", " x ": "x", "Bl ": " "}  # "Bl " from Bluish
        for old, new in replacements2.items():
            sn = sn.replace(old, new)

    # Level 3 replacements
    elif level == 3:
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
            "Grey": "Gry",
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
            "Type 2": "",
        }
        for old, new in replacements_l3.items():
            sn = sn.replace(old, new)

    # Level 4 replacements
    elif level == 4:
        replacements_l4 = {
            "Technic": "",
            "Single": "1",
            "Dual": "2",
            "Double": "Dbl",
            "Stud on": "stud",
            "Studs on Sides": "stud sides",
            "Studs on Side": "side studs",
            "Hinge Plate": "Hinge",
        }
        for old, new in replacements_l4.items():
            sn = sn.replace(old, new)

    # Level 5 replacements
    elif level == 5:
        replacements_l5 = {
            " on ": " ",
            " On ": " ",
            "Round": "Rnd",
            "round": "rnd",
            "Side": "Sd",
            "Groove": "Grv",
            "Minifig": "",
            "Curved": "Curv",
            "curved": "curv",
            "Notched": "notch",
            "Friction": "fric",
            "(Complete)": "",
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
            " - ": "-",
        }
        for old, new in replacements_l5.items():
            sn = sn.replace(old, new)

    # Level 6 replacements
    elif level == 6:
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
            "w/Curv Top": "curved",
            "Domed": "dome",
            "Clip": "clip",
            "Clips": "clip",
            "Convex": "cvx",
        }
        for old, new in replacements_l6.items():
            sn = sn.replace(old, new)

    # Final pass to consolidate spaces and capitalize first letter
    sn = " ".join(sn.split())  # Consolidate multiple spaces to single
    if sn:
        sn = sn[0].upper() + sn[1:]
    return sn
