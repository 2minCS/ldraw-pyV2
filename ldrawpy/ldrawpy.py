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
from typing import List, Tuple, Any, Union, Optional

from toolbox import Vector  # type: ignore

from .constants import LDR_DEF_COLOUR, LDR_OPT_COLOUR
from .ldrprimitives import LDRTriangle, LDRLine


def xyz_to_ldr(
    point: Union[Tuple[float, float, float], List[float], Vector],
    as_tuple: bool = False,
) -> Union[Vector, Tuple[float, float, float]]:

    v_internal: Vector
    if isinstance(point, (tuple, list)) and len(point) == 3:
        v_internal = Vector(point[0], -point[2], point[1])
    elif isinstance(point, Vector):
        v_internal = Vector(point.x, -point.z, point.y)
    else:
        raise TypeError(
            "Input point must be a 3-element tuple/list or a Vector object."
        )

    return v_internal.as_tuple() if as_tuple else v_internal  # type: ignore


def mesh_to_ldr(
    faces: List[Tuple[int, int, int]],
    vertices: List[Union[Tuple[float, float, float], List[float], Vector]],
    mesh_colour: int = LDR_DEF_COLOUR,
    edges: Optional[
        List[
            Tuple[
                Union[Tuple[float, float, float], List[float], Vector],
                Union[Tuple[float, float, float], List[float], Vector],
            ]
        ]
    ] = None,
    edge_colour: Optional[int] = None,
) -> str:
    s_list: List[str] = []

    ldr_vertices: List[Vector] = []
    for v_orig in vertices:
        ldr_v_union = xyz_to_ldr(v_orig)
        if isinstance(ldr_v_union, Vector):
            ldr_vertices.append(ldr_v_union)
        else:
            raise TypeError(
                "xyz_to_ldr did not return a Vector as expected when as_tuple=False."
            )

    for face_indices in faces:
        if len(face_indices) == 3:
            tri = LDRTriangle(mesh_colour, "mm")
            tri.p1 = ldr_vertices[face_indices[0]]
            tri.p2 = ldr_vertices[face_indices[1]]
            tri.p3 = ldr_vertices[face_indices[2]]
            s_list.append(str(tri))

    if edges is not None:
        actual_edge_colour = edge_colour if edge_colour is not None else LDR_OPT_COLOUR
        for edge_points in edges:
            if len(edge_points) == 2:
                line = LDRLine(actual_edge_colour, "mm")
                p1_orig, p2_orig = edge_points

                ldr_p1_union = xyz_to_ldr(p1_orig)
                ldr_p2_union = xyz_to_ldr(p2_orig)

                if isinstance(ldr_p1_union, Vector) and isinstance(
                    ldr_p2_union, Vector
                ):
                    line.p1 = ldr_p1_union
                    line.p2 = ldr_p2_union
                    s_list.append(str(line))
                else:
                    pass  # Or raise error

    return "".join(s_list)


def brick_name_strip(s: str, level: int = 0) -> str:
    sn = s
    if level == 0:
        sn = sn.replace("  ", " ")
        sn = sn.replace(
            "Plate 1 x 2 with Groove with 1 Centre Stud, without Understud",
            "Plate 1 x 2  Jumper",
        )
        sn = sn.replace(
            "Plate 1 x 2 without Groove with 1 Centre Stud", "Plate 1 x 2  Jumper"
        )
        sn = sn.replace(
            "Plate 1 x 2 with Groove with 1 Centre Stud", "Plate 1 x 2  Jumper"
        )
        sn = sn.replace("Brick 1 x 1 with Headlight", "Brick 1 x 1 Erling")
        sn = sn.replace("with Groove", "")
        sn = sn.replace("Bluish ", "Bl ")
        sn = sn.replace("Slope Brick", "Slope")
        sn = (
            sn.replace("0.667", "2/3")
            .replace("1.667", "1-2/3")
            .replace("1.333", "1-1/3")
        )
        sn = (
            sn.replace("1 And 1/3", "1-1/3")
            .replace("1 and 1/3", "1-1/3")
            .replace("1 & 1/3", "1-1/3")
        )
        sn = sn.replace("with Headlight", "Erling")
        sn = sn.replace("Angle Connector", "Conn")
        sn = sn.replace("~Plate", "Plate")
    elif level == 1:
        sn = sn.replace("with ", "w/").replace("With ", "w/")
        sn = sn.replace("Ribbed", "ribbed")
        sn = sn.replace("without ", "wo/").replace("Without ", "wo/")
        sn = (
            sn.replace("One", "1")
            .replace("Two", "2")
            .replace("Three", "3 ")
            .replace("Four", "4")
        )
        sn = sn.replace(" and ", " & ").replace(" And ", " & ")
        sn = sn.replace("Dark", "Dk").replace("Light", "Lt").replace("Bright", "Br")
        sn = (
            sn.replace("Reddish Brown", "Red Brown")
            .replace("Reddish", "Red")
            .replace("Yellowish", "Ylwish")
        )
        sn = (
            sn.replace("Medium", "Med")
            .replace("Offset", "offs")
            .replace("Adjacent", "adj")
        )
        sn = sn.replace(" degree", "°")
    elif level == 2:
        sn = sn.replace("Trans", "Tr").replace(" x ", "x").replace("Bl ", " ")
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
    elif level == 4:
        sn = sn.replace("Technic", "").replace("Single", "1").replace("Dual", "2")
        sn = sn.replace("Double", "Dbl").replace("Stud on", "stud")
        sn = sn.replace("Studs on Sides", "stud sides").replace(
            "Studs on Side", "side studs"
        )
        sn = sn.replace("Hinge Plate", "Hinge")
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
    if sn:
        sn = sn[0].upper() + sn[1:]
    return sn
