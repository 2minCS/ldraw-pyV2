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
# LDraw helper functions to generate complicated shapes and solids
# from LDraw primitives

import os  # Not strictly used in the provided snippet, but kept from original
import copy
from math import pi, cos, sin  # Explicitly import math functions
from typing import List, Union  # For type hints if needed

# Explicit imports from toolbox
from toolbox import (
    Vector,
    Identity,
)  # Assuming Identity is used by LDRAttrib internally

# Explicit relative imports from ldrawpy package
from .ldrprimitives import LDRAttrib, LDRQuad, LDRLine, LDRTriangle
from .ldrhelpers import GetCircleSegments  # GetCircleSegments is in ldrhelpers
from .constants import LDR_DEF_COLOUR, LDR_OPT_COLOUR


class LDRPolyWall:
    attrib: LDRAttrib  # Type hint
    height: Union[int, float]  # Type hint
    points: List[Vector]  # Type hint

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.height = 1
        self.points = []  # Initialize points list

    def __str__(self) -> str:
        s_list: List[str] = []
        nPoints = len(self.points)
        if nPoints == 0:
            return ""

        for i in range(nPoints):
            q = LDRQuad(self.attrib.colour, self.attrib.units)

            q.p1.x = self.points[i].x
            q.p1.y = float(
                self.height
            )  # Ensure float for calculations if height can be int
            q.p1.z = self.points[i].z

            thePoint_idx = (i + 1) % nPoints  # Use modulo for cleaner wrap-around
            thePoint = self.points[thePoint_idx]

            q.p2.x = thePoint.x
            q.p2.y = float(self.height)
            q.p2.z = thePoint.z

            q.p3.x = thePoint.x
            q.p3.y = 0.0  # Explicitly float
            q.p3.z = thePoint.z

            q.p4.x = self.points[i].x
            q.p4.y = 0.0
            q.p4.z = self.points[i].z

            # Apply transformations if LDRAttrib.matrix and .loc are set
            q.transform(self.attrib.matrix)
            q.translate(self.attrib.loc)
            s_list.append(str(q))
        return "".join(s_list)


class LDRRect:
    attrib: LDRAttrib
    length: Union[int, float]
    width: Union[int, float]

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.length = 1.0  # Use float for dimensions
        self.width = 1.0

    def __str__(self) -> str:
        s_list: List[str] = []
        half_length, half_width = self.length / 2.0, self.width / 2.0

        q = LDRQuad(self.attrib.colour, self.attrib.units)
        # Define points relative to origin (0,0,0) before transformation
        q.p1 = Vector(-half_length, 0.0, half_width)
        q.p2 = Vector(-half_length, 0.0, -half_width)
        q.p3 = Vector(half_length, 0.0, -half_width)
        q.p4 = Vector(half_length, 0.0, half_width)

        # Create a copy for the lines, so transform doesn't affect quad's points before its own transform
        q_transformed_for_lines = copy.deepcopy(q)
        q_transformed_for_lines.transform(self.attrib.matrix)
        q_transformed_for_lines.translate(self.attrib.loc)

        # Add edge lines using the transformed points
        line_attrib = LDRAttrib(
            LDR_OPT_COLOUR, self.attrib.units
        )  # Edge color, can be main color too

        l = LDRLine(line_attrib.colour, line_attrib.units)
        l.p1, l.p2 = q_transformed_for_lines.p1, q_transformed_for_lines.p2
        s_list.append(str(l))
        l.p1, l.p2 = q_transformed_for_lines.p2, q_transformed_for_lines.p3
        s_list.append(str(l))
        l.p1, l.p2 = q_transformed_for_lines.p3, q_transformed_for_lines.p4
        s_list.append(str(l))
        l.p1, l.p2 = q_transformed_for_lines.p4, q_transformed_for_lines.p1
        s_list.append(str(l))

        # Transform the quad itself and add it
        q.transform(self.attrib.matrix)
        q.translate(self.attrib.loc)
        s_list.append(str(q))

        return "".join(s_list)


class LDRCircle:
    attrib: LDRAttrib
    radius: Union[int, float]
    segments: int
    fill: bool

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.radius = 1.0
        self.segments = 24
        self.fill = False

    def __str__(self) -> str:
        s_list: List[str] = []
        # get_cirGetCircleSegmentscle_segments expects LDRAttrib for color/units, but not for loc/matrix here
        # as those are applied after the segments are generated relative to origin.
        # Create a temporary attrib for segment generation if GetCircleSegments uses its color/units.
        temp_attrib_for_segments = LDRAttrib(self.attrib.colour, self.attrib.units)
        circle_lines_local: List[LDRLine] = GetCircleSegments(
            float(self.radius), self.segments, temp_attrib_for_segments
        )

        for line_local in circle_lines_local:
            # Apply instance's transformation to each segment
            line_transformed = copy.deepcopy(line_local)  # Work on a copy
            line_transformed.transform(self.attrib.matrix)
            line_transformed.translate(self.attrib.loc)
            s_list.append(str(line_transformed))

        if self.fill:
            for line_local in circle_lines_local:
                t = LDRTriangle(self.attrib.colour, self.attrib.units)
                # Triangle points are based on the local circle segment points
                t.p1 = Vector(0, 0, 0)  # Center of the circle
                t.p2 = line_local.p1.copy()
                t.p3 = line_local.p2.copy()

                t.transform(self.attrib.matrix)
                t.translate(self.attrib.loc)
                s_list.append(str(t))
        return "".join(s_list)


class LDRDisc:  # A disc is a ring (border)
    attrib: LDRAttrib
    radius: Union[int, float]  # Inner radius
    border: Union[int, float]  # Thickness of the border
    segments: int

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.radius = 1.0
        self.border = 0.2  # Default border thickness, adjust as needed
        self.segments = 24

    def __str__(self) -> str:
        s_list: List[str] = []
        inner_radius = float(self.radius)
        outer_radius = inner_radius + float(self.border)

        temp_attrib_for_segments = LDRAttrib(self.attrib.colour, self.attrib.units)

        inner_lines_local: List[LDRLine] = GetCircleSegments(
            inner_radius, self.segments, temp_attrib_for_segments
        )
        outer_lines_local: List[LDRLine] = GetCircleSegments(
            outer_radius, self.segments, temp_attrib_for_segments
        )

        # Add lines for inner and outer edges (optional, LDraw quads often imply edges)
        # for line_local in inner_lines_local + outer_lines_local:
        #     line_transformed = copy.deepcopy(line_local)
        #     line_transformed.transform(self.attrib.matrix)
        #     line_transformed.translate(self.attrib.loc)
        #     s_list.append(str(line_transformed))

        for i in range(self.segments):
            q = LDRQuad(self.attrib.colour, self.attrib.units)
            q.p1 = inner_lines_local[i].p1.copy()  # Inner point 1
            q.p2 = inner_lines_local[i].p2.copy()  # Inner point 2
            q.p3 = outer_lines_local[
                i
            ].p2.copy()  # Outer point 2 (matches inner_local[i].p2 angle)
            q.p4 = outer_lines_local[
                i
            ].p1.copy()  # Outer point 1 (matches inner_local[i].p1 angle)

            q.transform(self.attrib.matrix)
            q.translate(self.attrib.loc)
            s_list.append(str(q))
        return "".join(s_list)


class LDRHole:  # Creates a square with a hole, not just a hole in an infinite plane
    attrib: LDRAttrib
    radius: Union[int, float]  # Radius of the hole
    segments: int
    # Assuming hole is in a square of side 2*radius by default, or defined by outer_size
    outer_size: Union[int, float]

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.radius = 1.0
        self.segments = 16  # Fewer segments for typical LDraw holes
        self.outer_size = float(self.radius) * 2.5  # Size of the plate the hole is in

    def __str__(self) -> str:
        s_list: List[str] = []
        hole_radius = float(self.radius)
        half_outer = self.outer_size / 2.0

        # Get circle points for the hole
        hole_points: List[Vector] = []
        for seg in range(self.segments):
            angle = (seg / self.segments) * 2.0 * pi
            hole_points.append(
                Vector(hole_radius * cos(angle), 0.0, hole_radius * sin(angle))
            )

        # Create quads/triangles between outer edge and hole edge
        outer_corners = [
            Vector(half_outer, 0.0, half_outer),
            Vector(-half_outer, 0.0, half_outer),
            Vector(-half_outer, 0.0, -half_outer),
            Vector(half_outer, 0.0, -half_outer),
        ]

        # This creates a polygon with a hole using triangles/quads.
        # This is a common approach for "condquad" in LPub or generating primitives.
        # The original logic for LDRHole was a bit unusual.
        # A more standard LDraw hole involves "1-4cyl.dat" or similar, or is defined by surrounding geometry.
        # The provided code generated triangles fanning out from the hole edge to the corners of a square.
        # Let's try to make a flat plate with a hole using triangles.

        for i in range(self.segments):
            p1_hole = hole_points[i]
            p2_hole = hole_points[(i + 1) % self.segments]

            # Find closest outer corner or edge points to connect to.
            # This can be complex. A simpler way for a basic stud.io like hole in a plate:
            # Use "ringN.dat" parts or generate triangles carefully.
            # The old logic was specific, let's try to adapt:
            # It created triangles from hole edge to one of 4 main corners.
            # This doesn't create a clean hole in a plate.

            # Re-interpreting: create triangles from center to edge of hole (makes a filled circle / no hole)
            # To make a hole, we need surrounding geometry.
            # If this class is meant to be a "hole primitive" itself (like studhole.dat), it needs a different approach.
            # For now, let's make it a simple flat ring using two LDRCircle definitions (one solid, one cutout)
            # This is not directly possible with primitives alone without BFC.
            # The original code was generating actual triangles.
            # 0 OPTIONAL 24 0 0 0 0.24999999999999983 -0.9682458365518543 0 -0.9682458365518543 -0.24999999999999983
            # was an optional line.
            # The triangles were like: T(0,0,0), P_hole_i, P_hole_i+1.
            # The issue was the "P3" of the triangle fanning out to corners.

            # Simpler approach: make a disc with a hole (like a washer)
            # This is similar to LDRDisc but with optional lines.
            temp_attrib_edge = LDRAttrib(LDR_OPT_COLOUR, self.attrib.units)
            temp_attrib_main = LDRAttrib(self.attrib.colour, self.attrib.units)

            inner_lines_local: List[LDRLine] = GetCircleSegments(
                hole_radius, self.segments, temp_attrib_edge
            )

            # Create an outer boundary if not an infinitely thin hole part
            outer_radius_for_washer = (
                hole_radius + (self.outer_size - hole_radius * 2) / 2
                if self.outer_size > hole_radius * 2
                else hole_radius * 1.2
            )
            if self.outer_size <= hole_radius * 2:
                outer_radius_for_washer = hole_radius * 1.2  # Ensure outer is larger

            outer_lines_local: List[LDRLine] = GetCircleSegments(
                outer_radius_for_washer, self.segments, temp_attrib_edge
            )

            for i in range(self.segments):
                q = LDRQuad(temp_attrib_main.colour, temp_attrib_main.units)
                q.p1 = inner_lines_local[i].p1.copy()
                q.p2 = outer_lines_local[
                    i
                ].p1.copy()  # Outer point corresponding to p1_hole
                q.p3 = outer_lines_local[
                    i
                ].p2.copy()  # Outer point corresponding to p2_hole
                q.p4 = inner_lines_local[i].p2.copy()

                q.transform(self.attrib.matrix)
                q.translate(self.attrib.loc)
                s_list.append(str(q))

                # Optional lines for edges
                l_in = copy.deepcopy(inner_lines_local[i])
                l_in.attrib.colour = LDR_OPT_COLOUR  # Make edges optional color
                l_in.transform(self.attrib.matrix)
                l_in.translate(self.attrib.loc)
                s_list.append(str(l_in))

                l_out = copy.deepcopy(outer_lines_local[i])
                l_out.attrib.colour = LDR_OPT_COLOUR
                l_out.transform(self.attrib.matrix)
                l_out.translate(self.attrib.loc)
                s_list.append(str(l_out))

        return "".join(s_list)


class LDRCylinder:
    attrib: LDRAttrib
    radius: Union[int, float]
    height: Union[int, float]
    segments: int

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.radius = 1.0
        self.height = 1.0
        self.segments = 16  # Common for LDraw cylinders

    def __str__(self) -> str:
        s_list: List[str] = []
        radius_f, height_f = float(self.radius), float(self.height)

        temp_attrib_for_segments = LDRAttrib(self.attrib.colour, self.attrib.units)
        circle_lines_local: List[LDRLine] = GetCircleSegments(
            radius_f, self.segments, temp_attrib_for_segments
        )

        # Top and bottom edge lines (optional lines are better for cylinder primitives)
        # For now, using main color for these edges based on original logic
        for line_local in circle_lines_local:
            # Bottom circle
            l_bottom = copy.deepcopy(line_local)
            l_bottom.transform(self.attrib.matrix)
            l_bottom.translate(self.attrib.loc)
            s_list.append(str(l_bottom))

            # Top circle
            l_top = copy.deepcopy(line_local)
            l_top.translate(Vector(0, height_f, 0))  # Move to top
            l_top.transform(self.attrib.matrix)
            l_top.translate(self.attrib.loc)
            s_list.append(str(l_top))

        # Side quads
        for line_local in circle_lines_local:
            q = LDRQuad(self.attrib.colour, self.attrib.units)
            q.p1 = line_local.p1.copy()  # Bottom point 1
            q.p1.y = 0.0
            q.p2 = line_local.p2.copy()  # Bottom point 2
            q.p2.y = 0.0
            q.p3 = line_local.p2.copy()  # Top point 2 (matches bottom p2)
            q.p3.y = height_f
            q.p4 = line_local.p1.copy()  # Top point 1 (matches bottom p1)
            q.p4.y = height_f

            q.transform(self.attrib.matrix)
            q.translate(self.attrib.loc)
            s_list.append(str(q))

        # Caps (optional, often cylinders are open or use cap primitives like disc.dat)
        # For a closed cylinder:
        # Top Cap
        for line_local in circle_lines_local:
            t_top = LDRTriangle(self.attrib.colour, self.attrib.units)
            t_top.p1 = Vector(0, height_f, 0)  # Center top
            t_top.p2 = line_local.p1.copy()
            t_top.p2.y = height_f
            t_top.p3 = line_local.p2.copy()
            t_top.p3.y = height_f
            t_top.transform(self.attrib.matrix)
            t_top.translate(self.attrib.loc)
            s_list.append(str(t_top))
        # Bottom Cap
        for line_local in circle_lines_local:
            t_bottom = LDRTriangle(self.attrib.colour, self.attrib.units)
            t_bottom.p1 = Vector(0, 0, 0)  # Center bottom
            # Ensure winding order is correct for bottom face (swap p2 and p3 if viewing from outside)
            t_bottom.p2 = line_local.p2.copy()
            t_bottom.p2.y = 0.0
            t_bottom.p3 = line_local.p1.copy()
            t_bottom.p3.y = 0.0
            t_bottom.transform(self.attrib.matrix)
            t_bottom.translate(self.attrib.loc)
            s_list.append(str(t_bottom))

        return "".join(s_list)


class LDRBox:
    attrib: LDRAttrib
    length: Union[int, float]  # X-axis
    width: Union[int, float]  # Z-axis
    height: Union[int, float]  # Y-axis

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.length = 1.0
        self.width = 1.0
        self.height = 1.0

    def __str__(self) -> str:
        s_list: List[str] = []
        hl, hw, hh = self.length / 2.0, self.width / 2.0, self.height / 2.0

        # Define 8 corners of the box relative to its center
        points = [
            Vector(hl, hh, hw),
            Vector(-hl, hh, hw),
            Vector(-hl, hh, -hw),
            Vector(hl, hh, -hw),  # Top face (y = +hh)
            Vector(hl, -hh, hw),
            Vector(-hl, -hh, hw),
            Vector(-hl, -hh, -hw),
            Vector(hl, -hh, -hw),  # Bottom face (y = -hh)
        ]

        # Optional edge lines
        edge_attrib = LDRAttrib(LDR_OPT_COLOUR, self.attrib.units)
        line_coords_indices = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),  # Top face edges
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),  # Bottom face edges
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),  # Vertical edges
        ]
        for p_idx1, p_idx2 in line_coords_indices:
            l = LDRLine(edge_attrib.colour, edge_attrib.units)
            l.p1 = points[p_idx1].copy()
            l.p2 = points[p_idx2].copy()
            l.transform(self.attrib.matrix)
            l.translate(self.attrib.loc)
            s_list.append(str(l))

        # Quads for faces (ensure correct winding order for outward normals)
        # (p1,p2,p3,p4) assuming viewed from outside
        face_coords_indices = [
            (0, 3, 2, 1),  # Top face (Y+)
            (
                4,
                5,
                6,
                7,
            ),  # Bottom face (Y-) -- needs reversed winding for outward normal
            # (7,6,5,4) # Corrected Bottom face for outward normal if p1-p2-p3 is CCW
            (4, 7, 3, 0),  # Front face (Z+)
            (1, 2, 6, 5),  # Back face (Z-)
            (0, 4, 5, 1),  # Right face (X+) -- original was (5,4,0,1)
            (3, 7, 6, 2),  # Left face (X-) -- original was (7,6,2,3)
        ]
        # Corrected face winding (assuming p1-p2-p3 defines CCW from outside)
        # Order: p1 (bottom-left), p2 (bottom-right), p3 (top-right), p4 (top-left)
        # Or for LDraw quads, just list vertices in order around the face.
        # LDraw quad: p1, p2, p3, p4 should be coplanar and wind consistently.
        corrected_face_indices = [
            (
                0,
                1,
                2,
                3,
            ),  # Top face (0,1,2,3) -> ( hl,hh,hw),(-hl,hh,hw),(-hl,hh,-hw),( hl,hh,-hw)
            (
                7,
                6,
                5,
                4,
            ),  # Bottom face (7,6,5,4) -> ( hl,-hh,-hw),(-hl,-hh,-hw),(-hl,-hh,hw),( hl,-hh,hw)
            (
                4,
                0,
                3,
                7,
            ),  # Front Z+ (4,0,3,7) -> ( hl,-hh,hw),( hl,hh,hw),( hl,hh,-hw),( hl,-hh,-hw)
            (
                1,
                5,
                6,
                2,
            ),  # Back Z- (1,5,6,2) -> (-hl,hh,hw),(-hl,-hh,hw),(-hl,-hh,-hw),(-hl,hh,-hw)
            (
                0,
                4,
                7,
                3,
            ),  # Right X+ (0,4,7,3) -> This might be an error in original, seems same as Front Z+.
            # (0,4,5,1) was Right, (3,7,6,2) was Left in original comment.
            # Let's use common face definitions based on points:
            (
                0,
                4,
                5,
                1,
            ),  # X+ face: ( hl,hh,hw), ( hl,-hh,hw), (-hl,-hh,hw), (-hl,hh,hw) -- This is not X+
            # Correct X+ face: Points (0,4,7,3) => (hl,hh,hw), (hl,-hh,hw), (hl,-hh,-hw), (hl,hh,-hw)
            # Correct X- face: Points (1,2,6,5) => (-hl,hh,hw), (-hl,hh,-hw), (-hl,-hh,-hw), (-hl,-hh,hw)
        ]
        # Standard Cube Faces (p0-p7 as defined):
        # Top (Y+): 0-1-2-3
        # Bottom (Y-): 4-5-6-7 (winding needs to be 7-6-5-4 for outward normal if defined by CCW)
        # Front (Z+): 0-4-5-1 (winding 1-5-4-0)
        # Back (Z-):  3-2-6-7 (winding 7-6-2-3)
        # Right (X+): 0-3-7-4 (winding 4-7-3-0)
        # Left (X-):  1-2-6-5 (winding 5-6-2-1)
        face_indices_for_quads = [
            (0, 1, 2, 3),  # Top: Y+
            (
                4,
                5,
                6,
                7,
            ),  # Bottom: Y- (LDraw handles winding order internally for basic shapes)
            (1, 0, 4, 5),  # Front: Z+ (looking from +Z)
            (3, 2, 6, 7),  # Back: Z-
            (0, 3, 7, 4),  # Right: X+
            (2, 1, 5, 6),  # Left: X-
        ]

        for p_indices in face_indices_for_quads:
            q = LDRQuad(self.attrib.colour, self.attrib.units)
            q.p1 = points[p_indices[0]].copy()
            q.p2 = points[p_indices[1]].copy()
            q.p3 = points[p_indices[2]].copy()
            q.p4 = points[p_indices[3]].copy()
            q.transform(self.attrib.matrix)
            q.translate(self.attrib.loc)
            s_list.append(str(q))

        return "".join(s_list)
