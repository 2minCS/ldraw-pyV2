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

import copy

# import os # REMOVED - Unused
from math import pi, cos, sin
from typing import (
    List,
    Union,
    Tuple,
    Optional,
)  # Tuple, Optional were not explicitly used but good for future

# Explicit imports from toolbox
from toolbox import Vector, Identity  # type: ignore

# Explicit relative imports from ldrawpy package
from .ldrprimitives import (
    LDRAttrib,
    LDRQuad,
    LDRLine,
    LDRTriangle,
)  # REPLACED wildcard import
from .ldrhelpers import GetCircleSegments
from .constants import LDR_DEF_COLOUR, LDR_OPT_COLOUR  # REPLACED wildcard import


class LDRPolyWall:
    """
    Represents a polygonal wall made of LDRQuad primitives.
    The wall is defined by a list of points in the XZ plane and a height.
    """

    attrib: LDRAttrib
    height: Union[int, float]
    points: List[Vector]  # type: ignore # List of Vector objects defining the polygon base

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.height = 1.0  # Default height
        self.points = []

    def __str__(self) -> str:
        s_list: List[str] = []
        num_points = len(self.points)
        if num_points < 2:  # Need at least 2 points to form a segment of a wall
            return ""

        for i in range(num_points):
            q = LDRQuad(self.attrib.colour, self.attrib.units)

            p_current_base = self.points[i]
            p_next_base = self.points[(i + 1) % num_points]

            q.p1.x = p_current_base.x
            q.p1.y = float(self.height)
            q.p1.z = p_current_base.z

            q.p2.x = p_next_base.x
            q.p2.y = float(self.height)
            q.p2.z = p_next_base.z

            q.p3.x = p_next_base.x
            q.p3.y = 0.0
            q.p3.z = p_next_base.z

            q.p4.x = p_current_base.x
            q.p4.y = 0.0
            q.p4.z = p_current_base.z

            q.transform(self.attrib.matrix)
            q.translate(self.attrib.loc)
            s_list.append(str(q))

        return "".join(s_list)


class LDRRect:
    """
    Represents a rectangle in the XZ plane, composed of an LDRQuad and optional edge lines.
    """

    attrib: LDRAttrib
    length: Union[int, float]
    width: Union[int, float]

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.length = 1.0
        self.width = 1.0

    def __str__(self) -> str:
        s: List[str] = []
        half_len, half_width = float(self.length) / 2.0, float(self.width) / 2.0

        q = LDRQuad(self.attrib.colour, self.attrib.units)
        q.p1 = Vector(-half_width, 0, half_len)
        q.p2 = Vector(-half_width, 0, -half_len)
        q.p3 = Vector(half_width, 0, -half_len)
        q.p4 = Vector(half_width, 0, half_len)

        quad_for_lines = copy.deepcopy(q)
        quad_for_lines.transform(self.attrib.matrix)
        quad_for_lines.translate(self.attrib.loc)

        edge_attrib = LDRAttrib(LDR_OPT_COLOUR, self.attrib.units)
        line_edge = LDRLine(edge_attrib.colour, edge_attrib.units)

        line_edge.p1, line_edge.p2 = quad_for_lines.p1, quad_for_lines.p2
        s.append(str(line_edge))
        line_edge.p1, line_edge.p2 = quad_for_lines.p2, quad_for_lines.p3
        s.append(str(line_edge))
        line_edge.p1, line_edge.p2 = quad_for_lines.p3, quad_for_lines.p4
        s.append(str(line_edge))
        line_edge.p1, line_edge.p2 = quad_for_lines.p4, quad_for_lines.p1
        s.append(str(line_edge))

        q.transform(self.attrib.matrix)
        q.translate(self.attrib.loc)
        s.append(str(q))

        return "".join(s)


class LDRCircle:
    """
    Represents a circle in the XZ plane, made of LDRLine segments.
    Can optionally be filled with LDRTriangle primitives.
    """

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
        s: List[str] = []
        line_attrib = LDRAttrib(self.attrib.colour, self.attrib.units)

        circle_lines: List[LDRLine] = GetCircleSegments(
            float(self.radius), self.segments, line_attrib
        )

        for single_line in circle_lines:
            line_transformed = copy.deepcopy(single_line)
            line_transformed.transform(self.attrib.matrix)
            line_transformed.translate(self.attrib.loc)
            s.append(str(line_transformed))

        if self.fill:
            center_point = Vector(0, 0, 0)  # type: ignore
            for fill_line_segment in circle_lines:
                tri = LDRTriangle(self.attrib.colour, self.attrib.units)
                tri.p1 = center_point.copy()
                tri.p2 = fill_line_segment.p1.copy()
                tri.p3 = fill_line_segment.p2.copy()

                tri.transform(self.attrib.matrix)
                tri.translate(self.attrib.loc)
                s.append(str(tri))

        return "".join(s)


class LDRDisc:
    """
    Represents a disc (a ring or annulus) in the XZ plane,
    made of LDRQuad primitives connecting an inner and outer circle.
    """

    attrib: LDRAttrib
    radius: Union[int, float]
    border: Union[int, float]
    segments: int

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.radius = 1.0
        self.border = 0.2
        self.segments = 24

    def __str__(self) -> str:
        s: List[str] = []
        inner_radius = float(self.radius)
        outer_radius = float(self.radius) + float(self.border)

        segment_attrib = LDRAttrib(self.attrib.colour, self.attrib.units)

        inner_circle_lines: List[LDRLine] = GetCircleSegments(
            inner_radius, self.segments, segment_attrib
        )
        outer_circle_lines: List[LDRLine] = GetCircleSegments(
            outer_radius, self.segments, segment_attrib
        )

        for i in range(self.segments):
            q = LDRQuad(self.attrib.colour, self.attrib.units)
            q.p1 = inner_circle_lines[i].p1.copy()
            q.p2 = inner_circle_lines[i].p2.copy()
            q.p3 = outer_circle_lines[i].p2.copy()
            q.p4 = outer_circle_lines[i].p1.copy()

            q.transform(self.attrib.matrix)
            q.translate(self.attrib.loc)
            s.append(str(q))

        return "".join(s)


class LDRHole:
    """
    Represents a filled planar surface with a circular hole in the center,
    lying in the XZ plane. Made of LDRQuads and optional edge lines.
    """

    attrib: LDRAttrib
    radius: Union[int, float]
    segments: int
    outer_size: Union[int, float]

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.radius = 1.0
        self.segments = 16
        self.outer_size = float(self.radius) * 2.5

    def __str__(self) -> str:
        s: List[str] = []
        hole_radius = float(self.radius)

        edge_attrib = LDRAttrib(LDR_OPT_COLOUR, self.attrib.units)
        surface_attrib = LDRAttrib(self.attrib.colour, self.attrib.units)

        inner_hole_lines: List[LDRLine] = GetCircleSegments(
            hole_radius, self.segments, edge_attrib
        )

        outer_boundary_radius = float(self.outer_size) / 2.0
        if outer_boundary_radius <= hole_radius:
            outer_boundary_radius = hole_radius * 1.2

        outer_boundary_lines: List[LDRLine] = GetCircleSegments(
            outer_boundary_radius, self.segments, edge_attrib
        )

        for i in range(self.segments):
            q = LDRQuad(surface_attrib.colour, surface_attrib.units)
            q.p1 = inner_hole_lines[i].p1.copy()
            q.p2 = outer_boundary_lines[i].p1.copy()
            q.p3 = outer_boundary_lines[i].p2.copy()
            q.p4 = inner_hole_lines[i].p2.copy()

            q.transform(self.attrib.matrix)
            q.translate(self.attrib.loc)
            s.append(str(q))

            line_inner_edge = copy.deepcopy(inner_hole_lines[i])
            line_inner_edge.attrib.colour = LDR_OPT_COLOUR
            line_inner_edge.transform(self.attrib.matrix)
            line_inner_edge.translate(self.attrib.loc)
            s.append(str(line_inner_edge))

            line_outer_edge = copy.deepcopy(outer_boundary_lines[i])
            line_outer_edge.attrib.colour = LDR_OPT_COLOUR
            line_outer_edge.transform(self.attrib.matrix)
            line_outer_edge.translate(self.attrib.loc)
            s.append(str(line_outer_edge))

        return "".join(s)


class LDRCylinder:
    """
    Represents a cylinder aligned with the Y-axis,
    composed of LDRQuads for the sides and LDRTriangles for the caps.
    Includes optional edge lines.
    """

    attrib: LDRAttrib
    radius: Union[int, float]
    height: Union[int, float]
    segments: int

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.radius = 1.0
        self.height = 1.0
        self.segments = 16

    def __str__(self) -> str:
        s: List[str] = []
        cyl_radius, cyl_height = float(self.radius), float(self.height)

        edge_line_attrib = LDRAttrib(LDR_OPT_COLOUR, self.attrib.units)

        base_circle_lines: List[LDRLine] = GetCircleSegments(
            cyl_radius, self.segments, edge_line_attrib
        )

        for base_line_segment in base_circle_lines:
            line_bottom_edge = copy.deepcopy(base_line_segment)
            line_bottom_edge.transform(self.attrib.matrix)
            line_bottom_edge.translate(self.attrib.loc)
            s.append(str(line_bottom_edge))

            line_top_edge = copy.deepcopy(base_line_segment)
            line_top_edge.translate(Vector(0, cyl_height, 0))  # type: ignore
            line_top_edge.transform(self.attrib.matrix)
            line_top_edge.translate(self.attrib.loc)
            s.append(str(line_top_edge))

        for side_quad_base_segment in base_circle_lines:
            q = LDRQuad(self.attrib.colour, self.attrib.units)
            q.p1 = side_quad_base_segment.p1.copy()
            q.p1.y = 0.0
            q.p2 = side_quad_base_segment.p2.copy()
            q.p2.y = 0.0
            q.p3 = side_quad_base_segment.p2.copy()
            q.p3.y = cyl_height
            q.p4 = side_quad_base_segment.p1.copy()
            q.p4.y = cyl_height

            q.transform(self.attrib.matrix)
            q.translate(self.attrib.loc)
            s.append(str(q))

        center_bottom = Vector(0, 0, 0)  # type: ignore
        center_top = Vector(0, cyl_height, 0)  # type: ignore

        for cap_tri_base_segment in base_circle_lines:
            tri_top = LDRTriangle(self.attrib.colour, self.attrib.units)
            tri_top.p1 = center_top.copy()
            tri_top.p2 = cap_tri_base_segment.p1.copy()
            tri_top.p2.y = cyl_height
            tri_top.p3 = cap_tri_base_segment.p2.copy()
            tri_top.p3.y = cyl_height

            tri_top.transform(self.attrib.matrix)
            tri_top.translate(self.attrib.loc)
            s.append(str(tri_top))

            tri_bottom = LDRTriangle(self.attrib.colour, self.attrib.units)
            tri_bottom.p1 = center_bottom.copy()
            tri_bottom.p2 = cap_tri_base_segment.p2.copy()
            tri_bottom.p2.y = 0.0
            tri_bottom.p3 = cap_tri_base_segment.p1.copy()
            tri_bottom.p3.y = 0.0

            tri_bottom.transform(self.attrib.matrix)
            tri_bottom.translate(self.attrib.loc)
            s.append(str(tri_bottom))

        return "".join(s)


class LDRBox:
    """
    Represents a 3D box centered at the origin before transformation,
    composed of LDRQuads for faces and optional LDRLines for edges.
    """

    attrib: LDRAttrib
    length: Union[int, float]
    width: Union[int, float]
    height: Union[int, float]

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.length = 1.0
        self.width = 1.0
        self.height = 1.0

    def __str__(self) -> str:
        s: List[str] = []
        hl, hw, hh = (
            float(self.length) / 2.0,
            float(self.width) / 2.0,
            float(self.height) / 2.0,
        )

        pts = [
            Vector(-hw, hh, hl),
            Vector(hw, hh, hl),
            Vector(hw, hh, -hl),
            Vector(-hw, hh, -hl),
            Vector(-hw, -hh, hl),
            Vector(hw, -hh, hl),
            Vector(hw, -hh, -hl),
            Vector(-hw, -hh, -hl),
        ]  # type: ignore

        edge_attrib = LDRAttrib(LDR_OPT_COLOUR, self.attrib.units)
        edge_indices = [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        for p_idx1, p_idx2 in edge_indices:
            line_edge = LDRLine(edge_attrib.colour, edge_attrib.units)
            line_edge.p1 = pts[p_idx1].copy()
            line_edge.p2 = pts[p_idx2].copy()
            line_edge.transform(self.attrib.matrix)
            line_edge.translate(self.attrib.loc)
            s.append(str(line_edge))

        face_indices_quads = [
            (0, 3, 2, 1),
            (4, 5, 6, 7),
            (0, 1, 5, 4),
            (1, 2, 6, 5),
            (2, 3, 7, 6),
            (3, 0, 4, 7),
        ]
        for p_indices in face_indices_quads:
            q = LDRQuad(self.attrib.colour, self.attrib.units)
            q.p1 = pts[p_indices[0]].copy()
            q.p2 = pts[p_indices[1]].copy()
            q.p3 = pts[p_indices[2]].copy()
            q.p4 = pts[p_indices[3]].copy()

            q.transform(self.attrib.matrix)
            q.translate(self.attrib.loc)
            s.append(str(q))

        return "".join(s)
