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

import os
import copy
from math import pi, cos, sin
from typing import List, Union, Tuple, Optional

# Explicit imports from toolbox
from toolbox import Vector, Identity  # type: ignore

# Explicit relative imports from ldrawpy package
from .ldrprimitives import LDRAttrib, LDRQuad, LDRLine, LDRTriangle
from .ldrhelpers import GetCircleSegments
from .constants import LDR_DEF_COLOUR, LDR_OPT_COLOUR


class LDRPolyWall:
    attrib: LDRAttrib
    height: Union[int, float]
    points: List[Vector]  # type: ignore

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.height = 1.0
        self.points = []

    def __str__(self) -> str:
        s_list: List[str] = []
        nPoints = len(self.points)
        if nPoints == 0:
            return ""
        for i in range(nPoints):
            q = LDRQuad(self.attrib.colour, self.attrib.units)
            q.p1.x = self.points[i].x
            q.p1.y = float(self.height)
            q.p1.z = self.points[i].z
            thePoint_idx = (i + 1) % nPoints
            thePoint = self.points[thePoint_idx]
            q.p2.x = thePoint.x
            q.p2.y = float(self.height)
            q.p2.z = thePoint.z
            q.p3.x = thePoint.x
            q.p3.y = 0.0
            q.p3.z = thePoint.z
            q.p4.x = self.points[i].x
            q.p4.y = 0.0
            q.p4.z = self.points[i].z
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
        self.length = 1.0
        self.width = 1.0

    def __str__(self) -> str:
        s: List[str] = []
        hl, hw = float(self.length) / 2.0, float(self.width) / 2.0
        q = LDRQuad(self.attrib.colour, self.attrib.units)
        q.p1, q.p2, q.p3, q.p4 = Vector(-hl, 0, hw), Vector(-hl, 0, -hw), Vector(hl, 0, -hw), Vector(hl, 0, hw)  # type: ignore
        qtfl = copy.deepcopy(q)
        qtfl.transform(self.attrib.matrix)
        qtfl.translate(self.attrib.loc)
        la = LDRAttrib(LDR_OPT_COLOUR, self.attrib.units)
        l = LDRLine(la.colour, la.units)
        l.p1, l.p2 = qtfl.p1, qtfl.p2
        s.append(str(l))
        l.p1, l.p2 = qtfl.p2, qtfl.p3
        s.append(str(l))
        l.p1, l.p2 = qtfl.p3, qtfl.p4
        s.append(str(l))
        l.p1, l.p2 = qtfl.p4, qtfl.p1
        s.append(str(l))
        q.transform(self.attrib.matrix)
        q.translate(self.attrib.loc)
        s.append(str(q))
        return "".join(s)


class LDRCircle:
    attrib: LDRAttrib
    radius: Union[int, float]
    segments: int
    fill: bool

    def __init__(self, c: int = LDR_DEF_COLOUR, u: str = "ldu"):
        self.attrib = LDRAttrib(c, u)
        self.radius = 1.0
        self.segments = 24
        self.fill = False

    def __str__(self) -> str:
        s: List[str] = []
        tas = LDRAttrib(self.attrib.colour, self.attrib.units)
        cll: List[LDRLine] = GetCircleSegments(float(self.radius), self.segments, tas)
        for ll in cll:
            lt = copy.deepcopy(ll)
            lt.transform(self.attrib.matrix)
            lt.translate(self.attrib.loc)
            s.append(str(lt))
        if self.fill:
            for ll_fill in cll:
                t = LDRTriangle(self.attrib.colour, self.attrib.units)
                t.p1 = Vector(0, 0, 0)
                t.p2 = ll_fill.p1.copy()
                t.p3 = ll_fill.p2.copy()  # type: ignore
                t.transform(self.attrib.matrix)
                t.translate(self.attrib.loc)
                s.append(str(t))
        return "".join(s)


class LDRDisc:
    attrib: LDRAttrib
    radius: Union[int, float]
    border: Union[int, float]
    segments: int

    def __init__(self, c: int = LDR_DEF_COLOUR, u: str = "ldu"):
        self.attrib = LDRAttrib(c, u)
        self.radius = 1.0
        self.border = 0.2
        self.segments = 24

    def __str__(self) -> str:
        s: List[str] = []
        ir, obr = float(self.radius), float(self.radius) + float(self.border)
        tas = LDRAttrib(self.attrib.colour, self.attrib.units)
        ill: List[LDRLine] = GetCircleSegments(ir, self.segments, tas)
        oll: List[LDRLine] = GetCircleSegments(obr, self.segments, tas)
        for i in range(self.segments):
            q = LDRQuad(self.attrib.colour, self.attrib.units)
            q.p1 = ill[i].p1.copy()
            q.p2 = ill[i].p2.copy()
            q.p3 = oll[i].p2.copy()
            q.p4 = oll[i].p1.copy()
            q.transform(self.attrib.matrix)
            q.translate(self.attrib.loc)
            s.append(str(q))
        return "".join(s)


class LDRHole:
    attrib: LDRAttrib
    radius: Union[int, float]
    segments: int
    outer_size: Union[int, float]

    def __init__(self, c: int = LDR_DEF_COLOUR, u: str = "ldu"):
        self.attrib = LDRAttrib(c, u)
        self.radius = 1.0
        self.segments = 16
        self.outer_size = float(self.radius) * 2.5

    def __str__(self) -> str:
        s: List[str] = []
        hr = float(self.radius)
        tae = LDRAttrib(LDR_OPT_COLOUR, self.attrib.units)
        tam = LDRAttrib(self.attrib.colour, self.attrib.units)
        ill: List[LDRLine] = GetCircleSegments(hr, self.segments, tae)
        oruw = (
            hr + (self.outer_size - hr * 2) / 2
            if self.outer_size > hr * 2
            else hr * 1.2
        )
        if self.outer_size <= hr * 2:
            oruw = hr * 1.2
        oll: List[LDRLine] = GetCircleSegments(oruw, self.segments, tae)
        for i in range(self.segments):
            q = LDRQuad(tam.colour, tam.units)
            q.p1 = ill[i].p1.copy()
            q.p2 = oll[i].p1.copy()
            q.p3 = oll[i].p2.copy()
            q.p4 = ill[i].p2.copy()
            q.transform(self.attrib.matrix)
            q.translate(self.attrib.loc)
            s.append(str(q))
            li = copy.deepcopy(ill[i])
            li.attrib.colour = LDR_OPT_COLOUR
            li.transform(self.attrib.matrix)
            li.translate(self.attrib.loc)
            s.append(str(li))
            lo = copy.deepcopy(oll[i])
            lo.attrib.colour = LDR_OPT_COLOUR
            lo.transform(self.attrib.matrix)
            lo.translate(self.attrib.loc)
            s.append(str(lo))
        return "".join(s)


class LDRCylinder:
    attrib: LDRAttrib
    radius: Union[int, float]
    height: Union[int, float]
    segments: int

    def __init__(self, c: int = LDR_DEF_COLOUR, u: str = "ldu"):
        self.attrib = LDRAttrib(c, u)
        self.radius = 1.0
        self.height = 1.0
        self.segments = 16

    def __str__(self) -> str:
        s: List[str] = []
        rf, hf = float(self.radius), float(self.height)
        tas = LDRAttrib(self.attrib.colour, self.attrib.units)
        cll: List[LDRLine] = GetCircleSegments(rf, self.segments, tas)
        for ll in cll:
            lb = copy.deepcopy(ll)
            lb.transform(self.attrib.matrix)
            lb.translate(self.attrib.loc)
            s.append(str(lb))
            lt = copy.deepcopy(ll)
            lt.translate(Vector(0, hf, 0))
            lt.transform(self.attrib.matrix)
            lt.translate(self.attrib.loc)
            s.append(str(lt))  # type: ignore
        for ll_quad in cll:
            q = LDRQuad(self.attrib.colour, self.attrib.units)
            q.p1 = ll_quad.p1.copy()
            q.p1.y = 0.0
            q.p2 = ll_quad.p2.copy()
            q.p2.y = 0.0
            q.p3 = ll_quad.p2.copy()
            q.p3.y = hf
            q.p4 = ll_quad.p1.copy()
            q.p4.y = hf
            q.transform(self.attrib.matrix)
            q.translate(self.attrib.loc)
            s.append(str(q))
        for ll_cap in cll:
            tt = LDRTriangle(self.attrib.colour, self.attrib.units)
            tt.p1 = Vector(0, hf, 0)
            tt.p2 = ll_cap.p1.copy()
            tt.p2.y = hf
            tt.p3 = ll_cap.p2.copy()
            tt.p3.y = hf  # type: ignore
            tt.transform(self.attrib.matrix)
            tt.translate(self.attrib.loc)
            s.append(str(tt))
            tb = LDRTriangle(self.attrib.colour, self.attrib.units)
            tb.p1 = Vector(0, 0, 0)
            tb.p2 = ll_cap.p2.copy()
            tb.p2.y = 0.0
            tb.p3 = ll_cap.p1.copy()
            tb.p3.y = 0.0  # type: ignore
            tb.transform(self.attrib.matrix)
            tb.translate(self.attrib.loc)
            s.append(str(tb))
        return "".join(s)


class LDRBox:
    attrib: LDRAttrib
    length: Union[int, float]
    width: Union[int, float]
    height: Union[int, float]

    def __init__(self, c: int = LDR_DEF_COLOUR, u: str = "ldu"):
        self.attrib = LDRAttrib(c, u)
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
            Vector(hl, hh, hw),
            Vector(-hl, hh, hw),
            Vector(-hl, hh, -hw),
            Vector(hl, hh, -hw),  # type: ignore
            Vector(hl, -hh, hw),
            Vector(-hl, -hh, hw),
            Vector(-hl, -hh, -hw),
            Vector(hl, -hh, -hw),
        ]  # type: ignore
        ea = LDRAttrib(LDR_OPT_COLOUR, self.attrib.units)
        lci = [
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
        for pi1, pi2 in lci:
            l = LDRLine(ea.colour, ea.units)
            l.p1 = pts[pi1].copy()
            l.p2 = pts[pi2].copy()
            l.transform(self.attrib.matrix)
            l.translate(self.attrib.loc)
            s.append(str(l))
        fiq = [
            (0, 1, 2, 3),
            (4, 5, 6, 7),
            (1, 0, 4, 5),
            (3, 2, 6, 7),
            (0, 3, 7, 4),
            (2, 1, 5, 6),
        ]
        for pis in fiq:
            q = LDRQuad(self.attrib.colour, self.attrib.units)
            q.p1, q.p2, q.p3, q.p4 = (
                pts[pis[0]].copy(),
                pts[pis[1]].copy(),
                pts[pis[2]].copy(),
                pts[pis[3]].copy(),
            )
            q.transform(self.attrib.matrix)
            q.translate(self.attrib.loc)
            s.append(str(q))
        return "".join(s)
