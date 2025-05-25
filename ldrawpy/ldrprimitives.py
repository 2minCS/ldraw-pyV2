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
# LDraw primitives

import hashlib
from functools import reduce
from typing import List, Tuple, Union, Optional
from dataclasses import dataclass, field

# Explicit imports from toolbox
from toolbox import Vector, Matrix, Identity, euler_to_rot_matrix, safe_vector  # type: ignore

# Explicit imports from .ldrhelpers (relative import within the package)
from .ldrhelpers import vector_str, mat_str, quantize
from .constants import LDR_DEF_COLOUR


@dataclass(eq=False)
class LDRAttrib:
    colour: int = LDR_DEF_COLOUR
    units: str = "ldu"
    loc: Vector = field(default_factory=lambda: Vector(0, 0, 0))  # type: ignore
    matrix: Matrix = field(default_factory=Identity)  # type: ignore

    def __post_init__(self):
        self.colour = int(self.colour)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.colour == other.colour
            and self.units == other.units
            and self.loc.almost_same_as(other.loc)  # type: ignore
            and self.matrix.is_almost_same_as(other.matrix)
        )  # type: ignore

    def copy(self) -> "LDRAttrib":
        new_loc = self.loc.copy() if hasattr(self.loc, "copy") else self.loc
        new_matrix = self.matrix.copy() if hasattr(self.matrix, "copy") else self.matrix
        return LDRAttrib(
            colour=self.colour, units=self.units, loc=new_loc, matrix=new_matrix
        )


class LDRLine:
    __slots__ = ["attrib", "p1", "p2"]
    attrib: LDRAttrib
    p1: Vector
    p2: Vector  # type: ignore

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.p1, self.p2 = Vector(0, 0, 0), Vector(0, 0, 0)  # type: ignore

    def __str__(self) -> str:
        return (
            f"2 {self.attrib.colour} "
            f"{vector_str(self.p1, self.attrib)}{vector_str(self.p2, self.attrib)}\n"
        )  # type: ignore

    def translate(self, offset: Vector):
        self.p1 += offset
        self.p2 += offset  # type: ignore

    def transform(self, matrix: Matrix):
        self.p1 = self.p1 * matrix
        self.p2 = self.p2 * matrix  # type: ignore


class LDRTriangle:
    __slots__ = ["attrib", "p1", "p2", "p3"]
    attrib: LDRAttrib
    p1: Vector
    p2: Vector
    p3: Vector  # type: ignore

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.p1, self.p2, self.p3 = Vector(0, 0, 0), Vector(0, 0, 0), Vector(0, 0, 0)  # type: ignore

    def __str__(self) -> str:
        return (
            f"3 {self.attrib.colour} "
            f"{vector_str(self.p1, self.attrib)}{vector_str(self.p2, self.attrib)}"  # type: ignore
            f"{vector_str(self.p3, self.attrib)}\n"
        )  # type: ignore

    def translate(self, offset: Vector):
        self.p1 += offset
        self.p2 += offset
        self.p3 += offset  # type: ignore

    def transform(self, matrix: Matrix):
        self.p1 *= matrix
        self.p2 *= matrix
        self.p3 *= matrix  # type: ignore


class LDRQuad:
    __slots__ = ["attrib", "p1", "p2", "p3", "p4"]
    attrib: LDRAttrib
    p1: Vector
    p2: Vector
    p3: Vector
    p4: Vector  # type: ignore

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.p1, self.p2, self.p3, self.p4 = Vector(0, 0, 0), Vector(0, 0, 0), Vector(0, 0, 0), Vector(0, 0, 0)  # type: ignore

    def __str__(self) -> str:
        return (
            f"4 {self.attrib.colour} "
            f"{vector_str(self.p1, self.attrib)}{vector_str(self.p2, self.attrib)}"  # type: ignore
            f"{vector_str(self.p3, self.attrib)}{vector_str(self.p4, self.attrib)}\n"
        )  # type: ignore

    def translate(self, offset: Vector):
        self.p1 += offset
        self.p2 += offset
        self.p3 += offset
        self.p4 += offset  # type: ignore

    def transform(self, matrix: Matrix):
        self.p1 *= matrix
        self.p2 *= matrix
        self.p3 *= matrix
        self.p4 *= matrix  # type: ignore


class LDRPart:
    __slots__ = ["attrib", "name", "wrapcallout"]
    attrib: LDRAttrib
    name: str
    wrapcallout: bool

    def __init__(
        self,
        colour: int = LDR_DEF_COLOUR,
        name: Optional[str] = None,
        units: str = "ldu",
    ):
        self.attrib = LDRAttrib(colour, units)
        self.name = name if name is not None else ""
        self.wrapcallout = False

    def __str__(self) -> str:
        matrix_elements = []
        if hasattr(self.attrib.matrix, "rows") and self.attrib.matrix.rows:
            matrix_elements = [
                item for sublist in self.attrib.matrix.rows for item in sublist
            ]
        else:
            matrix_elements = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        tup_matrix_str = mat_str(tuple(matrix_elements))
        name_to_write = self.name
        if name_to_write:
            ext = name_to_write[-4:].lower()
            if not (ext == ".ldr" or ext == ".dat"):
                name_to_write += ".dat"
        else:
            name_to_write = "unknown.dat"
        s = (
            f"1 {self.attrib.colour} "
            f"{vector_str(self.attrib.loc, self.attrib)}"  # type: ignore
            f"{tup_matrix_str}"
            f"{name_to_write}\n"
        )
        if self.wrapcallout and name_to_write.endswith(".ldr"):
            return f"0 !LPUB CALLOUT BEGIN\n{s}0 !LPUB CALLOUT END\n"
        return s

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.name == other.name and self.attrib.colour == other.attrib.colour

    def copy(self) -> "LDRPart":
        p = LDRPart(colour=self.attrib.colour, name=self.name, units=self.attrib.units)
        p.wrapcallout = self.wrapcallout
        p.attrib = self.attrib.copy()
        return p

    def sha1hash(self) -> str:
        shash = hashlib.sha1()
        shash.update(bytes(str(self), encoding="utf8"))
        return shash.hexdigest()

    def is_identical(self, other: "LDRPart") -> bool:
        return self.name == other.name and self.attrib == other.attrib

    def is_same(
        self,
        other: "LDRPart",
        ignore_location: bool = False,
        ignore_colour: bool = False,
        exact: bool = False,
    ) -> bool:
        if self.name != other.name:
            return False
        if exact:
            return self.is_identical(other)
        if not ignore_colour and self.attrib.colour != other.attrib.colour:
            return False
        if not ignore_location and not self.attrib.loc.almost_same_as(other.attrib.loc):
            return False  # type: ignore
        return True

    def is_coaligned(self, other: "LDRPart") -> bool:
        try:
            v1 = self.attrib.loc * self.attrib.matrix
            v2 = other.attrib.loc * other.attrib.matrix  # type: ignore
            if hasattr(v1, "is_colinear_with"):
                return v1.is_colinear_with(v2) == 2  # type: ignore
        except AttributeError:
            pass
        return False

    def change_colour(self, to_colour: int):
        self.attrib.colour = to_colour

    def set_rotation(self, angle: Union[Tuple[float, float, float], Vector]):
        self.attrib.matrix = euler_to_rot_matrix(angle)  # type: ignore

    def move_to(self, pos: Union[Tuple[float, float, float], Vector]):
        self.attrib.loc = safe_vector(pos)  # type: ignore

    def move_by(self, offset: Union[Tuple[float, float, float], Vector]):
        self.attrib.loc += safe_vector(offset)  # type: ignore

    def rotate_by(self, angle: Union[Tuple[float, float, float], Vector]):
        rm = euler_to_rot_matrix(angle)  # type: ignore
        self.attrib.loc = rm * self.attrib.loc  # type: ignore
        self.attrib.matrix = rm * self.attrib.matrix  # type: ignore

    def transform(self, matrix: Matrix = Identity(), offset: Vector = Vector(0, 0, 0)):  # type: ignore
        self.attrib.loc = matrix * self.attrib.loc + offset  # type: ignore
        self.attrib.matrix = matrix * self.attrib.matrix  # type: ignore

    def from_str(self, s: str) -> Optional["LDRPart"]:
        sl = s.strip().lower().split()
        if not len(sl) >= 14:
            return None
        try:
            if int(sl[0]) != 1:
                return None
            self.attrib.colour = int(sl[1])
            self.attrib.loc.x = quantize(sl[2])
            self.attrib.loc.y = quantize(sl[3])
            self.attrib.loc.z = quantize(sl[4])
            m = [quantize(sl[i]) for i in range(5, 14)]
            self.attrib.matrix = Matrix([m[0:3], m[3:6], m[6:9]])  # type: ignore
            self.name = " ".join(sl[14:]).replace(".dat", "") if sl[14:] else ""
        except (ValueError, IndexError):
            return None
        return self

    @staticmethod
    def translate_from_str(
        s: str, o_val: Union[Tuple[float, float, float], Vector]
    ) -> str:
        o = safe_vector(o_val)
        p = LDRPart()  # type: ignore
        if p.from_str(s) is None:
            return ""
        p.attrib.loc += o
        p.wrapcallout = False
        return str(p)

    @staticmethod
    def transform_from_str(
        s: str,
        mx_val: Optional[Matrix] = None,
        o_val: Optional[Union[Tuple[float, float, float], Vector]] = None,
        c: Optional[int] = None,
    ) -> str:
        mx = mx_val if mx_val is not None else Identity()
        o = safe_vector(o_val) if o_val is not None else Vector(0, 0, 0)  # type: ignore
        p = LDRPart()
        if p.from_str(s) is None:
            return ""
        p.attrib.loc = mx * p.attrib.loc + o  # type: ignore
        p.attrib.matrix = mx * p.attrib.matrix  # type: ignore
        p.wrapcallout = False
        if c is not None:
            p.attrib.colour = c
        return str(p)


@dataclass(eq=False)
class LDRHeader:
    title: str = ""
    file: str = ""
    name: str = ""
    author: str = ""

    def __str__(self) -> str:
        hl = []
        if self.title:
            hl.append(f"0 {self.title}")
        if self.file:
            hl.append(f"0 FILE {self.file}")
        if self.name:
            hl.append(f"0 Name: {self.name}")
        if self.author:
            hl.append(f"0 Author: {self.author}")
        return "\n".join(hl) + "\n" if hl else ""
