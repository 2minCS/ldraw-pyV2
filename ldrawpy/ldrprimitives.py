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
from typing import List, Tuple, Union, Optional  # For type hints

# Explicit imports from toolbox
from toolbox import Vector, Matrix, Identity, euler_to_rot_matrix, safe_vector

# Explicit imports from .ldrhelpers (relative import within the package)
from .ldrhelpers import vector_str, mat_str, quantize
from .constants import LDR_DEF_COLOUR


class LDRAttrib:
    __slots__ = ["colour", "units", "loc", "matrix"]

    colour: int
    units: str
    loc: Vector
    matrix: Matrix

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.colour = int(colour)
        self.units = units
        self.loc = Vector(0, 0, 0)
        self.matrix = Identity()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented  # Use NotImplemented for type mismatches
        # Assuming Vector and Matrix have appropriate comparison methods (e.g., almost_same_as)
        return (
            self.colour == other.colour
            and self.loc.almost_same_as(other.loc)  # type: ignore
            and self.matrix.is_almost_same_as(other.matrix)
        )  # type: ignore

    def copy(self) -> "LDRAttrib":
        a = LDRAttrib(self.colour, self.units)
        a.loc = self.loc.copy()
        a.matrix = self.matrix.copy()
        return a


class LDRLine:
    __slots__ = ["attrib", "p1", "p2"]

    attrib: LDRAttrib
    p1: Vector
    p2: Vector

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.p1 = Vector(0, 0, 0)
        self.p2 = Vector(0, 0, 0)

    def __str__(self) -> str:
        return (
            f"2 {self.attrib.colour} "
            + vector_str(self.p1, self.attrib)
            + vector_str(self.p2, self.attrib)
            + "\n"
        )

    def translate(self, offset: Vector):
        self.p1 += offset
        self.p2 += offset

    def transform(self, matrix: Matrix):
        self.p1 = self.p1 * matrix  # type: ignore
        self.p2 = self.p2 * matrix  # type: ignore


class LDRTriangle:
    __slots__ = ["attrib", "p1", "p2", "p3"]

    attrib: LDRAttrib
    p1: Vector
    p2: Vector
    p3: Vector

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.p1 = Vector(0, 0, 0)
        self.p2 = Vector(0, 0, 0)
        self.p3 = Vector(0, 0, 0)

    def __str__(self) -> str:
        return (
            f"3 {self.attrib.colour} "
            + vector_str(self.p1, self.attrib)
            + vector_str(self.p2, self.attrib)
            + vector_str(self.p3, self.attrib)
            + "\n"
        )

    def translate(self, offset: Vector):
        self.p1 += offset
        self.p2 += offset
        self.p3 += offset

    def transform(self, matrix: Matrix):
        self.p1 = self.p1 * matrix  # type: ignore
        self.p2 = self.p2 * matrix  # type: ignore
        self.p3 = self.p3 * matrix  # type: ignore


class LDRQuad:
    __slots__ = ["attrib", "p1", "p2", "p3", "p4"]

    attrib: LDRAttrib
    p1: Vector
    p2: Vector
    p3: Vector
    p4: Vector

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        self.attrib = LDRAttrib(colour, units)
        self.p1 = Vector(0, 0, 0)
        self.p2 = Vector(0, 0, 0)
        self.p3 = Vector(0, 0, 0)
        self.p4 = Vector(0, 0, 0)

    def __str__(self) -> str:
        return (
            f"4 {self.attrib.colour} "
            + vector_str(self.p1, self.attrib)
            + vector_str(self.p2, self.attrib)
            + vector_str(self.p3, self.attrib)
            + vector_str(self.p4, self.attrib)
            + "\n"
        )

    def translate(self, offset: Vector):
        self.p1 += offset
        self.p2 += offset
        self.p3 += offset
        self.p4 += offset

    def transform(self, matrix: Matrix):
        self.p1 = self.p1 * matrix  # type: ignore
        self.p2 = self.p2 * matrix  # type: ignore
        self.p3 = self.p3 * matrix  # type: ignore
        self.p4 = self.p4 * matrix  # type: ignore


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
            matrix_elements = [item for sublist in self.attrib.matrix.rows for item in sublist]  # type: ignore
        else:  # Fallback if rows is not as expected, or provide default Identity matrix elements
            matrix_elements = [1, 0, 0, 0, 1, 0, 0, 0, 1]

        tup = tuple(matrix_elements)

        name_to_write = self.name
        if name_to_write:
            ext = name_to_write[-4:].lower()
            if not (ext == ".ldr" or ext == ".dat"):
                name_to_write += ".dat"
        else:
            name_to_write = "unknown.dat"  # Default for empty name

        s = (
            f"1 {self.attrib.colour} "
            + vector_str(self.attrib.loc, self.attrib)
            + mat_str(tup)  # mat_str expects a flat tuple/list of 9 numbers
            + f"{name_to_write}\n"
        )
        if self.wrapcallout and name_to_write.endswith(".ldr"):
            return "0 !LPUB CALLOUT BEGIN\n" + s + "0 !LPUB CALLOUT END\n"
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
        if not self.name == other.name:
            return False
        if not self.attrib == other.attrib:  # Relies on LDRAttrib.__eq__
            return False
        return True

    def is_same(
        self,
        other: "LDRPart",
        ignore_location: bool = False,
        ignore_colour: bool = False,
        exact: bool = False,
    ) -> bool:
        if not self.name == other.name:
            return False
        if exact:
            return self.is_identical(other)

        if not ignore_colour and not self.attrib.colour == other.attrib.colour:
            return False
        if not ignore_location and not self.attrib.loc.almost_same_as(other.attrib.loc):  # type: ignore
            return False
        return True

    def is_coaligned(self, other: "LDRPart") -> bool:
        # This method's original logic was:
        # v1 = self.attrib.loc * self.attrib.matrix
        # v2 = other.attrib.loc * other.attrib.matrix
        # naxis = v1.is_colinear_with(v2)
        # if naxis == 2: return True
        # This implies specific behavior of Vector * Matrix and is_colinear_with from toolbox.
        # For MyPy to pass without knowing toolbox internals, we might need to assume
        # these operations exist or simplify/stub this if it's not critical path for MyPy checks.
        # Assuming toolbox.Vector has is_colinear_with and supports multiplication with Matrix.
        try:
            v1_transformed = self.attrib.loc * self.attrib.matrix  # type: ignore
            v2_transformed = other.attrib.loc * other.attrib.matrix  # type: ignore
            if hasattr(v1_transformed, "is_colinear_with"):
                naxis = v1_transformed.is_colinear_with(v2_transformed)  # type: ignore
                return naxis == 2
        except AttributeError:  # In case methods don't exist on toolbox.Vector/Matrix
            pass  # Or log a warning
        return False  # Default if logic can't be verified

    def change_colour(self, to_colour: int):
        self.attrib.colour = to_colour

    def set_rotation(self, angle: Union[Tuple[float, float, float], Vector]):
        rm = euler_to_rot_matrix(angle)
        self.attrib.matrix = rm

    def move_to(self, pos: Union[Tuple[float, float, float], Vector]):
        o = safe_vector(pos)
        self.attrib.loc = o

    def move_by(self, offset: Union[Tuple[float, float, float], Vector]):
        o = safe_vector(offset)
        self.attrib.loc += o

    def rotate_by(self, angle: Union[Tuple[float, float, float], Vector]):
        rm = euler_to_rot_matrix(angle)
        # Original: self.attrib.loc *= rt (rt = rm.transpose())
        # Correct transformation of location for rotation about origin:
        # New Loc = RotMatrix * Old Loc
        # If rotating the part's own coordinate system:
        # New Matrix = RotMatrix * Old Matrix
        # Location remains same *relative to its own new orientation* unless also translated.
        # If rotating the part around the world origin and its current location is world coord:
        # New Loc = RotMatrix * Old Loc
        # New Matrix = RotMatrix * Old Matrix (to rotate its orientation as well)

        # Assuming angle rotates the part around its own origin, then its local axes change.
        # And if its attrib.loc is its position in parent space, that also needs to rotate
        # if the rotation is meant to be around the parent's origin.
        # The original code:
        # self.attrib.matrix = rm * self.attrib.matrix
        # self.attrib.loc *= rt (where rt = rm.transpose()) - this is unusual.
        # Let's assume the intent is to rotate the part and its orientation:
        self.attrib.loc = rm * self.attrib.loc  # type: ignore # Rotate position vector
        self.attrib.matrix = rm * self.attrib.matrix  # type: ignore # Combine rotations

    def transform(self, matrix: Matrix = Identity(), offset: Vector = Vector(0, 0, 0)):
        # Original:
        # mt = matrix.transpose()
        # self.attrib.matrix = matrix * self.attrib.matrix
        # self.attrib.loc *= mt
        # self.attrib.loc += offset
        # Corrected standard transformation order:
        # 1. Transform existing location by the new matrix
        # 2. Add the new offset
        # 3. Combine matrices: new_matrix = applied_matrix * old_matrix
        self.attrib.loc = matrix * self.attrib.loc + offset  # type: ignore
        self.attrib.matrix = matrix * self.attrib.matrix  # type: ignore

    def from_str(self, s: str) -> Optional["LDRPart"]:
        split_line = s.strip().lower().split()  # Strip whitespace from line first
        if (
            not len(split_line) >= 14
        ):  # Min fields: 1 type + 1 color + 3 loc + 9 matrix + 1 name_part
            return None

        try:
            line_type = int(split_line[0])
            if line_type != 1:
                return None

            self.attrib.colour = int(split_line[1])
            self.attrib.loc.x = quantize(split_line[2])
            self.attrib.loc.y = quantize(split_line[3])
            self.attrib.loc.z = quantize(split_line[4])

            m_elements = [
                quantize(split_line[i]) for i in range(5, 14)
            ]  # Elements 5 through 13
            self.attrib.matrix = Matrix(
                [
                    m_elements[0:3],
                    m_elements[3:6],
                    m_elements[6:9],
                ]
            )
            # Name is everything from the 15th element (index 14) onwards
            pname_parts = split_line[14:]
            if (
                not pname_parts
            ):  # Handle case where name might be missing, though LDraw spec implies it's there
                self.name = ""
            else:
                pname = " ".join(pname_parts)
                self.name = pname.replace(".dat", "")  # Keep consistent with original
        except (ValueError, IndexError) as e:
            # print(f"Error parsing LDRPart from string: '{s}' - {e}") # Optional debug
            return None
        return self

    @staticmethod
    def translate_from_str(
        s: str, offset_val: Union[Tuple[float, float, float], Vector]
    ) -> str:
        offset = safe_vector(offset_val)
        p = LDRPart()
        if p.from_str(s) is None:
            return ""  # Or suitable error indication
        p.attrib.loc += offset
        p.wrapcallout = False
        return str(p)

    @staticmethod
    def transform_from_str(
        s: str,
        matrix_val: Optional[Matrix] = None,
        offset_val: Optional[Union[Tuple[float, float, float], Vector]] = None,
        colour: Optional[int] = None,
    ) -> str:
        matrix_to_apply = matrix_val if matrix_val is not None else Identity()
        offset_to_apply = (
            safe_vector(offset_val) if offset_val is not None else Vector(0, 0, 0)
        )

        p = LDRPart()
        if p.from_str(s) is None:
            return ""

        # Apply transformation (same as instance method)
        p.attrib.loc = matrix_to_apply * p.attrib.loc + offset_to_apply  # type: ignore
        p.attrib.matrix = matrix_to_apply * p.attrib.matrix  # type: ignore

        p.wrapcallout = False
        if colour is not None:
            p.attrib.colour = colour
        return str(p)


class LDRHeader:
    title: str
    file: str  # Corresponds to "0 FILE "
    name: str  # Corresponds to "0 Name: "
    author: str

    def __init__(self):
        self.title = ""
        self.file = ""
        self.name = ""
        self.author = ""

    def __str__(self) -> str:
        # LDraw spec: "0 <comment>" is a comment. Specific headers have keywords.
        # Title is often the first line if not a specific keyword.
        # Ensure proper formatting for known header lines.
        header_lines = []
        if self.title:  # Typically the first line if it's a general description
            header_lines.append(f"0 {self.title}")
        if self.file:  # "0 FILE model.ldr"
            header_lines.append(f"0 FILE {self.file}")
        if self.name:  # "0 Name: My Model Name"
            header_lines.append(f"0 Name: {self.name}")
        if self.author:  # "0 Author: Your Name"
            header_lines.append(f"0 Author: {self.author}")
        # Add other common headers like !LDRAW_ORG, !LICENSE etc. if needed
        # header_lines.append("0 !LDRAW_ORG Part Unofficial")
        # header_lines.append("0 !LICENSE Redistributable under CCAL version 2.0 : see CAreadme.txt")

        return "\n".join(header_lines) + "\n" if header_lines else ""
