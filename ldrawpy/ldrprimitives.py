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
ldrprimitives.py: Defines fundamental LDraw geometric primitives and attributes.

This module contains classes for representing LDraw attributes (colour, location,
matrix), basic geometric shapes (lines, triangles, quads), LDraw parts (type 1 lines),
and LDraw file headers (type 0 lines for metadata).
"""

import hashlib

# from functools import reduce # REMOVED - Not used in this file
from typing import List, Tuple, Union, Optional, Sequence  # Added Sequence
from dataclasses import dataclass, field

# Explicit imports from toolbox
from toolbox import Vector, Matrix, Identity, euler_to_rot_matrix, safe_vector  # type: ignore

# Explicit imports from .ldrhelpers (relative import within the package)
from .ldrhelpers import vector_str, mat_str, quantize
from .constants import LDR_DEF_COLOUR


@dataclass(eq=False)  # eq=False because custom __eq__ is provided
class LDRAttrib:
    """
    Represents common LDraw attributes like colour, transformation matrix, and location.

    This dataclass is used by other LDraw primitives to store their visual and
    positional properties.

    Attributes:
        colour: LDraw color code (integer).
        units: Units for coordinates (e.g., "ldu", "mm"). Default is "ldu".
        loc: A toolbox.Vector representing the location (x, y, z).
        matrix: A toolbox.Matrix representing the rotation and scale transformation.
    """

    colour: int = LDR_DEF_COLOUR
    units: str = "ldu"
    loc: Vector = field(default_factory=lambda: Vector(0, 0, 0))  # type: ignore
    matrix: Matrix = field(default_factory=Identity)  # type: ignore

    def __post_init__(self):
        """Ensures colour is an integer after initialization by @dataclass."""
        self.colour = int(self.colour)

    def __eq__(self, other: object) -> bool:
        """
        Compares this LDRAttrib with another for equality.

        Equality is based on colour, units, and near-equality (within tolerance)
        of the location Vector and transformation Matrix.

        Args:
            other: The object to compare against.

        Returns:
            True if the objects are considered equal, False otherwise.
            NotImplemented if the other object is not an LDRAttrib instance.
        """
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            self.colour == other.colour
            and self.units == other.units
            and self.loc.almost_same_as(other.loc)  # type: ignore
            and self.matrix.is_almost_same_as(other.matrix)  # type: ignore
        )

    def copy(self) -> "LDRAttrib":
        """Creates a deep copy of this LDRAttrib instance."""
        # Ensure Vector and Matrix are copied if they have a copy method
        new_loc = self.loc.copy() if hasattr(self.loc, "copy") else self.loc
        new_matrix = self.matrix.copy() if hasattr(self.matrix, "copy") else self.matrix
        return LDRAttrib(
            colour=self.colour, units=self.units, loc=new_loc, matrix=new_matrix
        )


class LDRLine:
    """
    Represents an LDraw line primitive (type 2).

    A line is defined by two points (p1, p2) and its attributes (colour, units).

    Attributes:
        attrib: LDRAttrib object holding colour and unit information.
        p1: Start Vector of the line.
        p2: End Vector of the line.
    """

    __slots__ = ["attrib", "p1", "p2"]  # Memory optimization
    attrib: LDRAttrib
    p1: Vector  # type: ignore
    p2: Vector  # type: ignore

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        """
        Initializes an LDRLine.

        Args:
            colour: LDraw color code for the line. Defaults to LDR_DEF_COLOUR.
            units: Units for the line's coordinates. Defaults to "ldu".
        """
        self.attrib = LDRAttrib(colour, units)
        self.p1, self.p2 = Vector(0, 0, 0), Vector(0, 0, 0)  # type: ignore

    def __str__(self) -> str:
        """Returns the LDraw string representation of the line (type 2 line)."""
        return (
            f"2 {self.attrib.colour} "
            f"{vector_str(self.p1, self.attrib)}{vector_str(self.p2, self.attrib)}\n"
        )

    def translate(self, offset: Vector):  # type: ignore
        """Translates the line by a given offset Vector."""
        self.p1 += offset
        self.p2 += offset

    def transform(self, matrix: Matrix):  # type: ignore
        """Transforms the line's points by a given transformation Matrix."""
        self.p1 = self.p1 * matrix
        self.p2 = self.p2 * matrix


class LDRTriangle:
    """
    Represents an LDraw triangle primitive (type 3).

    A triangle is defined by three points (p1, p2, p3) and its attributes.

    Attributes:
        attrib: LDRAttrib object.
        p1, p2, p3: Vector objects representing the vertices of the triangle.
    """

    __slots__ = ["attrib", "p1", "p2", "p3"]
    attrib: LDRAttrib
    p1: Vector  # type: ignore
    p2: Vector  # type: ignore
    p3: Vector  # type: ignore

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        """
        Initializes an LDRTriangle.

        Args:
            colour: LDraw color code.
            units: Coordinate units.
        """
        self.attrib = LDRAttrib(colour, units)
        self.p1, self.p2, self.p3 = Vector(0, 0, 0), Vector(0, 0, 0), Vector(0, 0, 0)  # type: ignore

    def __str__(self) -> str:
        """Returns the LDraw string representation of the triangle (type 3 line)."""
        return (
            f"3 {self.attrib.colour} "
            f"{vector_str(self.p1, self.attrib)}{vector_str(self.p2, self.attrib)}"
            f"{vector_str(self.p3, self.attrib)}\n"
        )

    def translate(self, offset: Vector):  # type: ignore
        """Translates the triangle by a given offset Vector."""
        self.p1 += offset
        self.p2 += offset
        self.p3 += offset

    def transform(self, matrix: Matrix):  # type: ignore
        """Transforms the triangle's vertices by a given transformation Matrix."""
        self.p1 *= matrix
        self.p2 *= matrix
        self.p3 *= matrix


class LDRQuad:
    """
    Represents an LDraw quadrilateral primitive (type 4).

    A quadrilateral is defined by four points (p1, p2, p3, p4) and its attributes.
    The points should be coplanar and ordered (e.g., counter-clockwise).

    Attributes:
        attrib: LDRAttrib object.
        p1, p2, p3, p4: Vector objects representing the vertices.
    """

    __slots__ = ["attrib", "p1", "p2", "p3", "p4"]
    attrib: LDRAttrib
    p1: Vector  # type: ignore
    p2: Vector  # type: ignore
    p3: Vector  # type: ignore
    p4: Vector  # type: ignore

    def __init__(self, colour: int = LDR_DEF_COLOUR, units: str = "ldu"):
        """
        Initializes an LDRQuad.

        Args:
            colour: LDraw color code.
            units: Coordinate units.
        """
        self.attrib = LDRAttrib(colour, units)
        self.p1, self.p2, self.p3, self.p4 = Vector(0, 0, 0), Vector(0, 0, 0), Vector(0, 0, 0), Vector(0, 0, 0)  # type: ignore

    def __str__(self) -> str:
        """Returns the LDraw string representation of the quadrilateral (type 4 line)."""
        return (
            f"4 {self.attrib.colour} "
            f"{vector_str(self.p1, self.attrib)}{vector_str(self.p2, self.attrib)}"
            f"{vector_str(self.p3, self.attrib)}{vector_str(self.p4, self.attrib)}\n"
        )

    def translate(self, offset: Vector):  # type: ignore
        """Translates the quadrilateral by a given offset Vector."""
        self.p1 += offset
        self.p2 += offset
        self.p3 += offset
        self.p4 += offset

    def transform(self, matrix: Matrix):  # type: ignore
        """Transforms the quadrilateral's vertices by a given transformation Matrix."""
        self.p1 *= matrix
        self.p2 *= matrix
        self.p3 *= matrix
        self.p4 *= matrix


class LDRPart:
    """
    Represents an LDraw part or submodel reference (type 1 line).

    This class stores the part's name, colour, location, and transformation matrix.
    It can also handle wrapping with LPUB CALLOUT meta-commands.

    Attributes:
        attrib: LDRAttrib object containing colour, location, matrix, and units.
        name: The filename of the part (e.g., "3001.dat" or "submodel.ldr").
        wrapcallout: Boolean indicating if this part reference should be wrapped
                     in LPUB CALLOUT BEGIN/END meta-commands.
    """

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
        """
        Initializes an LDRPart.

        Args:
            colour: LDraw color code for the part.
            name: Filename of the part (e.g., "3001", "3001.dat").
                  The ".dat" or ".ldr" extension will be appended if missing.
            units: Coordinate units.
        """
        self.attrib = LDRAttrib(colour, units)
        self.name = name if name is not None else ""
        self.wrapcallout = False

    def __str__(self) -> str:
        """
        Returns the LDraw string representation of the part (type 1 line).
        Includes .dat or .ldr extension if missing and LPUB CALLOUT wrappers if enabled.
        """
        matrix_elements: Sequence[float] = []  # Use Sequence for broader compatibility
        # Ensure matrix is correctly formed for mat_str
        if hasattr(self.attrib.matrix, "rows") and self.attrib.matrix.rows:  # type: ignore
            # Flatten the matrix rows into a single list of 9 elements
            matrix_elements = [
                item for sublist in self.attrib.matrix.rows for item in sublist  # type: ignore
            ]
        else:  # Default to identity matrix string components if matrix is invalid
            matrix_elements = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        # Ensure matrix_elements has 9 items for mat_str
        if len(matrix_elements) != 9:  # Fallback if flattening failed
            matrix_elements = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

        tup_matrix_str = mat_str(tuple(matrix_elements))  # mat_str expects a tuple

        name_to_write = self.name
        if name_to_write:  # Ensure filename has an extension
            ext = name_to_write[-4:].lower()
            if not (ext == ".ldr" or ext == ".dat"):
                # Assume .dat if not .ldr, common for parts
                name_to_write += ".dat"
        else:  # Default to a placeholder if name is empty
            name_to_write = "unknown.dat"

        s = (
            f"1 {self.attrib.colour} "
            f"{vector_str(self.attrib.loc, self.attrib)}"
            f"{tup_matrix_str}"
            f"{name_to_write}\n"
        )
        if self.wrapcallout and name_to_write.endswith(
            ".ldr"
        ):  # Callouts usually for submodels
            return f"0 !LPUB CALLOUT BEGIN\n{s}0 !LPUB CALLOUT END\n"
        return s

    def __eq__(self, other: object) -> bool:
        """
        Compares this LDRPart with another for basic equality.
        Equality is based on part name and colour only. For a more comprehensive
        comparison (including location and matrix), use `is_identical`.
        """
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.name == other.name and self.attrib.colour == other.attrib.colour

    def copy(self) -> "LDRPart":
        """Creates a deep copy of this LDRPart instance."""
        p = LDRPart(colour=self.attrib.colour, name=self.name, units=self.attrib.units)
        p.wrapcallout = self.wrapcallout
        p.attrib = self.attrib.copy()  # Deep copy of LDRAttrib
        return p

    def sha1hash(self) -> str:
        """Generates a SHA1 hash of the part's string representation."""
        shash = hashlib.sha1()
        shash.update(bytes(str(self), encoding="utf8"))
        return shash.hexdigest()

    def is_identical(self, other: "LDRPart") -> bool:
        """
        Checks if this LDRPart is identical to another, including name,
        colour, location, and transformation matrix.
        """
        return self.name == other.name and self.attrib == other.attrib

    def is_same(
        self,
        other: "LDRPart",
        ignore_location: bool = False,
        ignore_colour: bool = False,
        exact: bool = False,  # Kept for compatibility, but is_identical is clearer for exactness
    ) -> bool:
        """
        Checks if this LDRPart is the same as another, with options to ignore
        location and/or colour.

        Args:
            other: The LDRPart to compare against.
            ignore_location: If True, location is not considered in comparison.
            ignore_colour: If True, colour is not considered.
            exact: If True, performs an identical check (equivalent to `is_identical`).

        Returns:
            True if parts are considered the same based on criteria, False otherwise.
        """
        if self.name != other.name:
            return False
        if exact:  # If exact is True, defer to is_identical
            return self.is_identical(other)
        if not ignore_colour and self.attrib.colour != other.attrib.colour:
            return False
        if not ignore_location and not self.attrib.loc.almost_same_as(other.attrib.loc):  # type: ignore
            return False
        # If we reach here, name matches, and color/location match or are ignored.
        # Note: This doesn't check the matrix by default unless `exact` is True.
        return True

    def is_coaligned(self, other: "LDRPart") -> bool:
        """
        Checks if this LDRPart is co-aligned with another.
        (Relies on toolbox.Vector.is_colinear_with, which might need specific definition).
        """
        try:
            # This logic assumes that the product of location and matrix gives a
            # meaningful vector for colinearity checks. The interpretation might vary.
            v1 = self.attrib.loc * self.attrib.matrix  # type: ignore
            v2 = other.attrib.loc * other.attrib.matrix  # type: ignore
            if hasattr(v1, "is_colinear_with"):
                # is_colinear_with might return an integer indicating type of colinearity
                return (
                    v1.is_colinear_with(v2) == 2
                )  # Assuming 2 means perfectly co-aligned
            else:  # Fallback if method doesn't exist
                return False
        except AttributeError:  # If Vector or Matrix doesn't support ops as expected
            return False
        # Consider adding a tolerance if comparing floating point vectors.

    def change_colour(self, to_colour: int):
        """Changes the colour of the part."""
        self.attrib.colour = int(to_colour)

    def set_rotation(self, angle: Union[Tuple[float, float, float], Vector]):  # type: ignore
        """
        Sets the part's rotation matrix based on Euler angles.

        Args:
            angle: A tuple or Vector of Euler angles (degrees or radians,
                   depending on euler_to_rot_matrix implementation).
        """
        self.attrib.matrix = euler_to_rot_matrix(angle)  # type: ignore

    def move_to(self, pos: Union[Tuple[float, float, float], Vector]):  # type: ignore
        """
        Moves the part to an absolute location.

        Args:
            pos: A tuple or Vector representing the new absolute location.
        """
        self.attrib.loc = safe_vector(pos)  # type: ignore

    def move_by(self, offset: Union[Tuple[float, float, float], Vector]):  # type: ignore
        """
        Moves the part by a relative offset.

        Args:
            offset: A tuple or Vector representing the offset to apply.
        """
        self.attrib.loc += safe_vector(offset)  # type: ignore

    def rotate_by(self, angle: Union[Tuple[float, float, float], Vector]):  # type: ignore
        """
        Rotates the part by given Euler angles relative to its current orientation.
        This also rotates its current location vector around the origin.

        Args:
            angle: A tuple or Vector of Euler angles for relative rotation.
        """
        rotation_matrix = euler_to_rot_matrix(angle)  # type: ignore
        # Rotate the part's location vector around the origin
        self.attrib.loc = rotation_matrix * self.attrib.loc  # type: ignore
        # Concatenate the new rotation with the existing transformation matrix
        self.attrib.matrix = rotation_matrix * self.attrib.matrix  # type: ignore

    def transform(self, matrix: Matrix = Identity(), offset: Vector = Vector(0, 0, 0)):  # type: ignore
        """
        Applies a full transformation (matrix and offset) to the part.
        The matrix is applied first, then the offset.

        Args:
            matrix: The transformation Matrix to apply. Defaults to Identity.
            offset: The offset Vector to apply after matrix transformation. Defaults to (0,0,0).
        """
        # Transform location: M * current_loc + offset
        self.attrib.loc = matrix * self.attrib.loc + offset  # type: ignore
        # Concatenate matrices: M_new = M_applied * M_current
        self.attrib.matrix = matrix * self.attrib.matrix  # type: ignore

    def from_str(self, s: str) -> Optional["LDRPart"]:
        """
        Parses an LDraw type 1 line string and populates this LDRPart instance.

        Args:
            s: The LDraw string for a type 1 line.

        Returns:
            Self if parsing is successful, None otherwise.
        """
        sl = s.strip().lower().split()
        if not len(sl) >= 14:  # Minimum tokens for a valid type 1 line
            return None
        try:
            if int(sl[0]) != 1:  # Must be a type 1 line
                return None
            self.attrib.colour = int(sl[1])
            # Parse location
            self.attrib.loc.x = quantize(sl[2])
            self.attrib.loc.y = quantize(sl[3])
            self.attrib.loc.z = quantize(sl[4])
            # Parse matrix (9 elements)
            m_elements = [quantize(sl[i]) for i in range(5, 14)]
            self.attrib.matrix = Matrix([m_elements[0:3], m_elements[3:6], m_elements[6:9]])  # type: ignore
            # Part name is the rest of the line
            self.name = " ".join(sl[14:]).replace(".dat", "") if len(sl) > 14 else ""
            # Note: Original code had `if sl[14:] else ""`, changed to check len(sl)
            # Also, .dat is stripped here, but __str__ adds it back. Consider consistency.
        except (
            ValueError,
            IndexError,
        ):  # Catch errors from int/float conversion or list indexing
            return None
        return self

    @staticmethod
    def translate_from_str(
        s: str, offset_val: Union[Tuple[float, float, float], Vector]  # type: ignore
    ) -> str:
        """
        Parses an LDRPart from a string, applies a translation, and returns its new string form.

        Args:
            s: The LDraw string for a type 1 line.
            offset_val: The translation offset (tuple or Vector).

        Returns:
            The LDraw string of the translated part, or an empty string if parsing failed.
        """
        offset_vector = safe_vector(offset_val)  # type: ignore
        part_instance = LDRPart()
        if part_instance.from_str(s) is None:  # Populate part_instance from string
            return ""
        part_instance.attrib.loc += offset_vector  # Apply translation
        part_instance.wrapcallout = (
            False  # Ensure no callout wrapping for this static method
        )
        return str(part_instance)

    @staticmethod
    def transform_from_str(
        s: str,
        matrix_val: Optional[Matrix] = None,  # type: ignore
        offset_val: Optional[Union[Tuple[float, float, float], Vector]] = None,  # type: ignore
        new_colour: Optional[int] = None,
    ) -> str:
        """
        Parses an LDRPart from a string, applies a transformation (matrix and/or offset),
        optionally changes its colour, and returns its new string form.

        Args:
            s: The LDraw string for a type 1 line.
            matrix_val: Optional transformation Matrix to apply. Defaults to Identity.
            offset_val: Optional offset Vector to apply. Defaults to (0,0,0).
            new_colour: Optional new LDraw color code.

        Returns:
            The LDraw string of the transformed part, or an empty string if parsing failed.
        """
        transform_matrix = matrix_val if matrix_val is not None else Identity()  # type: ignore
        offset_vector = safe_vector(offset_val) if offset_val is not None else Vector(0, 0, 0)  # type: ignore

        part_instance = LDRPart()
        if part_instance.from_str(s) is None:  # Populate from string
            return ""

        # Apply transformation (matrix first, then offset)
        part_instance.attrib.loc = transform_matrix * part_instance.attrib.loc + offset_vector  # type: ignore
        part_instance.attrib.matrix = transform_matrix * part_instance.attrib.matrix  # type: ignore

        part_instance.wrapcallout = False  # Ensure no callout wrapping
        if new_colour is not None:
            part_instance.attrib.colour = int(new_colour)
        return str(part_instance)


@dataclass(eq=False)  # eq=False as __str__ is primary, and content defines identity
class LDRHeader:
    """
    Represents LDraw file header information (type 0 meta-commands).

    Common header lines include title, filename, author, etc.

    Attributes:
        title: The main title of the LDraw file (0 <title>).
        file: The declared filename within the LDraw file (0 FILE <filename>).
        name: The "Name:" field often used for description (0 Name: <name>).
        author: The author of the file (0 Author: <author>).
    """

    title: str = ""
    file: str = ""  # Corresponds to "0 FILE filename.ldr"
    name: str = ""  # Corresponds to "0 Name: Descriptive Name"
    author: str = ""

    def __str__(self) -> str:
        """Returns the LDraw string representation of the header lines."""
        header_lines: List[str] = []
        if self.title:  # This is usually the first line if not a "0 FILE" directive
            header_lines.append(f"0 {self.title}")
        if self.file:  # Standard "0 FILE" directive
            header_lines.append(f"0 FILE {self.file}")
        if self.name:  # Common "0 Name:" directive
            header_lines.append(f"0 Name: {self.name}")
        if self.author:  # Common "0 Author:" directive
            header_lines.append(f"0 Author: {self.author}")

        # Join with newlines and ensure a trailing newline if there are any header lines
        return "\n".join(header_lines) + "\n" if header_lines else ""
