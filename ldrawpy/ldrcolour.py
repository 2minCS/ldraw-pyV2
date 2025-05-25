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
# LDraw colour class

import string
from typing import Tuple, Union, List, Optional, Any, Dict  # Added Dict

# Explicit imports from within the ldrawpy package
from .constants import LDR_DEF_COLOUR
from .ldrcolourdict import (
    LDR_COLOUR_RGB,
    LDR_COLOUR_NAME,
    LDR_COLOUR_TITLE,
    BL_TO_LDR_COLOUR,
    LDR_FILL_CODES,
    LDR_FILL_TITLES,
)

# Define a type alias for more complex color input
ColourInputType = Union[int, str, Tuple[float, ...], List[float], "LDRColour"]


class LDRColour(object):
    """LDraw colour helper class.  This class can be used to store a
    colour and to perform conversions among:
      LDraw colour code, Bricklink colour code, colour name,
      RGB floating point, RGB hex"""

    code: int
    r: float
    g: float
    b: float

    def __init__(self, colour: ColourInputType = LDR_DEF_COLOUR):
        self.code = LDR_DEF_COLOUR
        self.r = 0.8  # Default r, g, b if not determined otherwise
        self.g = 0.8
        self.b = 0.8

        if isinstance(colour, (tuple, list)):
            if len(colour) == 3:  # Make sure it's a 3-element tuple/list
                # Ensure elements are numbers before comparison
                if all(isinstance(c_val, (int, float)) for c_val in colour):
                    # Check if RGB values are 0-255 or 0.0-1.0
                    if any(float(c_val) > 1.0 for c_val in colour):  # type: ignore
                        self.r = min(float(colour[0]) / 255.0, 1.0)
                        self.g = min(float(colour[1]) / 255.0, 1.0)
                        self.b = min(float(colour[2]) / 255.0, 1.0)
                    else:
                        self.r = float(colour[0])
                        self.g = float(colour[1])
                        self.b = float(colour[2])

                    # Try to find matching LDR code from RGB
                    rgb_hex_str: str = self.as_hex().lower()
                    found_code: int = LDR_DEF_COLOUR
                    for code_val, hex_val in LDR_COLOUR_RGB.items():
                        if rgb_hex_str == hex_val.lower():
                            found_code = code_val
                            break
                    self.code = found_code
                else:  # Non-numeric elements in tuple/list
                    self.code_to_rgb()  # Fallback to default LDR_DEF_COLOUR's RGB
            else:  # Tuple/list not of length 3
                self.code_to_rgb()

        elif isinstance(colour, str):
            parsed_code: int = LDRColour.ColourCodeFromString(colour)
            if parsed_code != LDR_DEF_COLOUR:
                self.code = parsed_code
                self.code_to_rgb()
            elif colour.startswith("#") and (len(colour) == 7 or len(colour) == 4):
                try:
                    r_val, g_val, b_val = LDRColour.RGBFromHex(colour)
                    self.r, self.g, self.b = r_val, g_val, b_val
                    # self.code remains LDR_DEF_COLOUR for custom hex not in LDR_COLOUR_RGB
                except ValueError:
                    self.code_to_rgb()
            else:
                self.code_to_rgb()

        elif isinstance(colour, LDRColour):
            self.code = colour.code
            self.r = colour.r
            self.g = colour.g
            self.b = colour.b
        elif isinstance(colour, int):
            self.code = colour
            self.code_to_rgb()
        else:  # Fallback for other unexpected types
            self.code_to_rgb()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.code}, "
            f"r: {self.r:.2f} g: {self.g:.2f} b: {self.b:.2f}, #{self.as_hex()})"
        )

    def __str__(self) -> str:
        return LDRColour.SafeLDRColourName(self.code)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, int):
            return self.code == other
        if isinstance(other, LDRColour):
            if self.code != LDR_DEF_COLOUR and other.code != LDR_DEF_COLOUR:
                return self.code == other.code
            # For LDR_DEF_COLOUR or custom RGB, compare float values approximately
            return (
                abs(self.r - other.r) < 1e-6
                and abs(self.g - other.g) < 1e-6
                and abs(self.b - other.b) < 1e-6
            )
        return NotImplemented

    def code_to_rgb(self) -> None:
        if self.code == LDR_DEF_COLOUR:
            self.r, self.g, self.b = 0.62, 0.62, 0.62
            return

        rgb_hex: Optional[str] = LDR_COLOUR_RGB.get(self.code)  # Use .get for safety
        if rgb_hex:
            try:
                self.r, self.g, self.b = LDRColour.RGBFromHex(rgb_hex)
            except ValueError:
                self.r, self.g, self.b = 0.8, 0.8, 0.8  # Fallback for bad hex in dict
        else:  # Code not in LDR_COLOUR_RGB
            # If r,g,b are still at initial defaults, set to LDR_DEF_COLOUR's RGB
            if (
                abs(self.r - 0.8) < 1e-6
                and abs(self.g - 0.8) < 1e-6
                and abs(self.b - 0.8) < 1e-6
            ):
                self.r, self.g, self.b = 0.62, 0.62, 0.62

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.r, self.g, self.b)

    def as_bgr(self) -> Tuple[int, int, int]:
        return (int(self.b * 255), int(self.g * 255), int(self.r * 255))

    def as_hex(self) -> str:
        # Ensure r,g,b are clamped between 0 and 1 before scaling to 255
        r_clamped = max(0.0, min(1.0, self.r))
        g_clamped = max(0.0, min(1.0, self.g))
        b_clamped = max(0.0, min(1.0, self.b))
        return (
            f"{int(r_clamped * 255.0):02X}"
            f"{int(g_clamped * 255.0):02X}"
            f"{int(b_clamped * 255.0):02X}"
        )

    def ldvcode(self) -> int:
        return self.code

    def name(self) -> str:
        theName: str = LDRColour.SafeLDRColourName(self.code)
        # If code is LDR_DEF_COLOUR but RGB is custom (e.g., from hex input)
        is_default_rgb = (
            abs(self.r - 0.62) < 1e-6
            and abs(self.g - 0.62) < 1e-6
            and abs(self.b - 0.62) < 1e-6
        )
        is_initial_placeholder_rgb = (
            abs(self.r - 0.8) < 1e-6
            and abs(self.g - 0.8) < 1e-6
            and abs(self.b - 0.8) < 1e-6
        )

        if self.code == LDR_DEF_COLOUR and not (
            is_default_rgb or is_initial_placeholder_rgb
        ):
            return f"#{self.as_hex()}"
        elif not theName:  # Truly unknown code with no name
            return f"LDR Code {self.code}"
        return theName

    def high_contrast_complement(self) -> Tuple[float, float, float]:
        luminance: float = 0.299 * self.r + 0.587 * self.g + 0.114 * self.b
        return (0.0, 0.0, 0.0) if luminance > 0.5 else (1.0, 1.0, 1.0)

    def to_bricklink(self) -> int:
        for bl_code, ldr_code in BL_TO_LDR_COLOUR.items():
            if ldr_code == self.code:
                return bl_code
        return 0

    @staticmethod
    def SafeLDRColourName(ldrCode: int) -> str:
        name_val: Optional[str] = LDR_COLOUR_NAME.get(ldrCode)
        if name_val is not None:
            return name_val
        title_val: Optional[str] = LDR_COLOUR_TITLE.get(ldrCode)
        if title_val is not None:
            return title_val
        return ""

    @staticmethod
    def SafeLDRColourRGB(ldrCode: int) -> str:
        return LDR_COLOUR_RGB.get(ldrCode, "CCCCCC")  # Default hex for LDR_DEF_COLOUR

    @staticmethod
    def BLColourCodeFromLDR(ldr_code_val: int) -> int:
        for bl_code, ldr_code_map in BL_TO_LDR_COLOUR.items():
            if ldr_code_map == ldr_code_val:
                return bl_code
        return 0

    @staticmethod
    def ColourCodeFromString(colourStr: str) -> int:
        for code, name_val in LDR_COLOUR_NAME.items():
            if name_val.lower() == colourStr.lower():
                return code
        norm_colour_str = colourStr.lstrip("#").lower()
        if len(norm_colour_str) == 6 and all(
            c in string.hexdigits for c in norm_colour_str
        ):
            for code, rgb_hex_val in LDR_COLOUR_RGB.items():
                if norm_colour_str == rgb_hex_val.lower():
                    return code
        return LDR_DEF_COLOUR

    @staticmethod
    def RGBFromHex(hexStr: str) -> Tuple[float, float, float]:
        hs = hexStr.lstrip("#")
        if not (len(hs) == 6 and all(c in string.hexdigits for c in hs)):
            if len(hs) == 3 and all(c in string.hexdigits for c in hs):
                hs = "".join([c * 2 for c in hs])
            else:
                raise ValueError(f"Invalid hex string: {hexStr}")

        r_val: float = int(hs[0:2], 16) / 255.0
        g_val: float = int(hs[2:4], 16) / 255.0
        b_val: float = int(hs[4:6], 16) / 255.0
        return (r_val, g_val, b_val)


def FillColoursFromLDRCode(ldrCode: int) -> List[Tuple[float, float, float]]:
    fill_colour_hex_list: List[str] = []
    if ldrCode in LDR_COLOUR_RGB:
        fill_colour_hex_list.append(LDR_COLOUR_RGB[ldrCode])
    elif ldrCode in LDR_FILL_CODES:
        fill_colour_hex_list = LDR_FILL_CODES[
            ldrCode
        ]  # LDR_FILL_CODES values are List[str]

    rgb_tuples: List[Tuple[float, float, float]] = []
    for hex_val in fill_colour_hex_list:
        try:
            rgb_tuples.append(LDRColour.RGBFromHex(hex_val))
        except ValueError:
            pass
    return rgb_tuples


def FillTitlesFromLDRCode(ldrCode: int) -> List[str]:
    fill_titles_list: List[str] = []
    if ldrCode in LDR_COLOUR_NAME:
        fill_titles_list.append(LDR_COLOUR_NAME[ldrCode])
    elif ldrCode in LDR_FILL_TITLES:
        fill_titles_list = LDR_FILL_TITLES[
            ldrCode
        ]  # LDR_FILL_TITLES values are List[str]
    return fill_titles_list
