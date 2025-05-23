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
from typing import Tuple, Union, List, Optional, Any  # For type hints

# Explicit imports from within the ldrawpy package
from .constants import LDR_DEF_COLOUR  # Import LDR_DEF_COLOUR
from .ldrcolourdict import (  # Import necessary dictionaries
    LDR_COLOUR_RGB,
    LDR_COLOUR_NAME,
    LDR_COLOUR_TITLE,
    BL_TO_LDR_COLOUR,
    LDR_FILL_CODES,  # Used by FillColoursFromLDRCode
    LDR_FILL_TITLES,  # Used by FillTitlesFromLDRCode
)


class LDRColour(object):
    """LDraw colour helper class.  This class can be used to store a
    colour and to perform conversions among:
      LDraw colour code, Bricklink colour code, colour name,
      RGB floating point, RGB hex"""

    code: int
    r: float
    g: float
    b: float

    def __init__(
        self,
        colour: Union[
            int, str, Tuple[float, ...], List[float], "LDRColour"
        ] = LDR_DEF_COLOUR,
    ):
        self.code = LDR_DEF_COLOUR  # Default
        self.r = 0.8
        self.g = 0.8
        self.b = 0.8

        if isinstance(colour, (tuple, list)):
            if len(colour) == 3:
                # Check if RGB values are 0-255 or 0.0-1.0
                if any(c > 1.0 for c in colour if isinstance(c, (int, float))):  # type: ignore
                    self.r = min(float(colour[0]) / 255.0, 1.0)
                    self.g = min(float(colour[1]) / 255.0, 1.0)
                    self.b = min(float(colour[2]) / 255.0, 1.0)
                else:
                    self.r = float(colour[0])
                    self.g = float(colour[1])
                    self.b = float(colour[2])

                # Try to find matching LDR code from RGB
                rgb_hex_str = self.as_hex().lower()
                found_code = LDR_DEF_COLOUR
                for code_val, hex_val in LDR_COLOUR_RGB.items():
                    if rgb_hex_str == hex_val.lower():
                        found_code = code_val
                        break
                self.code = found_code  # Assign found code or keep LDR_DEF_COLOUR
            else:
                # Invalid tuple/list, retain default LDR_DEF_COLOUR and its RGB
                self.code_to_rgb()  # Sets r,g,b for LDR_DEF_COLOUR

        elif isinstance(colour, str):
            # Try parsing as LDraw colour name or hex string
            parsed_code = LDRColour.ColourCodeFromString(colour)
            if parsed_code != LDR_DEF_COLOUR:  # Found by name or hex in LDR_COLOUR_RGB
                self.code = parsed_code
                self.code_to_rgb()
            elif colour.startswith("#") and (
                len(colour) == 7 or len(colour) == 4
            ):  # Check for hex string
                try:
                    r_val, g_val, b_val = LDRColour.RGBFromHex(colour)
                    self.r, self.g, self.b = r_val, g_val, b_val
                    # self.code remains LDR_DEF_COLOUR as it's a custom hex
                except ValueError:  # Invalid hex
                    self.code_to_rgb()  # Default
            else:  # Unknown string
                self.code_to_rgb()  # Default

        elif isinstance(colour, LDRColour):  # If LDRColour instance is passed
            self.code = colour.code
            self.r = colour.r
            self.g = colour.g
            self.b = colour.b
        elif isinstance(colour, int):  # If integer (LDraw code)
            self.code = colour
            self.code_to_rgb()
        else:  # Fallback for other types
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
            # If both have non-default codes, compare codes
            if self.code != LDR_DEF_COLOUR and other.code != LDR_DEF_COLOUR:
                return self.code == other.code
            # Otherwise, compare RGB values (approximately)
            return (
                abs(self.r - other.r) < 1e-6
                and abs(self.g - other.g) < 1e-6
                and abs(self.b - other.b) < 1e-6
            )
        return NotImplemented

    def code_to_rgb(self):
        if self.code == LDR_DEF_COLOUR:
            self.r, self.g, self.b = 0.62, 0.62, 0.62  # Default LDraw color RGB
            return

        if self.code in LDR_COLOUR_RGB:
            rgb_hex = LDR_COLOUR_RGB[self.code]
            try:
                self.r, self.g, self.b = LDRColour.RGBFromHex(rgb_hex)
            except ValueError:  # Should not happen if LDR_COLOUR_RGB is well-formed
                self.r, self.g, self.b = 0.8, 0.8, 0.8  # Fallback
        else:
            # Code not in LDR_COLOUR_RGB, use default RGB or keep current if set by hex
            # If self.r, self.g, self.b were already set (e.g. from custom hex), don't override
            # This case implies self.code might be LDR_DEF_COLOUR but r,g,b are custom
            # For safety, if code is unknown and not LDR_DEF_COLOUR, revert to default LDR_DEF_COLOUR's RGB
            if (
                self.r == 0.8 and self.g == 0.8 and self.b == 0.8
            ):  # If still at initial defaults
                self.r, self.g, self.b = 0.62, 0.62, 0.62

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.r, self.g, self.b)

    def as_bgr(self) -> Tuple[int, int, int]:  # For OpenCV etc.
        return (int(self.b * 255), int(self.g * 255), int(self.r * 255))

    def as_hex(self) -> str:
        return (
            f"{int(self.r * 255.0):02X}"
            f"{int(self.g * 255.0):02X}"
            f"{int(self.b * 255.0):02X}"
        )

    def ldvcode(self) -> int:  # LDView compatible code
        # LDView might not support all custom codes > 1000 that are not official direct/edge colors
        # For now, assume all defined codes are usable or LDView handles them.
        # Original logic: if self.code >= 1000: pc = LDR_DEF_COLOUR
        # This might be too restrictive if codes like 1004 (LDR_BLKWHT_COLOUR) are used.
        # LDraw spec allows codes up to 499 for solid, 511 for transparent.
        # Codes for special purposes (like multi-color fills) are application-specific.
        return self.code  # Return the actual code for now

    def name(self) -> str:
        theName = LDRColour.SafeLDRColourName(self.code)
        if not theName or theName == str(
            LDR_DEF_COLOUR
        ):  # If no name or default code name
            # For custom RGB (where code might be LDR_DEF_COLOUR but RGB is specific)
            if self.code == LDR_DEF_COLOUR and not (
                self.r == 0.62 and self.g == 0.62 and self.b == 0.62
            ):
                return f"#{self.as_hex()}"
            elif not theName:  # Truly unknown code with no name
                return f"LDR Code {self.code}"
        return theName

    def high_contrast_complement(self) -> Tuple[float, float, float]:
        # Calculate perceived luminance (simplified)
        luminance = 0.299 * self.r + 0.587 * self.g + 0.114 * self.b
        return (0.0, 0.0, 0.0) if luminance > 0.5 else (1.0, 1.0, 1.0)

    def to_bricklink(self) -> int:
        for bl_code, ldr_code in BL_TO_LDR_COLOUR.items():
            if ldr_code == self.code:
                return bl_code
        return 0  # Default if no mapping

    @staticmethod
    def SafeLDRColourName(ldrCode: int) -> str:
        if ldrCode in LDR_COLOUR_NAME:
            return LDR_COLOUR_NAME[ldrCode]
        if ldrCode in LDR_COLOUR_TITLE:  # For special group codes
            return LDR_COLOUR_TITLE[ldrCode]
        return ""  # Return empty if no name found

    @staticmethod
    def SafeLDRColourRGB(ldrCode: int) -> str:  # Returns hex string
        if ldrCode in LDR_COLOUR_RGB:
            return LDR_COLOUR_RGB[ldrCode]
        return "CCCCCC"  # Default LDR_DEF_COLOUR hex (approx)

    @staticmethod
    def BLColourCodeFromLDR(ldr_code_val: int) -> int:
        for bl_code, ldr_code_map in BL_TO_LDR_COLOUR.items():
            if ldr_code_map == ldr_code_val:
                return bl_code
        return 0

    @staticmethod
    def ColourCodeFromString(colourStr: str) -> int:
        # Check by name (case-insensitive)
        for code, name_val in LDR_COLOUR_NAME.items():
            if name_val.lower() == colourStr.lower():
                return code
        # Check by hex string (e.g., "#RRGGBB" or "RRGGBB")
        norm_colour_str = colourStr.lstrip("#").lower()
        if len(norm_colour_str) == 6 and all(
            c in string.hexdigits for c in norm_colour_str
        ):
            for code, rgb_hex_val in LDR_COLOUR_RGB.items():
                if norm_colour_str == rgb_hex_val.lower():
                    return code
        return LDR_DEF_COLOUR  # Default if not found

    @staticmethod
    def RGBFromHex(hexStr: str) -> Tuple[float, float, float]:
        hs = hexStr.lstrip("#")
        if not (len(hs) == 6 and all(c in string.hexdigits for c in hs)):
            # Try 3-digit hex
            if len(hs) == 3 and all(c in string.hexdigits for c in hs):
                hs = "".join([c * 2 for c in hs])  # Expand "RGB" to "RRGGBB"
            else:
                raise ValueError(f"Invalid hex string: {hexStr}")

        r_val = int(hs[0:2], 16) / 255.0
        g_val = int(hs[2:4], 16) / 255.0
        b_val = int(hs[4:6], 16) / 255.0
        return (r_val, g_val, b_val)


# These functions use constants from ldrcolourdict and LDRColour class methods
def FillColoursFromLDRCode(ldrCode: int) -> List[Tuple[float, float, float]]:
    fill_colour_hex_list: List[str] = []
    if ldrCode in LDR_COLOUR_RGB:  # Single official color
        fill_colour_hex_list.append(LDR_COLOUR_RGB[ldrCode])
    elif ldrCode in LDR_FILL_CODES:  # Special fill group code
        fill_colour_hex_list = LDR_FILL_CODES[ldrCode]

    rgb_tuples: List[Tuple[float, float, float]] = []
    for hex_val in fill_colour_hex_list:
        try:
            rgb_tuples.append(LDRColour.RGBFromHex(hex_val))
        except ValueError:
            pass  # Skip invalid hex strings in definitions
    return rgb_tuples


def FillTitlesFromLDRCode(ldrCode: int) -> List[str]:
    fill_titles_list: List[str] = []
    if ldrCode in LDR_COLOUR_NAME:  # Single official color
        fill_titles_list.append(LDR_COLOUR_NAME[ldrCode])
    elif ldrCode in LDR_FILL_TITLES:  # Special fill group code
        fill_titles_list = LDR_FILL_TITLES[ldrCode]
    return fill_titles_list
