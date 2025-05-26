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
ldrcolour.py: Defines the LDRColour class for LDraw colour management.

This module provides functionality to represent and convert LDraw colours
between various formats including LDraw code, BrickLink code, colour name,
RGB float values, and RGB hex strings.
"""

import string
from typing import Tuple, Union, List, Optional, Any, Dict

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

# Type alias for more complex color input accepted by LDRColour constructor
ColourInputType = Union[int, str, Tuple[float, ...], List[float], "LDRColour"]


class LDRColour(object):
    """
    LDraw colour helper class.

    This class can be used to store an LDraw colour and perform conversions
    among various representations:
      - LDraw colour code (integer)
      - BrickLink colour code (integer)
      - Colour name (string)
      - RGB floating point tuple (0.0-1.0 for each channel)
      - RGB hex string (e.g., "#RRGGBB")

    Attributes:
        code (int): The LDraw colour code.
        r (float): Red component (0.0 - 1.0).
        g (float): Green component (0.0 - 1.0).
        b (float): Blue component (0.0 - 1.0).
    """

    code: int
    r: float
    g: float
    b: float

    def __init__(self, colour_input: ColourInputType = LDR_DEF_COLOUR):
        """
        Initializes an LDRColour object.

        The colour can be specified as an LDraw code (int), a colour name (str),
        an RGB tuple/list (either 0-255 ints or 0.0-1.0 floats), an RGB hex string,
        or another LDRColour instance.

        Args:
            colour_input: The input colour value. Defaults to LDR_DEF_COLOUR.
        """
        self.code = LDR_DEF_COLOUR  # Default LDraw code
        # Default RGB for LDR_DEF_COLOUR (often a placeholder like medium grey)
        # These will be overridden if a valid colour_input is provided.
        self.r = 0.62
        self.g = 0.62
        self.b = 0.62

        if isinstance(colour_input, (tuple, list)):
            if len(colour_input) == 3:
                # Ensure all elements are numeric before processing
                if all(isinstance(c_val, (int, float)) for c_val in colour_input):
                    # Check if RGB values are 0-255 (int) or 0.0-1.0 (float)
                    # Assuming if any value > 1.0, it's 0-255 range.
                    if any(float(c_val) > 1.0 for c_val in colour_input):  # type: ignore
                        self.r = min(float(colour_input[0]) / 255.0, 1.0)
                        self.g = min(float(colour_input[1]) / 255.0, 1.0)
                        self.b = min(float(colour_input[2]) / 255.0, 1.0)
                    else:  # Assumed to be 0.0-1.0 float range
                        self.r = float(colour_input[0])
                        self.g = float(colour_input[1])
                        self.b = float(colour_input[2])
                    # Try to find a matching LDraw code from the determined RGB values
                    self.rgb_to_code()  # Updates self.code if a match is found
                else:  # Non-numeric elements in tuple/list
                    self.code_to_rgb()  # Fallback to default LDR_DEF_COLOUR's RGB
            else:  # Tuple/list not of length 3
                self.code_to_rgb()  # Fallback

        elif isinstance(colour_input, str):
            # Try parsing as LDraw colour name or hex string
            parsed_code_from_string: int = LDRColour.ColourCodeFromString(colour_input)
            if (
                parsed_code_from_string != LDR_DEF_COLOUR
            ):  # Found a known name or hex matching a code
                self.code = parsed_code_from_string
                self.code_to_rgb()
            elif colour_input.startswith("#") and (
                len(colour_input) == 7 or len(colour_input) == 4
            ):
                # It's a hex string not directly in LDR_COLOUR_RGB, treat as custom color
                try:
                    r_val, g_val, b_val = LDRColour.RGBFromHex(colour_input)
                    self.r, self.g, self.b = r_val, g_val, b_val
                    # self.code remains LDR_DEF_COLOUR for custom hex not in LDR_COLOUR_RGB
                    # unless rgb_to_code() finds a match for this custom hex.
                    self.rgb_to_code()
                except ValueError:  # Invalid hex string
                    self.code_to_rgb()  # Fallback
            else:  # String is not a known name or valid hex
                self.code_to_rgb()  # Fallback

        elif isinstance(colour_input, LDRColour):  # Input is another LDRColour instance
            self.code = colour_input.code
            self.r = colour_input.r
            self.g = colour_input.g
            self.b = colour_input.b
        elif isinstance(colour_input, int):  # Input is an LDraw code
            self.code = colour_input
            self.code_to_rgb()
        else:  # Fallback for other unexpected types
            self.code_to_rgb()

    def __repr__(self) -> str:
        """Returns a detailed string representation of the LDRColour object."""
        return (
            f"{self.__class__.__name__}(code={self.code}, name='{self.name()}', "
            f"rgb=({self.r:.3f}, {self.g:.3f}, {self.b:.3f}), hex='#{self.as_hex()}')"
        )

    def __str__(self) -> str:
        """Returns the common name of the LDraw colour."""
        return self.name()  # Use the name() method for consistent output

    def __eq__(self, other: object) -> bool:
        """
        Compares this LDRColour with another for equality.

        If both colours have known LDraw codes (not LDR_DEF_COLOUR),
        equality is based on the code. Otherwise, it's based on
        approximate equality of RGB float values.
        Can also compare directly with an integer LDraw code.
        """
        if isinstance(other, int):  # Direct comparison with an LDraw code
            return self.code == other
        if isinstance(other, LDRColour):
            # If both have specific LDraw codes (not the default/placeholder code), compare codes
            if self.code != LDR_DEF_COLOUR and other.code != LDR_DEF_COLOUR:
                return self.code == other.code
            # Otherwise (one or both are LDR_DEF_COLOUR or custom RGB), compare RGB values
            # Use a small tolerance for float comparison
            return (
                abs(self.r - other.r) < 1e-6
                and abs(self.g - other.g) < 1e-6
                and abs(self.b - other.b) < 1e-6
            )
        return NotImplemented

    def code_to_rgb(self) -> None:
        """
        Sets the r, g, b attributes based on the current LDraw `self.code`.
        If the code is not found in the LDR_COLOUR_RGB dictionary,
        it defaults to the RGB values for LDR_DEF_COLOUR.
        """
        if self.code == LDR_DEF_COLOUR:  # Special handling for default colour
            # These are typical RGB values for LDR_DEF_COLOUR (often a medium grey)
            self.r, self.g, self.b = 0.62, 0.62, 0.62
            return

        rgb_hex_string: Optional[str] = LDR_COLOUR_RGB.get(self.code)
        if rgb_hex_string:
            try:
                self.r, self.g, self.b = LDRColour.RGBFromHex(rgb_hex_string)
            except ValueError:  # Should not happen if LDR_COLOUR_RGB is well-formed
                # Fallback to LDR_DEF_COLOUR's RGB if hex string in dict is invalid
                self.r, self.g, self.b = 0.62, 0.62, 0.62
        else:
            # Code not in LDR_COLOUR_RGB, default to LDR_DEF_COLOUR's RGB
            # This handles unknown or special codes not having direct RGB mappings.
            self.r, self.g, self.b = 0.62, 0.62, 0.62

    def rgb_to_code(self) -> None:
        """
        Attempts to find and set the LDraw `self.code` based on the current
        r, g, b attributes. If no exact RGB match is found in LDR_COLOUR_RGB,
        `self.code` remains unchanged (or defaults to LDR_DEF_COLOUR if it was already that).
        This is useful after setting RGB directly or from a hex string.
        """
        current_hex = self.as_hex().lower()
        for ldr_code_val, hex_val_in_dict in LDR_COLOUR_RGB.items():
            if current_hex == hex_val_in_dict.lower():
                self.code = ldr_code_val
                return
        # If no match, self.code remains as it was. If it was LDR_DEF_COLOUR
        # because it was a custom RGB, it stays LDR_DEF_COLOUR.

    def as_tuple(self) -> Tuple[float, float, float]:
        """Returns the colour as an RGB tuple of floats (0.0-1.0)."""
        return (self.r, self.g, self.b)

    def as_bgr_int_tuple(self) -> Tuple[int, int, int]:  # Renamed for clarity
        """Returns the colour as a BGR tuple of integers (0-255)."""
        return (int(self.b * 255.0), int(self.g * 255.0), int(self.r * 255.0))

    def as_hex(self) -> str:
        """
        Returns the colour as an RGB hex string (e.g., "RRGGBB").
        Ensures components are clamped to 0.0-1.0 before conversion.
        """
        # Clamp r,g,b values to the [0.0, 1.0] range before converting to hex
        r_clamped = max(0.0, min(1.0, self.r))
        g_clamped = max(0.0, min(1.0, self.g))
        b_clamped = max(0.0, min(1.0, self.b))
        return (
            f"{int(r_clamped * 255.0):02X}"
            f"{int(g_clamped * 255.0):02X}"
            f"{int(b_clamped * 255.0):02X}"
        )

    def ldvcode(self) -> int:  # "ldvcode" might be a typo for "ldcode" or "code"
        """Returns the LDraw colour code."""
        return self.code

    def name(self) -> str:
        """
        Returns the common name of the LDraw colour.
        If the code is LDR_DEF_COLOUR but the RGB values are custom (e.g., from a hex input
        not matching a known LDraw colour), it returns the hex string itself.
        For truly unknown codes with no name, returns "LDR Code <number>".
        """
        colour_name_from_dict: str = LDRColour.SafeLDRColourName(self.code)

        # Check if RGB is the standard for LDR_DEF_COLOUR
        is_standard_default_rgb = (
            abs(self.r - 0.62) < 1e-6
            and abs(self.g - 0.62) < 1e-6
            and abs(self.b - 0.62) < 1e-6
        )
        # Check if RGB is the initial placeholder (0.8, 0.8, 0.8) from constructor
        is_initial_placeholder_rgb = (
            abs(self.r - 0.8) < 1e-6
            and abs(self.g - 0.8) < 1e-6
            and abs(self.b - 0.8) < 1e-6
        )

        if self.code == LDR_DEF_COLOUR and not (
            is_standard_default_rgb or is_initial_placeholder_rgb
        ):
            # It's LDR_DEF_COLOUR code but with custom RGB values (e.g., from a hex string input)
            return f"#{self.as_hex()}"
        elif (
            not colour_name_from_dict
        ):  # Code is not LDR_DEF_COLOUR and has no name in dictionaries
            return f"LDR Code {self.code}"
        return colour_name_from_dict  # Return name from dictionary

    def high_contrast_complement(self) -> Tuple[float, float, float]:
        """
        Returns a high-contrast complementary colour (black or white)
        based on the luminance of the current colour. Useful for text overlays.
        """
        # Standard luminance calculation
        luminance: float = 0.299 * self.r + 0.587 * self.g + 0.114 * self.b
        # Return black for light colours, white for dark colours
        return (0.0, 0.0, 0.0) if luminance > 0.5 else (1.0, 1.0, 1.0)

    def to_bricklink(self) -> int:
        """
        Converts the LDraw colour code to its corresponding BrickLink colour code.
        Returns 0 if no mapping is found.
        """
        return LDRColour.BLColourCodeFromLDR(self.code)  # Delegate to static method

    @staticmethod
    def SafeLDRColourName(ldr_code_val: int) -> str:
        """
        Safely retrieves the LDraw colour name for a given code.
        Checks LDR_COLOUR_NAME first, then LDR_COLOUR_TITLE (for group codes).
        Returns an empty string if no name or title is found.
        """
        name_val: Optional[str] = LDR_COLOUR_NAME.get(ldr_code_val)
        if name_val is not None:
            return name_val
        title_val: Optional[str] = LDR_COLOUR_TITLE.get(ldr_code_val)
        if title_val is not None:
            return title_val
        return ""  # Return empty string if no name or title found

    @staticmethod
    def SafeLDRColourRGB(ldr_code_val: int) -> str:
        """
        Safely retrieves the RGB hex string for a given LDraw code.
        Defaults to "CCCCCC" (a light grey) if the code is not found,
        which is a common placeholder for LDR_DEF_COLOUR if its specific RGB isn't set.
        """
        return LDR_COLOUR_RGB.get(ldr_code_val, "CCCCCC")

    @staticmethod
    def BLColourCodeFromLDR(ldr_code_val: int) -> int:
        """
        Converts an LDraw colour code to its BrickLink equivalent.
        Returns 0 if no mapping exists.
        """
        for bl_code, ldr_code_mapped in BL_TO_LDR_COLOUR.items():
            if ldr_code_mapped == ldr_code_val:
                return bl_code
        return 0  # No BrickLink mapping found

    @staticmethod
    def ColourCodeFromString(colour_str_input: str) -> int:
        """
        Attempts to determine an LDraw colour code from a string.
        The string can be a colour name (case-insensitive) or an RGB hex string.
        If a name matches, its code is returned.
        If a hex string matches an RGB value in LDR_COLOUR_RGB, its code is returned.
        Otherwise, LDR_DEF_COLOUR is returned.
        """
        # Check against colour names (case-insensitive)
        for code_val, name_val_in_dict in LDR_COLOUR_NAME.items():
            if name_val_in_dict.lower() == colour_str_input.lower():
                return code_val

        # Check if it's a hex string matching a known LDraw colour's RGB
        normalized_hex_str = colour_str_input.lstrip("#").lower()
        if len(normalized_hex_str) == 6 and all(
            c in string.hexdigits for c in normalized_hex_str
        ):
            for code_val, rgb_hex_in_dict in LDR_COLOUR_RGB.items():
                if normalized_hex_str == rgb_hex_in_dict.lower():
                    return code_val
        elif len(normalized_hex_str) == 3 and all(
            c in string.hexdigits for c in normalized_hex_str
        ):
            # Expand 3-digit hex (e.g., "ABC" to "AABBCC") and check again
            expanded_hex = "".join([c * 2 for c in normalized_hex_str])
            for code_val, rgb_hex_in_dict in LDR_COLOUR_RGB.items():
                if expanded_hex == rgb_hex_in_dict.lower():
                    return code_val

        return LDR_DEF_COLOUR  # Default if no match found

    @staticmethod
    def RGBFromHex(hex_str_input: str) -> Tuple[float, float, float]:
        """
        Converts an RGB hex string (e.g., "#RRGGBB" or "RRGGBB" or "#RGB" or "RGB")
        to a tuple of RGB float values (0.0-1.0).
        Raises ValueError if the hex string is invalid.
        """
        hex_cleaned = hex_str_input.lstrip("#")
        num_chars = len(hex_cleaned)

        if not (
            (num_chars == 6 or num_chars == 3)
            and all(c in string.hexdigits for c in hex_cleaned)
        ):
            raise ValueError(f"Invalid RGB hex string format: {hex_str_input}")

        if num_chars == 3:  # Expand 3-digit hex (e.g., "RGB" to "RRGGBB")
            hex_cleaned = "".join([c * 2 for c in hex_cleaned])

        # Convert hex pairs to integers, then normalize to 0.0-1.0 floats
        r_float: float = int(hex_cleaned[0:2], 16) / 255.0
        g_float: float = int(hex_cleaned[2:4], 16) / 255.0
        b_float: float = int(hex_cleaned[4:6], 16) / 255.0
        return (r_float, g_float, b_float)


def FillColoursFromLDRCode(ldr_code_input: int) -> List[Tuple[float, float, float]]:
    """
    Retrieves a list of RGB tuples for a given LDraw code.
    If the code is a direct colour, returns a list with its single RGB tuple.
    If the code is a special group code (e.g., LDR_BLUES_COLOUR),
    returns a list of RGB tuples for the colours in that group.
    Returns an empty list if the code is not found or has no associated fill colours.
    """
    fill_colour_hex_list: List[str] = []
    if ldr_code_input in LDR_COLOUR_RGB:  # Direct LDraw colour code
        fill_colour_hex_list.append(LDR_COLOUR_RGB[ldr_code_input])
    elif ldr_code_input in LDR_FILL_CODES:  # Special group code
        # LDR_FILL_CODES values are already List[str] of hex codes
        fill_colour_hex_list = LDR_FILL_CODES[ldr_code_input]

    rgb_tuples_list: List[Tuple[float, float, float]] = []
    for hex_val_str in fill_colour_hex_list:
        try:
            rgb_tuples_list.append(LDRColour.RGBFromHex(hex_val_str))
        except ValueError:  # Skip if a hex string in LDR_FILL_CODES is invalid
            pass
    return rgb_tuples_list


def FillTitlesFromLDRCode(ldr_code_input: int) -> List[str]:
    """
    Retrieves a list of colour names/titles for a given LDraw code.
    If the code is a direct colour, returns its name.
    If the code is a special group code, returns a list of names for that group.
    Returns an empty list if the code is not found or has no associated titles.
    """
    fill_titles_list_output: List[str] = []
    if ldr_code_input in LDR_COLOUR_NAME:  # Direct LDraw colour code
        fill_titles_list_output.append(LDR_COLOUR_NAME[ldr_code_input])
    elif ldr_code_input in LDR_FILL_TITLES:  # Special group code
        # LDR_FILL_TITLES values are already List[str] of names/titles
        fill_titles_list_output = LDR_FILL_TITLES[ldr_code_input]
    return fill_titles_list_output
