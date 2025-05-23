# tests/test_ldrcolour.py

import os

# import sys # Not used
import pytest

# Explicit imports from ldrawpy package
from ldrawpy.ldrcolour import LDRColour, FillColoursFromLDRCode, FillTitlesFromLDRCode
from ldrawpy.constants import (
    LDR_ORGYLW_COLOUR,
    LDR_BLKWHT_COLOUR,
    LDR_DEF_COLOUR,  # If needed for comparison, though LDRColour defaults to it
)

# from ldrawpy import * # AVOID wildcard imports


def test_init_colour():
    c1 = LDRColour(15)  # White by LDR code
    c2 = LDRColour("White")  # White by name
    c3 = LDRColour([1.0, 1.0, 1.0])  # White by float RGB
    c4 = LDRColour([255, 255, 255])  # White by int RGB
    c5 = LDRColour("#FFFFFF")  # White by hex
    assert c1 == c2
    assert c1 == c3
    assert c1 == c4
    assert c1 == c5
    assert c1.code == 15  # Explicitly check code for c1

    c_default = LDRColour()  # Should be LDR_DEF_COLOUR
    assert c_default.code == LDR_DEF_COLOUR
    # Check RGB for default (approx values, allow tolerance)
    assert abs(c_default.r - 0.62) < 1e-5
    assert abs(c_default.g - 0.62) < 1e-5
    assert abs(c_default.b - 0.62) < 1e-5


def test_equality():
    c1 = LDRColour([0.4, 0.2, 0.6])  # RGB: 102, 51, 153 -> Hex #663399
    c2 = LDRColour("#663399")
    assert c1 == c2  # Should be equal based on RGB comparison
    c3 = LDRColour([102, 51, 153])
    assert c2 == c3
    assert c1 == c3

    # Test equality with known LDR codes if #663399 corresponds to one
    # For example, if LDR code 89 ("Blue Violet") is #4C61DB
    # c_blue_violet_code = LDRColour(89)
    # c_blue_violet_hex = LDRColour("#4C61DB")
    # assert c_blue_violet_code == c_blue_violet_hex
    # assert c_blue_violet_code.code == 89
    # assert c_blue_violet_hex.code == 89 # After resolving hex to known code


def test_dict_lookup():
    # LDR_ORGYLW_COLOUR is a special code (e.g., 1017) for a group, not a single RGB.
    # LDRColour constructor might default it to LDR_DEF_COLOUR if it's not in LDR_COLOUR_RGB.
    # The test should reflect how LDRColour handles such special codes.

    # Test SafeLDRColourName for a known single color code
    c_orange = LDRColour(25)  # LDR Orange
    assert c_orange.name() == "Orange"
    assert LDRColour.SafeLDRColourName(25) == "Orange"

    # Test SafeLDRColourName for a special group code
    # It should return the title from LDR_COLOUR_TITLE
    assert LDRColour.SafeLDRColourName(LDR_ORGYLW_COLOUR) == "Orange/Yellow"

    # Test FillColoursFromLDRCode and FillTitlesFromLDRCode for a special group
    # LDR_BLKWHT_COLOUR (e.g. 1004)
    colours_rgb_tuples = FillColoursFromLDRCode(
        LDR_BLKWHT_COLOUR
    )  # Returns list of (r,g,b) tuples

    # Expected RGB for Black (hex "05131D") and White (hex "FFFFFF")
    # R: 5/255, G: 19/255, B: 29/255
    expected_black_rgb = (
        5 / 255.0,  # Corrected R value for "05131D"
        19 / 255.0,
        29 / 255.0,
    )
    expected_white_rgb = (1.0, 1.0, 1.0)

    # Use pytest.approx for float tuple comparison to handle potential precision issues
    found_black = False
    found_white = False
    for tpl in colours_rgb_tuples:
        if tpl == pytest.approx(expected_black_rgb, abs=1e-7):
            found_black = True
        if tpl == pytest.approx(expected_white_rgb, abs=1e-7):
            found_white = True

    assert (
        found_black
    ), f"Expected Black RGB {expected_black_rgb} not found in {colours_rgb_tuples}"
    assert (
        found_white
    ), f"Expected White RGB {expected_white_rgb} not found in {colours_rgb_tuples}"
    assert len(colours_rgb_tuples) == 2

    titles = FillTitlesFromLDRCode(LDR_BLKWHT_COLOUR)
    assert "Black" in titles
    assert "White" in titles
    assert len(titles) == 2

    # Test with a direct LDR code that has an RGB value
    colours_single_rgb = FillColoursFromLDRCode(14)  # Yellow
    yellow_rgb_from_dict = LDRColour.RGBFromHex(LDRColour.SafeLDRColourRGB(14))

    assert len(colours_single_rgb) == 1
    assert colours_single_rgb[0] == pytest.approx(yellow_rgb_from_dict, abs=1e-7)

    titles_single = FillTitlesFromLDRCode(14)  # Yellow
    assert "Yellow" in titles_single
    assert len(titles_single) == 1
