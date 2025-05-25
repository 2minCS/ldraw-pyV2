# tests/test_ldrprimitives.py

import os

# import sys # sys is not used in this file
import pytest

# Explicit import from toolbox
from toolbox import Vector

# Explicit imports from the ldrawpy package
from ldrawpy.ldrprimitives import LDRLine, LDRTriangle, LDRQuad, LDRPart
from ldrawpy.ldrmodel import sort_parts  # sort_parts is in ldrmodel

# Constants like LDR_DEF_COLOUR are often used implicitly by class defaults,
# but if used directly in tests, they'd be imported from ldrawpy.constants.


def test_ldrline():
    l1 = LDRLine(
        0
    )  # Assumes LDR_DEF_COLOUR or similar is handled by default in LDRLine
    l1.p2 = Vector(1, 2, 3)
    l1.translate(Vector(5, 10, 20))
    assert l1.p2.x == 6
    assert l1.p2.y == 12
    assert l1.p2.z == 23
    ls = str(l1).rstrip()
    assert ls == "2 0 5 10 20 6 12 23"


def test_ldrtriangle():
    t1 = LDRTriangle(0)
    t1.p2 = Vector(5, 10, 0)
    t1.p3 = Vector(10, 0, 0)
    t1.translate(Vector(2, 3, -7))
    assert t1.p3.x == 12
    assert t1.p3.y == 3
    assert t1.p3.z == -7
    ts = str(t1).rstrip()
    assert ts == "3 0 2 3 -7 7 13 -7 12 3 -7"


def test_ldrquad():
    q1 = LDRQuad(0)
    q1.p2 = Vector(0, 5, 0)
    q1.p3 = Vector(20, 5, 0)
    q1.p4 = Vector(20, 0, 0)
    q1.translate(Vector(7, 3, 8))
    assert q1.p4.x == 27
    assert q1.p4.y == 3
    assert q1.p4.z == 8
    qs = str(q1).rstrip()
    assert qs == "4 0 7 3 8 7 8 8 27 8 8 27 3 8"


def test_ldrpart():
    p1 = LDRPart(0)  # Assuming color 0 is a valid default or test case
    p1.attrib.loc = Vector(5, 7, 8)
    p1.name = "3002"
    assert p1.attrib.loc.x == 5
    assert p1.attrib.loc.y == 7
    assert p1.attrib.loc.z == 8
    ps = str(p1).rstrip()
    assert ps == "1 0 5 7 8 1 0 0 0 1 0 0 0 1 3002.dat"


def test_ldrpart_translate():
    p1 = LDRPart(0)
    p1.attrib.loc = Vector(5, 7, 8)
    p1.name = "3002"
    assert p1.attrib.loc.x == 5
    assert p1.attrib.loc.y == 7
    assert p1.attrib.loc.z == 8
    ps = str(p1).rstrip()
    assert ps == "1 0 5 7 8 1 0 0 0 1 0 0 0 1 3002.dat"
    # LDRPart.translate_from_str is a static method
    p2_str = LDRPart.translate_from_str(ps, Vector(0, -2, -4))  # p2_str is a string
    # To assert properties of p2, you'd need to parse p2_str back or compare strings
    assert p2_str.rstrip() == "1 0 5 5 4 1 0 0 0 1 0 0 0 1 3002.dat"


def test_ldrpart_equality():
    p1 = LDRPart(0, name="3001")
    p2 = LDRPart(0, name="3001")
    assert p1 == p2  # Relies on LDRPart.__eq__
    p3 = LDRPart(0, name="3002")
    assert p1 != p3
    assert p1.is_identical(p2)
    p2.move_to((0, 8, 20))  # move_to uses safe_vector internally
    assert p1 == p2  # Equality might only check name and color, not location/matrix
    assert not p1.is_identical(p2)  # is_identical checks all attributes


def test_ldrpart_sort():
    p1 = LDRPart(0, name="3001")
    p2 = LDRPart(1, name="3666")  # Different color code
    p3 = LDRPart(14, name="3070b")  # Different color code

    # sort_parts is from ldrmodel
    sp_sha1 = sort_parts([p1, p2, p3], key="sha1")
    # The order depends on the SHA1 hash, which is sensitive to all attributes including color.
    # This assertion order might change if colors or other minor details change.
    # For stable testing, ensure the inputs to SHA1 are consistent or mock SHA1.
    # Original test assumed a specific order based on hashes of parts with color 0.
    # Let's check names if the order is as expected from original test:
    assert sp_sha1[0].name == "3070b"  # Smallest hash (example)
    assert sp_sha1[1].name == "3001"
    assert sp_sha1[2].name == "3666"

    sp_name_desc = sort_parts([p1, p2, p3], key="name", order="descending")
    assert sp_name_desc[0].name == "3666"
    assert sp_name_desc[1].name == "3070b"
    assert sp_name_desc[2].name == "3001"
