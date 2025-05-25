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
# LDraw related helper functions

import decimal
from toolbox import Vector
from .ldrprimitives import LDRLine, LDRPart
from .constants import ASPECT_DICT, FLIP_DICT


def quantize(x):
    """Quantizes an string LDraw value to 4 decimal places"""
    v = decimal.Decimal(x.strip()).quantize(decimal.Decimal(10) ** -4)
    return float(v)


def MM2LDU(x):
    return x * 2.5


def LDU2MM(x):
    return x * 0.4


def val_units(value, units="ldu"):
    """
    Writes a floating point value in units of either mm or ldu.
    It restricts the number of decimal places to 4 and minimizes
    redundant trailing zeros (as recommended by ldraw.org)
    """
    x = value * 2.5 if units == "mm" else value
    xs = "%.5f" % (x)
    ns = str(quantize(xs)).replace("0E-4", "0.")
    if "E" not in ns:
        ns = ns.rstrip("0")
    ns = ns.rstrip(".")
    if ns == "-0":
        return "0 "
    return ns + " "


def mat_str(m):
    """
    Writes the values of a matrix formatted by PUnits.
    """
    return "".join([val_units(v, "ldu") for v in m])


def vector_str(p, attrib):
    return (
        val_units(p.x, attrib.units)
        + val_units(p.y, attrib.units)
        + val_units(p.z, attrib.units)
    )


def get_circle_segments(radius, segments, attrib):

    lines = []
    for seg in range(segments):
        p1 = Vector(0, 0, 0)
        p2 = Vector(0, 0, 0)
        a1 = seg / segments * 2.0 * pi
        a2 = (seg + 1) / segments * 2.0 * pi
        p1.x = radius * cos(a1)
        p1.z = radius * sin(a1)
        p2.x = radius * cos(a2)
        p2.z = radius * sin(a2)
        l = LDRLine(attrib.colour, attrib.units)
        l.p1 = p1
        l.p2 = p2
        lines.append(l)
    return lines


def ldrlist_from_parts(parts):
    """Returns a list of LDRPart objects from either a list of LDRParts,
    a list of strings representing parts or a string with line feed
    delimited parts."""

    p = []
    if isinstance(parts, str):
        # assume its a string of LDraw lines of text
        parts = parts.splitlines()
    if isinstance(parts, list):
        if len(parts) < 1:
            return p
        if isinstance(parts[0], LDRPart):
            p.extend(parts)
        else:
            p = [LDRPart().from_str(e) for e in parts]
            p = [e for e in p if e is not None]
    return p


def ldrstring_from_list(parts):
    """Returns a LDraw formatted string from a list of parts.  Each part
    is represented in a line feed terminated string concatenated together."""

    s = []
    for p in parts:
        if isinstance(p, LDRPart):
            s.append(str(p))
        else:
            if not p[-1] == "\n":
                s.append(p + "\n")
            else:
                s.append(p)
    return "".join(s)


def merge_same_parts(parts, other, ignore_colour=False, as_str=False):
    """Merges parts + other where the the parts in other take precedence."""

    op = ldrlist_from_parts(other)
    p = ldrlist_from_parts(other)
    np = ldrlist_from_parts(parts)
    for n in np:
        if not any(
            [
                n.is_same(o, ignore_location=False, ignore_colour=ignore_colour)
                for o in op
            ]
        ):
            p.append(n)
    if as_str:
        return ldrstring_from_list(p)
    return p


def remove_parts_from_list(
    parts, other, ignore_colour=True, ignore_location=True, exact=False, as_str=False
):
    """Returns a list based on removing the parts from other from parts."""
    pp = ldrlist_from_parts(parts)
    op = ldrlist_from_parts(other)
    np = []
    for p in pp:
        if ignore_colour and ignore_location:
            if not any([o.name == p.name for o in op]):
                np.append(p)
        elif not any(
            [
                p.is_same(
                    o,
                    ignore_location=ignore_colour,
                    ignore_colour=ignore_colour,
                    exact=exact,
                )
                for o in op
            ]
        ):
            np.append(p)
    if as_str:
        return ldrstring_from_list(np)
    return np


def norm_angle(a):
    """Normalizes an angle in degrees to -180 ~ +180 deg."""
    a = a % 360
    if a >= 0 and a <= 180:
        return a
    if a > 180:
        return -180 + (-180 + a)
    if a >= -180 and a < 0:
        return a
    return 180 + (a + 180)


def norm_aspect(a):
    """Normalizes the three angle components of aspect angle to -180 ~ +180 deg."""
    return tuple([norm_angle(v) for v in a])


def _flip_x(a):
    return (-a[0], a[1], a[2])


def _add_aspect(a, b):
    return norm_aspect(
        new_aspect=(
            a[0] + b[0],
            a[1] + b[1],
            a[2] + b[2],
        )
    )


def preset_aspect(current_aspect, aspect_change):
    if isinstance(aspect_change, list):
        changes = aspect_change
    else:
        changes = [aspect_change]
    new_aspect = tuple(current_aspect)
    for aspect in changes:
        a = aspect.lower()
        if a in ASPECT_DICT:
            new_aspect = _flip_x(ASPECT_DICT[a])
        elif a in FLIP_DICT:
            r = FLIP_DICT[a]
            new_aspect = _add_aspect(new_aspect, r)
        elif a == "down":
            if new_aspect[0] < 0:
                new_aspect = (145, new_aspect[1], new_aspect[2])
        elif a == "up":
            if new_aspect[0] > 0:
                new_aspect = (-35, new_aspect[1], new_aspect[2])
    return norm_aspect(new_aspect)


def clean_line(line):
    sl = line.split()
    nl = []
    for i, s in enumerate(sl):
        xs = s
        if i > 0 and "_" not in s:
            try:
                x = float(s)
                xs = val_units(float(x)).rstrip()
            except ValueError:
                pass
        nl.append(xs)
        nl.append(" ")
    nl = "".join(nl).rstrip()
    return nl


def clean_file(fn, fno=None, verbose=False, as_str=False):
    """Cleans an LDraw file by changing all floating point numbers to
    an optimum representation within the suggested precision of up to
    4 decimal places.
    """
    if fno is None:
        fno = fn.replace(".ldr", "_clean.ldr")
    ns = []
    bytes_in = 0
    bytes_out = 0
    with open(fn, "r") as f:
        lines = f.readlines()
        for line in lines:
            bytes_in += len(line)
            nl = clean_line(line)
            # sl = line.split()
            # nl = []
            # for i, s in enumerate(sl):
            #     xs = s
            #     if i > 0 and "_" not in s:
            #         try:
            #             x = float(s)
            #             xs = val_units(float(x)).rstrip()
            #         except ValueError:
            #             pass
            #     nl.append(xs)
            #     nl.append(" ")
            # nl = "".join(nl).rstrip()
            bytes_out += len(nl)
            ns.append(nl)
    if verbose:
        print(
            "%s : %d bytes in / %d bytes out (%.1f%% saved)"
            % (fn, bytes_in, bytes_out, ((bytes_in - bytes_out) / bytes_in * 100.0))
        )
    if as_str:
        return ns
    ns = "\n".join(ns)
    with open(fno, "w") as f:
        f.write(ns)
