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
# LDraw arrow callout utilties

import os
import copy
from math import sin, cos, pi
from functools import reduce
from typing import List, Tuple, Union, Optional, Any, Dict

from toolbox import Vector, Matrix, Identity, ZAxis, safe_vector
from .ldrprimitives import LDRPart

ARROW_PREFIX = """0 BUFEXCHG A STORE"""
ARROW_PLI = """0 !LPUB PLI BEGIN IGN"""
ARROW_SUFFIX = """0 !LPUB PLI END
0 STEP
0 BUFEXCHG A RETRIEVE"""
ARROW_PLI_SUFFIX = """0 !LPUB PLI END"""
ARROW_PARTS = ["hashl2", "hashl3", "hashl4", "hashl5", "hashl6"]

ARROW_MZ = Matrix([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
ARROW_PZ = Matrix([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]])
ARROW_MX = Matrix([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
ARROW_PX = Matrix([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
ARROW_MY = Matrix([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
ARROW_PY = Matrix([[0.0, -1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]])


def value_after_token(
    tokens: List[str], value_token: str, default_val: Any, xtype: type = int
) -> Any:
    try:
        idx = tokens.index(value_token)
        if idx + 1 < len(tokens):
            return xtype(tokens[idx + 1])
    except (ValueError, TypeError):
        pass
    return default_val


def norm_angle_arrow(angle: float) -> float:
    return angle % 45.0


def vectorize_arrow(s_coords: List[str]) -> Optional[Vector]:
    if len(s_coords) == 3:
        try:
            return Vector(*(float(x) for x in s_coords))
        except ValueError:
            return None
    return None


class ArrowContext:
    colour: int
    length: int
    scale: float
    yscale: float
    offset: List[Vector]
    rotstep: Vector
    ratio: float
    outline_colour: int

    def __init__(self, colour: int = 804, length: int = 2):
        self.colour, self.length = colour, length
        self.scale, self.yscale, self.ratio = 25.0, 20.0, 0.5
        self.offset, self.rotstep = [], Vector(0, 0, 0)
        self.outline_colour = 804

    def part_for_length(self, length: int) -> str:
        if length <= 2:
            return "hashl2"
        if length == 3:
            return "hashl3"
        if length == 4:
            return "hashl4"
        return "hashl5"  # Handles >= 5

    def matrix_for_offset(
        self, ov: Vector, mask: str = "", inv: bool = False, tilt: float = 0.0
    ) -> Matrix:
        ax, ay, az = abs(ov.x), abs(ov.y), abs(ov.z)
        base = Identity()
        if "x" not in mask and ax > max(ay, az, 1e-9):
            base = (
                (ARROW_PX if ov.x < 0 else ARROW_MX)
                if not inv
                else (ARROW_MX if ov.x < 0 else ARROW_PX)
            )
        elif "y" not in mask and ay > max(ax, az, 1e-9):
            base = (
                (ARROW_MY if ov.y < 0 else ARROW_PY)
                if not inv
                else (ARROW_PY if ov.y < 0 else ARROW_MY)
            )
        elif "z" not in mask and az > max(ax, ay, 1e-9):
            base = (
                (ARROW_PZ if ov.z < 0 else ARROW_MZ)
                if not inv
                else (ARROW_MZ if ov.z < 0 else ARROW_PZ)
            )
        return base.rotate(tilt, ZAxis) if tilt != 0.0 and ZAxis else base  # type: ignore

    def loc_for_offset(
        self, ov: Vector, l: int, mask: str = "", r: float = 0.5
    ) -> Vector:
        lo = Vector(0, 0, 0)
        slx, sly, slz = (
            float(l) * r * self.scale,
            float(l) * r * self.yscale,
            float(l) * r * self.scale,
        )
        if "x" not in mask:
            lo.x = (ov.x / 2.0) - (slx if ov.x > 1e-9 else -slx if ov.x < -1e-9 else 0)
        if "y" not in mask:
            lo.y = (ov.y / 2.0) - (sly if ov.y > 1e-9 else -sly if ov.y < -1e-9 else 0)
        if "z" not in mask:
            lo.z = (ov.z / 2.0) - (slz if ov.z > 1e-9 else -slz if ov.z < -1e-9 else 0)
        if "x" in mask:
            lo.x += ov.x
        if "y" in mask:
            lo.y += ov.y
        if "z" in mask:
            lo.z += ov.z
        return lo

    def part_loc_for_offset(self, ov: Vector, mask: str = "") -> Vector:
        pfl = Vector(0, 0, 0)
        if "x" not in mask:
            pfl.x = ov.x
        if "y" not in mask:
            pfl.y = ov.y
        if "z" not in mask:
            pfl.z = ov.z
        return pfl

    def _mask_axis(self, ol: List[Vector]) -> str:
        if not ol or len(ol) <= 1:
            return ""
        mc, Xc = Vector(
            min(o.x for o in ol), min(o.y for o in ol), min(o.z for o in ol)
        ), Vector(max(o.x for o in ol), max(o.y for o in ol), max(o.z for o in ol))
        m = ""
        tol = 1e-6
        if abs(Xc.x - mc.x) > tol:
            m += "x"
        if abs(Xc.y - mc.y) > tol:
            m += "y"
        if abs(Xc.z - mc.z) > tol:
            m += "z"
        return m

    def arrow_from_dict(self, ad: Dict[str, Any]) -> str:
        al: List[str] = []
        ofg: List[Vector] = ad.get("offset", [])
        if not (isinstance(ofg, list) and all(isinstance(v, Vector) for v in ofg)):
            return ""
        mask = self._mask_axis(ofg)
        for sov in ofg:
            lpo = LDRPart()
            if lpo.from_str(ad["line"]) is None:
                continue
            ap = LDRPart()
            ap.name = self.part_for_length(ad["length"])
            abo = self.loc_for_offset(sov, ad["length"], mask, ratio=ad["ratio"])
            ap.attrib.loc = lpo.attrib.loc + abo
            ap.attrib.matrix = self.matrix_for_offset(
                sov, mask, invert=ad["invert"], tilt=ad["tilt"]
            )
            ap.attrib.colour = ad["colour"]
            al.append(str(ap))
        return "".join(al)

    def dict_for_line(
        self, ls: str, inv: bool, r: float, c: Optional[int] = None, t: float = 0.0
    ) -> Dict[str, Any]:
        return {
            "line": ls,
            "colour": c if c is not None else self.colour,
            "length": self.length,
            "offset": copy.deepcopy(self.offset),
            "invert": inv,
            "ratio": r,
            "tilt": t,
        }


def arrows_for_step(
    arrow_ctx: ArrowContext,
    step_content: str,
    as_lpub: bool = True,
    only_arrows: bool = False,
    as_dict: bool = False,
) -> Union[str, List[Dict[str, Any]]]:
    pl: List[str] = []
    adl: List[Dict[str, Any]] = []
    oplfab: List[str] = (
        []
    )  # processed_lines, arrow_data_list, original_part_lines_from_arrow_blocks
    iamb = False
    cbo: List[Vector] = []
    cbac = arrow_ctx.colour
    cbal = arrow_ctx.length
    cbar = arrow_ctx.ratio
    cbat = 0.0
    for ls in step_content.splitlines():
        s = ls.lstrip()
        lt = int(s[0]) if s and s[0].isdigit() else -1
        if lt == 0:
            tk = ls.upper().split()
            ipac = "!PY" in tk and "ARROW" in tk
            if ipac:
                if "BEGIN" in tk:
                    iamb = True
                    cbo = []
                    ct = [
                        t
                        for t in tk
                        if t
                        not in {
                            "!PY",
                            "ARROW",
                            "BEGIN",
                            "COLOUR",
                            "LENGTH",
                            "RATIO",
                            "TILT",
                        }
                        and not t.isalpha()
                    ]
                    ix = 0
                    while ix + 2 < len(ct):
                        v = vectorize_arrow(ct[ix : ix + 3])
                        _ = v and cbo.append(v)
                        ix += 3  # type: ignore
                    arrow_ctx.offset = cbo
                    cbac = value_after_token(tk, "COLOUR", arrow_ctx.colour, int)
                    cbal = value_after_token(tk, "LENGTH", arrow_ctx.length, int)
                    cbar = value_after_token(tk, "RATIO", arrow_ctx.ratio, float)
                    cbat = value_after_token(tk, "TILT", 0.0, float)
                    arrow_ctx.colour = cbac
                    arrow_ctx.length = cbal  # Update context from BEGIN
                elif "END" in tk and iamb:
                    iamb = False
                elif not iamb:
                    arrow_ctx.colour = value_after_token(
                        tk, "COLOUR", arrow_ctx.colour, int
                    )
                    arrow_ctx.length = value_after_token(
                        tk, "LENGTH", arrow_ctx.length, int
                    )
                    arrow_ctx.ratio = value_after_token(
                        tk, "RATIO", arrow_ctx.ratio, float
                    )
            if not as_lpub or (as_lpub and not ipac):
                if not only_arrows:
                    pl.append(ls)
        elif lt == 1:
            if iamb:
                oplfab.append(ls)
                adl.extend(
                    [
                        arrow_ctx.dict_for_line(ls, False, cbar, cbac, cbat),
                        arrow_ctx.dict_for_line(ls, True, cbar, cbac, cbat),
                    ]
                )
            if not only_arrows and not as_lpub:
                pl.append(ls)
        elif not only_arrows:
            pl.append(ls)

    if as_dict:
        return adl
    if as_lpub:
        lrl: List[str] = []
        if oplfab or adl:
            lrl.append(ARROW_PREFIX)
            ulp = set()
            for adi in adl:
                if adi["line"] not in ulp:
                    po = LDRPart()
                    if po.from_str(adi["line"]):
                        if adi["offset"]:
                            fo = adi["offset"][0]
                            m = arrow_ctx._mask_axis(adi["offset"])
                            pofb = arrow_ctx.part_loc_for_offset(fo, m)
                            po.attrib.loc += pofb
                        lrl.append(str(po).strip())
                        ulp.add(adi["line"])
            lrl.append(ARROW_PLI)
            for adi in adl:
                ags = arrow_ctx.arrow_from_dict(adi)
                _ = ags and lrl.append(ags.strip())  # type: ignore
            lrl.append(ARROW_SUFFIX)
            lrl.append(ARROW_PLI)
            for pl_line in oplfab:
                lrl.append(pl_line.strip())
            lrl.append(ARROW_PLI_SUFFIX)
            return "\n".join(lrl) + "\n" if lrl else ""
        return "\n".join(pl) + "\n" if pl else ""
    fnll: List[str] = []
    if not only_arrows:
        fnll.extend(pl)
    for adi in adl:
        ags = arrow_ctx.arrow_from_dict(adi)
        _ = ags and fnll.append(ags.strip())  # type: ignore
    return "\n".join(fnll) + "\n" if fnll else ""


def arrows_for_lpub_file(filename: str, outfile: str):
    arrow_ctx = ArrowContext()
    try:
        with open(filename, "rt", encoding="utf-8") as fp_in:
            content = fp_in.read()
    except FileNotFoundError:
        print(f"Error: Input file {filename} not found.")
        return

    output_final_parts: List[str] = []  # Stores final string parts to be joined
    file_blocks = content.split("0 FILE")

    first_block_is_raw_model = not content.strip().startswith("0 FILE") and bool(
        file_blocks
    )
    start_block_idx = 1 if first_block_is_raw_model else 0

    if first_block_is_raw_model:
        # Process the very first block (before any "0 FILE")
        steps_content = file_blocks[0].split("0 STEP")
        for i, step_text in enumerate(steps_content):
            if i == 0 and not step_text.strip() and len(steps_content) > 1:
                continue
            # Add "0 STEP" if it's not the first step *of this block*
            # and the content is not empty.
            prefix = "0 STEP\n" if i > 0 and step_text.strip() else ""
            processed_str = arrows_for_step(arrow_ctx, step_text, as_lpub=True)
            if processed_str.strip():
                output_final_parts.append(prefix + processed_str)

    # Process blocks starting with "0 FILE"
    for i in range(start_block_idx, len(file_blocks)):
        block_text_with_maybe_header = file_blocks[i]
        if not block_text_with_maybe_header.strip():
            continue

        # Ensure "0 FILE" prefix for these blocks
        current_file_block_str = block_text_with_maybe_header.strip()
        if (
            not current_file_block_str.startswith("0 FILE") and i > 0
        ):  # Should only happen if file_blocks[0] was empty
            current_file_block_str = "0 FILE " + current_file_block_str

        block_lines = current_file_block_str.splitlines()
        if not block_lines:
            continue

        output_final_parts.append(block_lines[0])  # Add "0 FILE <name>" line
        content_for_steps_parsing = "\n".join(block_lines[1:])

        steps_content_in_block = content_for_steps_parsing.split("0 STEP")
        for j, step_text in enumerate(steps_content_in_block):
            if j == 0 and not step_text.strip() and len(steps_content_in_block) > 1:
                continue
            prefix = (
                "0 STEP\n" if j > 0 and step_text.strip() else ""
            )  # Add "0 STEP" for subsequent steps
            processed_str = arrows_for_step(arrow_ctx, step_text, as_lpub=True)
            # arrows_for_step with as_lpub=True might return a block that itself contains "0 STEP"
            # from ARROW_SUFFIX. We need to avoid doubling "0 STEP".
            # If processed_str starts with ARROW_PREFIX, it's an arrow block that handles its own STEP.
            if processed_str.strip():
                if processed_str.strip().startswith(ARROW_PREFIX.strip()):
                    output_final_parts.append(processed_str)  # It's self-contained
                else:  # It's regular content, needs prefix if not first step of this FILE block
                    output_final_parts.append(prefix + processed_str)

    final_output_str = "\n".join(filter(None, [s.strip() for s in output_final_parts]))
    if final_output_str and not final_output_str.endswith("\n"):
        final_output_str += "\n"

    try:
        with open(outfile, "w", encoding="utf-8") as fpo:
            fpo.write(final_output_str)
    except Exception as e:
        print(f"Error writing output file {outfile}: {e}")


def remove_offset_parts(
    parts: List[Union[LDRPart, str]],
    oparts: List[Union[LDRPart, str]],
    arrow_dict_list: List[Dict[str, Any]],
    as_str: bool = False,
) -> Union[List[LDRPart], List[str]]:
    pp_objs: List[LDRPart] = []
    for item in parts:
        if isinstance(item, LDRPart):
            pp_objs.append(item)
        elif isinstance(item, str):
            p_obj = LDRPart()
            _ = p_obj.from_str(item) and pp_objs.append(p_obj)  # type: ignore
    op_objs: List[LDRPart] = []
    for item in oparts:
        if isinstance(item, LDRPart):
            op_objs.append(item)
        elif isinstance(item, str):
            p_obj = LDRPart()
            _ = p_obj.from_str(item) and op_objs.append(p_obj)  # type: ignore

    arrow_part_names: set[str] = set()
    arrow_offsets_world: List[Vector] = []
    for adi in arrow_dict_list:
        if not isinstance(adi, dict):
            continue
        offs_in_instr = adi.get("offset", [])
        if isinstance(offs_in_instr, list) and all(
            isinstance(v, Vector) for v in offs_in_instr
        ):
            arrow_offsets_world.extend(offs_in_instr)
        ags = adi.get("arrow")
        if isinstance(ags, str):
            tagp = LDRPart()
            _ = tagp.from_str(ags) and arrow_part_names.add(tagp.name)  # type: ignore

    kept_parts: List[LDRPart] = []
    for pc in pp_objs:
        if pc.name in arrow_part_names:
            kept_parts.append(pc)
            continue
        is_offset_ver = False
        for o_ref in op_objs:
            if not (pc.name == o_ref.name and pc.attrib.colour == o_ref.attrib.colour):
                continue
            p_loc, o_loc = pc.attrib.loc, o_ref.attrib.loc
            for off_v in arrow_offsets_world:
                if p_loc.almost_same_as(o_loc + off_v, 0.1) or o_loc.almost_same_as(p_loc + off_v, 0.1):  # type: ignore
                    is_offset_ver = True
                    break
            if is_offset_ver:
                break
        if not is_offset_ver:
            kept_parts.append(pc)
    return [str(p) for p in kept_parts] if as_str else kept_parts
