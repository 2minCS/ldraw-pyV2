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
# LDraw model classes and helper functions

import hashlib
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional, Union

# Explicit imports from toolbox
from toolbox import (
    Vector,
    Matrix,
    Identity,
    apply_params,
    split_path,
    progress_bar,
    safe_vector,
)

# Explicit imports from ldrawpy package
from .constants import (
    SPECIAL_TOKENS,
    LDR_DEF_COLOUR,
    ASPECT_DICT,
    FLIP_DICT,
    START_TOKENS,
    END_TOKENS,
    EXCEPTION_LIST,
    IGNORE_LIST,
    COMMON_SUBSTITUTIONS,
)
from .ldrprimitives import LDRPart
from .ldrhelpers import norm_aspect, preset_aspect

# Conditional import for brickbom
try:
    from brickbom import BOM, BOMPart
except ImportError:
    BOM = None  # type: ignore
    BOMPart = None  # type: ignore

# Import rich directly as it's a dependency
from rich import print as rich_print

# No need for has_rich and fallback if rich is a hard dependency.
# If it were optional, the try-except for has_rich would be appropriate.


def substitute_part(part: LDRPart) -> LDRPart:
    for e in COMMON_SUBSTITUTIONS:
        if part.name == e[0]:
            part.name = e[1]
    return part


def line_has_all_tokens(line: str, tokenlist: List[str]) -> bool:
    line_tokens = line.split()
    for t_group_str in tokenlist:
        if all(req_token in line_tokens for req_token in t_group_str.split()):
            return True
    return False


def parse_special_tokens(line: str) -> List[Dict[str, Any]]:
    ls = line.strip().split()
    metas: List[Dict[str, Any]] = []
    for cmd_key, token_patterns in SPECIAL_TOKENS.items():
        for pattern_str in token_patterns:
            pattern_tokens = pattern_str.split()
            non_placeholder_pattern_tokens = [
                pt for pt in pattern_tokens if not pt.startswith("%")
            ]
            if not all(nppt in ls for nppt in non_placeholder_pattern_tokens):
                continue
            extracted_values: List[str] = []
            valid_match_for_values = True
            try:
                cmd_in_pattern = non_placeholder_pattern_tokens[0]
                cmd_start_idx_in_line = ls.index(cmd_in_pattern)
                for pt in pattern_tokens:
                    if pt.startswith("%") and pt[1:].isdigit():
                        idx_in_pattern = int(pt[1:])
                        actual_idx_in_line = cmd_start_idx_in_line + idx_in_pattern
                        if actual_idx_in_line < len(ls):
                            extracted_values.append(ls[actual_idx_in_line])
                        else:
                            valid_match_for_values = False
                            break
            except (ValueError, IndexError):
                valid_match_for_values = False
            if not valid_match_for_values and any(
                pt.startswith("%") for pt in pattern_tokens
            ):
                continue
            if extracted_values:
                metas.append(
                    {cmd_key: {"values": extracted_values, "text": line.strip()}}
                )
            else:
                metas.append({cmd_key: {"text": line.strip()}})
            break
        if metas and metas[-1].get(cmd_key):
            break
    return metas


def get_meta_commands(ldr_string: str) -> List[Dict[str, Any]]:
    cmd: List[Dict[str, Any]] = []
    for line in ldr_string.splitlines():
        stripped_line = line.lstrip()
        if not stripped_line or not stripped_line.startswith("0 "):
            continue
        meta_for_line = parse_special_tokens(line)
        if meta_for_line:
            cmd.extend(meta_for_line)
    return cmd


def get_parts_from_model(ldr_string: str) -> List[Dict[str, str]]:
    parts: List[Dict[str, str]] = []
    lines = ldr_string.splitlines()
    mask_depth = 0
    bufex = False
    for line in lines:
        stripped_line = line.lstrip()
        if not stripped_line:
            continue
        if line_has_all_tokens(line, ["BUFEXCHG STORE"]):
            bufex = True
        if line_has_all_tokens(line, ["BUFEXCHG RETRIEVE"]):
            bufex = False
        if line_has_all_tokens(line, START_TOKENS):
            mask_depth += 1
        if line_has_all_tokens(line, END_TOKENS):
            if mask_depth > 0:
                mask_depth -= 1
        try:
            line_type = int(stripped_line[0])
        except (ValueError, IndexError):
            continue
        if line_type == 1:
            split_line_tokens = line.split()
            if len(split_line_tokens) < 15:
                continue
            part_dict = {"ldrtext": line, "partname": " ".join(split_line_tokens[14:])}
            if mask_depth == 0:
                parts.append(part_dict)
            else:
                if part_dict["partname"] in EXCEPTION_LIST:
                    parts.append(part_dict)
                elif not bufex and part_dict["partname"].endswith(".ldr"):
                    parts.append(part_dict)
    return parts


def recursive_parse_model(
    model_entries: List[Dict[str, str]],
    all_submodels_data: Dict[str, List[Dict[str, str]]],
    output_parts_list: List[LDRPart],
    current_offset: Vector = Vector(0, 0, 0),
    current_matrix: Matrix = Identity(),
    reset_parts_list_on_call: bool = False,
    filter_for_submodel_name: Optional[str] = None,
):
    if reset_parts_list_on_call:
        output_parts_list.clear()
    for entry_dict in model_entries:
        part_name, ldr_text = entry_dict["partname"], entry_dict["ldrtext"]
        if filter_for_submodel_name and part_name != filter_for_submodel_name:
            continue
        if part_name in all_submodels_data:
            submodel_entries = all_submodels_data[part_name]
            ref_part = LDRPart()
            if ref_part.from_str(ldr_text) is None:
                continue
            new_matrix = current_matrix * ref_part.attrib.matrix  # type: ignore
            new_offset = current_matrix * ref_part.attrib.loc + current_offset  # type: ignore
            recursive_parse_model(
                submodel_entries,
                all_submodels_data,
                output_parts_list,
                new_offset,
                new_matrix,
                False,
                None,
            )
        elif filter_for_submodel_name is None:
            actual_part = LDRPart()
            if actual_part.from_str(ldr_text) is None:
                continue
            actual_part = substitute_part(actual_part)
            actual_part.transform(matrix=current_matrix, offset=current_offset)
            if (
                actual_part.name not in IGNORE_LIST
                and actual_part.name.upper() not in IGNORE_LIST
            ):
                output_parts_list.append(actual_part)


def unique_set(items: List[Any]) -> Dict[Any, int]:
    return dict(defaultdict(int, {k: items.count(k) for k in set(items)}))


def key_name(elem: LDRPart) -> str:
    return elem.name


def key_colour(elem: LDRPart) -> int:
    return elem.attrib.colour


def get_sha1_hash(parts: List[LDRPart]) -> str:
    if not all(isinstance(p, LDRPart) for p in parts):
        return "error_parts_not_LDRPart_objects"
    hashes = sorted([p.sha1hash() for p in parts])
    shash = hashlib.sha1()
    for p_hash_val in hashes:
        shash.update(bytes(p_hash_val, encoding="utf8"))
    return shash.hexdigest()


def sort_parts(
    parts: List[LDRPart], key: str = "name", order: str = "ascending"
) -> List[LDRPart]:
    if not all(isinstance(p, LDRPart) for p in parts):
        return []
    sp = list(parts)
    is_desc = order.lower() == "descending"
    key_func_map = {
        "sha1": lambda p: p.sha1hash(),
        "name": key_name,
        "colour": key_colour,
    }
    sort_key_func = key_func_map.get(key.lower())
    if sort_key_func is None:
        return sp
    sp.sort(key=sort_key_func, reverse=is_desc)
    return sp


class LDRModel:
    PARAMS: Dict[str, Any] = {
        "global_origin": (0.0, 0.0, 0.0),
        "global_aspect": (-40.0, 55.0, 0.0),
        "global_scale": 1.0,
        "pli_aspect": (-25.0, -40.0, 0.0),
        "pli_exceptions": {
            "32001": (-50.0, -25.0, 0.0),
            "3676": (-25.0, 50.0, 0.0),
            "3045": (-25.0, 50.0, 0.0),
        },
        "callout_step_thr": 6,
        "continuous_step_numbers": False,
    }
    filename: str
    title: str
    bom: Optional[BOM]
    steps: Dict[int, Dict[str, Any]]
    pli: Dict[int, List[LDRPart]]
    sub_models: Dict[str, List[Dict[str, str]]]
    sub_model_str: Dict[str, str]
    unwrapped: Optional[List[Dict[str, Any]]]
    callouts: Dict[int, Dict[str, Any]]
    continuous_step_count: int
    _parsed_submodel_steps_cache: Dict[str, Any]
    global_origin: Tuple[float, float, float]
    global_aspect: Tuple[float, float, float]
    global_scale: float
    pli_aspect: Tuple[float, float, float]
    pli_exceptions: Dict[str, Tuple[float, float, float]]
    callout_step_thr: int
    continuous_step_numbers: bool

    def __init__(self, filename: str, **kwargs: Any):
        self.filename = filename
        self.title = ""
        self.bom = BOM() if BOM else None
        self.steps = {}
        self.pli = {}
        self.sub_models = {}
        self.sub_model_str = {}
        self.unwrapped = None
        self.callouts = {}
        self.continuous_step_count = 0
        self._parsed_submodel_steps_cache = {}
        for key, value in self.PARAMS.items():
            setattr(self, key, value)
        apply_params(self, kwargs)
        _, self.title = split_path(filename)
        if self.bom and hasattr(self.bom, "ignore_parts"):
            self.bom.ignore_parts = []

    def __str__(self) -> str:
        # CONVERTED TO F-STRING
        return (
            f"LDRModel: {self.title}\n"
            f"  Global origin: {self.global_origin} Global aspect: {self.global_aspect}\n"
            f"  Number of steps: {len(self.steps)}\n"
            f"  Number of sub-models: {len(self.sub_models)}"
        )

    def __getitem__(self, key: int) -> Dict[str, Any]:
        if self.unwrapped is None:
            self.unwrap()
        if self.unwrapped is None:
            raise IndexError("Model not unwrapped.")
        return self.unwrapped[key]

    def print_step_dict(self, key: int):
        if key in self.steps:
            s_dict = self.steps[key]
            for k, v in s_dict.items():
                # CONVERTED TO F-STRINGS (example, can be more detailed)
                if k == "sub_parts" and isinstance(v, dict):
                    rich_print(f"[bold blue]{k}:[/bold blue]")
                    for ks_sub, vs_list_sub in v.items():
                        rich_print(f"  [cyan]{ks_sub}:[/cyan]")
                        for e_part_sub in vs_list_sub:
                            rich_print(f"    {str(e_part_sub).rstrip()}")
                elif isinstance(v, list) and all(
                    isinstance(item, LDRPart) for item in v
                ):
                    rich_print(f"[bold blue]{k}:[/bold blue] ({len(v)} items)")
                    for vx_part_item in v:
                        rich_print(f"  {str(vx_part_item).rstrip()}")
                elif k == "pli_bom" and BOM is not None and isinstance(v, BOM):
                    rich_print(f"[bold blue]{k}:[/bold blue]")
                    if hasattr(v, "summary_str"):
                        rich_print(f"  {v.summary_str()}")
                    else:
                        rich_print(f"  {v}")
                else:
                    rich_print(f"[bold blue]{k}:[/bold blue] {v}")
        else:
            rich_print(f"Step {key} not found.")  # f-string

    def print_unwrapped_dict(self, idx: int):
        if self.unwrapped is None or not (0 <= idx < len(self.unwrapped)):
            rich_print(f"Index {idx} out of bounds for unwrapped model.")
            return  # f-string
        s_dict = self.unwrapped[idx]
        for k, v in s_dict.items():
            # CONVERTED TO F-STRINGS (example)
            if (
                k in ("parts", "step_parts")
                and isinstance(v, list)
                and all(isinstance(item, LDRPart) for item in v)
            ):
                rich_print(f"[bold green]{k}:[/bold green] ({len(v)} parts)")
                for vx_part_item in v:
                    rich_print(f"  {str(vx_part_item).rstrip()}")
            elif k == "pli_bom" and BOM is not None and isinstance(v, BOM):
                rich_print(f"[bold green]{k}:[/bold green]")
                if hasattr(v, "summary_str"):
                    rich_print(f"  {v.summary_str()}")
                else:
                    rich_print(f"  {v}")
            else:
                rich_print(f"[bold green]{k}:[/bold green] {v}")

    def print_unwrapped_verbose(self):
        if not self.unwrapped:
            rich_print("Model not unwrapped.")
            return
        for i, v in enumerate(self.unwrapped):
            aspect = v.get("aspect", (0.0, 0.0, 0.0))
            # ALREADY F-STRING
            rich_print(
                f"{i:3d}. idx:{v.get('idx','N/A'):3} "
                f"[pl:{v.get('prev_level', 'N/A')} l:{v.get('level','N/A')} nl:{v.get('next_level', 'N/A')}] "
                f"[s:{v.get('step','N/A'):2} ns:{v.get('next_step','N/A'):2} sc:{v.get('num_steps','N/A'):2}] "
                f"{str(v.get('model','N/A'))[:16]:<16} q:{v.get('qty',0):d} sc:{v.get('scale',0.0):.2f} "
                f"asp:({aspect[0]:3.0f},{aspect[1]:4.0f},{aspect[2]:3.0f})"
            )

    def print_unwrapped(self):
        if not self.unwrapped:
            rich_print("Model not unwrapped.")
            return
        for v_step in self.unwrapped:
            self.print_step(v_step)

    def print_step(self, v_step: dict):
        _print_func = rich_print
        pb = "break" if v_step.get("page_break") else ""
        co = str(v_step.get("callout", 0))
        model_name = str(v_step.get("model", "")).replace(".ldr", "")[:16]
        qty = f"({v_step.get('qty',0):2d}x)" if v_step.get("qty", 0) > 0 else "     "
        level_str = " " * v_step.get("level", 0) + f"Level {v_step.get('level',0)}"
        level_str_padded = f"{level_str:<11}"
        pli_bom_obj = v_step.get("pli_bom")
        parts_count = 0
        if BOM and isinstance(pli_bom_obj, BOM) and hasattr(pli_bom_obj, "parts"):
            parts_count = len(pli_bom_obj.parts)  # type: ignore
        elif isinstance(pli_bom_obj, list):
            parts_count = len(pli_bom_obj)
        parts_str = f"({parts_count:2d}x pcs)"
        meta_tags = []
        for m_item in v_step.get("meta", []):
            if isinstance(m_item, dict):
                for k, v_dict in m_item.items():
                    tag_str = k.replace("_", " ")
                    if k == "columns" and "values" in v_dict:
                        meta_tags.append(f"[green]COL{v_dict['values'][0]}[/]")
                    else:
                        meta_tags.append(f"[dim]{tag_str}[/dim]")
            else:
                meta_tags.append(str(m_item))
        meta_str = " ".join(meta_tags)
        aspect = v_step.get("aspect", (0.0, 0.0, 0.0))
        # ALREADY F-STRINGS (from previous update)
        fmt_base = (
            f"{v_step.get('idx','N/A'):3}. {level_str_padded} Step "
            f"{'[yellow]' if co != '0' and has_rich else '[green]'}{v_step.get('step','N/A'):3}/{v_step.get('num_steps','N/A'):3}{'[/]' if has_rich else ''} "
            f"Model: {'[red]' if co != '0' and model_name != 'root' and has_rich else '[green]'}{model_name:<16}{'[/]' if has_rich else ''} "
            f"{qty} {parts_str} scale: {v_step.get('scale',0.0):.2f} "
            f"({aspect[0]:3.0f},{aspect[1]:4.0f},{aspect[2]:3.0f})"
        )
        fmt_co = f" {'[yellow]' if has_rich and co != '0' else '[dim]' if has_rich else ''}{co}{'[/]' if has_rich else ''}"
        fmt_pb = f" {'[magenta]BR[/]' if has_rich and pb else ''}"
        _print_func(f"{fmt_base}{fmt_co}{fmt_pb} {meta_str}")

    def transform_parts_to(
        self,
        parts: List[LDRPart],
        origin: Optional[Union[Tuple[float, float, float], Vector]] = None,
        aspect: Optional[Union[Tuple[float, float, float], Vector]] = None,
        use_exceptions: bool = False,
    ) -> List[LDRPart]:
        final_aspect_tuple = aspect if aspect is not None else self.pli_aspect
        final_origin_vec = safe_vector(origin) if origin is not None else None
        transformed_parts = []
        for p in parts:
            np = p.copy()
            current_aspect_for_part = list(final_aspect_tuple)
            if use_exceptions and p.name in self.pli_exceptions:
                current_aspect_for_part = list(self.pli_exceptions[p.name])
            np.set_rotation(tuple(current_aspect_for_part))  # type: ignore
            if final_origin_vec:
                np.move_to(final_origin_vec)
            transformed_parts.append(np)
        return transformed_parts

    def transform_parts(
        self,
        parts: Union[List[LDRPart], List[str]],
        offset: Optional[Union[Tuple[float, float, float], Vector]] = None,
        aspect: Optional[Union[Tuple[float, float, float], Vector]] = None,
    ) -> Union[List[LDRPart], List[str]]:
        final_aspect_tuple = aspect if aspect is not None else self.global_aspect
        final_offset_vec = (
            safe_vector(offset) if offset is not None else Vector(0, 0, 0)
        )
        if not parts:
            return []
        if all(isinstance(p, LDRPart) for p in parts):
            processed_ldr_parts: List[LDRPart] = []
            for p_obj_any in parts:
                if isinstance(p_obj_any, LDRPart):
                    np = p_obj_any.copy()
                    np.rotate_by(final_aspect_tuple)  # type: ignore
                    np.move_by(final_offset_vec)
                    processed_ldr_parts.append(np)
            return processed_ldr_parts
        elif all(isinstance(p, str) for p in parts):
            processed_str_parts: List[str] = []
            for p_str_any in parts:
                if isinstance(p_str_any, str):
                    np = LDRPart()
                    if np.from_str(p_str_any):
                        np.rotate_by(final_aspect_tuple)  # type: ignore
                        np.move_by(final_offset_vec)
                        processed_str_parts.append(str(np))
                    else:
                        processed_str_parts.append(p_str_any)
            return processed_str_parts
        return []

    def parse_file(self):
        self.sub_models = {}
        self.sub_model_str = {}
        try:
            with open(self.filename, "rt", encoding="utf-8") as fp:
                content = fp.read()
        except FileNotFoundError:
            # CONVERTED TO F-STRING
            rich_print(f"Error: File {self.filename} not found.")
            self.pli, self.steps = {}, {}
            return
        except Exception as e:
            # CONVERTED TO F-STRING
            rich_print(f"Error reading file {self.filename}: {e}")
            self.pli, self.steps = {}, {}
            return
        file_blocks = content.split("0 FILE")
        root_model_content = ""
        sub_file_blocks = []
        if not content.strip().startswith("0 FILE") and file_blocks:
            root_model_content = file_blocks[0]
            sub_file_blocks = file_blocks[1:]
        elif len(file_blocks) > 1:
            root_model_content = "0 FILE " + file_blocks[1].strip()
            sub_file_blocks = file_blocks[2:]
        else:
            # CONVERTED TO F-STRING
            rich_print(
                f"Warning: No root model found in {self.filename} based on '0 FILE' structure."
            )
            self.pli, self.steps = {}, {}
            if (
                file_blocks
                and file_blocks[0].strip()
                and not content.strip().startswith("0 FILE")
            ):
                pass
            else:
                return
        for sub_block in sub_file_blocks:
            if not sub_block.strip():
                continue
            full_sub = "0 FILE " + sub_block.strip()
            lines = full_sub.splitlines()
            if not lines:
                continue
            sub_name = lines[0].replace("0 FILE", "").strip().lower()
            if sub_name:
                self.sub_model_str[sub_name] = full_sub
                self.sub_models[sub_name] = get_parts_from_model(full_sub)
        if root_model_content.strip():
            self.pli, self.steps = self.parse_model(root_model_content, True)
        else:
            # CONVERTED TO F-STRING
            rich_print(
                f"Warning: Root model content for {self.filename} empty after processing '0 FILE' directives."
            )
            self.pli, self.steps = {}, {}
        self.unwrap()

    def unwrap(self):
        if self.unwrapped is not None:
            return
        self._parsed_submodel_steps_cache = {}
        for name, sub_model_str_content in self.sub_model_str.items():
            _, steps_dict = self.parse_model(sub_model_str_content, is_top_level=False)
            self._parsed_submodel_steps_cache[name] = steps_dict
        self.unwrapped = self._unwrap_model_recursive(current_model_steps=self.steps)

    def _unwrap_model_recursive(
        self,
        current_model_steps: Dict[int, Dict[str, Any]],
        current_idx: int = 0,
        current_level: int = 0,
        model_name_for_step: str = "root",
        model_qty_for_step: int = 1,
        unwrapped_list: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], int]]:
        if unwrapped_list is None:
            unwrapped_list = []
            self.continuous_step_count = 0
        sorted_steps = sorted(current_model_steps.keys())
        for step_no in sorted_steps:
            step_data = current_model_steps[step_no]
            if step_data.get("sub_models"):
                for sub_name, qty in unique_set(step_data["sub_models"]).items():
                    sub_steps = self._parsed_submodel_steps_cache.get(sub_name)
                    if not sub_steps:
                        continue
                    _, current_idx = self._unwrap_model_recursive(sub_steps, current_idx, current_level + 1, sub_name, qty, unwrapped_list)  # type: ignore
            entry = {
                "idx": current_idx,
                "level": current_level,
                "step": step_no,
                "next_step": step_no + 1 if step_no < len(sorted_steps) else step_no,
                "num_steps": len(sorted_steps),
                "model": model_name_for_step,
                "qty": model_qty_for_step,
                "scale": step_data.get("scale", self.global_scale),
                "model_scale": step_data.get("model_scale", self.global_scale),
                "aspect": step_data.get("aspect", self.global_aspect),
                "parts": step_data.get("parts", []),
                "step_parts": step_data.get("step_parts", []),
                "pli_bom": step_data.get("pli_bom", BOM() if BOM else []),
                "meta": step_data.get("meta", []),
                "aspect_change": step_data.get("aspect_change", False),
                "raw_ldraw": step_data.get("raw_ldraw", ""),
                "sub_parts": step_data.get("sub_parts", {}),
            }
            unwrapped_list.append(entry)
            current_idx += 1
        if current_level == 0:
            final_model: List[Dict[str, Any]] = []
            cont_step = 1
            self.callouts = {}
            callout_starts: Dict[int, int] = {}
            for i, e in enumerate(unwrapped_list):
                e["prev_level"] = unwrapped_list[i - 1]["level"] if i > 0 else 0
                e["next_level"] = (
                    unwrapped_list[i + 1]["level"]
                    if i < len(unwrapped_list) - 1
                    else e["level"]
                )
                e["page_break"] = any("page_break" in m for m in e.get("meta", [])) or (
                    e["next_level"] > e["level"]
                    and e.get("num_steps", 0) >= self.callout_step_thr
                    and not any("no_callout" in m for m in e.get("meta", []))
                )
                is_no_callout = any("no_callout" in m for m in e.get("meta", []))
                if (
                    e["level"] > e["prev_level"]
                    and not is_no_callout
                    and e.get("num_steps", 0) < self.callout_step_thr
                ):
                    callout_starts[e["level"]] = e["idx"]
                current_callout_lvl = 0
                for lvl_cs in sorted(callout_starts.keys(), reverse=True):
                    if e["level"] >= lvl_cs:
                        current_callout_lvl = lvl_cs
                        break
                e["callout"] = current_callout_lvl
                if e["level"] < e["prev_level"] and e["prev_level"] in callout_starts:
                    start_idx = callout_starts.pop(e["prev_level"])
                    self.callouts[start_idx] = {
                        "level": e["prev_level"],
                        "end": unwrapped_list[i - 1]["idx"],
                        "parent": unwrapped_list[start_idx]["prev_level"],
                    }
                    if any(
                        "model_scale" in m
                        for m in unwrapped_list[start_idx].get("meta", [])
                    ):
                        scale_meta = next(
                            m
                            for m in unwrapped_list[start_idx].get("meta", [])
                            if "model_scale" in m
                        )
                        self.callouts[start_idx]["scale"] = float(
                            scale_meta["model_scale"]["values"][0]
                        )
                if self.continuous_step_numbers and e["callout"] == 0:
                    e["step"] = cont_step
                    cont_step += 1
                final_model.append(e)
            if self.continuous_step_numbers:
                self.continuous_step_count = cont_step - 1
                for e_fm in final_model:
                    if e_fm["callout"] == 0:
                        e_fm["num_steps"] = self.continuous_step_count
            for e_fm_pli in final_model:
                for meta in e_fm_pli.get("meta", []):
                    if (
                        "pli_proxy" in meta
                        and BOM
                        and BOMPart
                        and isinstance(e_fm_pli["pli_bom"], BOM)
                    ):
                        for item_str in meta["pli_proxy"].get("values", []):
                            pname, pcol = (
                                item_str.split("_")
                                if "_" in item_str
                                else (item_str, LDR_DEF_COLOUR)
                            )
                            e_fm_pli["pli_bom"].add_part(BOMPart(1, pname, int(pcol)))  # type: ignore
                            if self.bom:
                                self.bom.add_part(BOMPart(1, pname, int(pcol)))
            return final_model
        return unwrapped_list, current_idx

    def parse_model(
        self,
        model_source: Union[str, List[Dict[str, str]]],
        is_top_level: bool = True,
        mask_submodels: bool = False,
    ) -> Tuple[Dict[int, List[LDRPart]], Dict[int, Dict[str, Any]]]:
        model_content_str: str = ""
        if isinstance(model_source, str):
            if not is_top_level:
                model_key = model_source.lower().replace(".ldr", "")
                found_key = next(
                    (
                        k
                        for k in self.sub_model_str
                        if k.lower().replace(".ldr", "") == model_key
                    ),
                    None,
                )
                if found_key:
                    model_content_str = self.sub_model_str[found_key]
                else:
                    return {}, {}
            else:
                model_content_str = model_source
        else:
            return {}, {}
        if not model_content_str.strip():
            return {}, {}
        pli_dict: Dict[int, List[LDRPart]] = {}
        steps_dict: Dict[int, Dict[str, Any]] = {}
        step_blocks = model_content_str.split("0 STEP")
        cumulative_parts: List[LDRPart] = []
        current_aspect: List[float] = list(self.global_aspect if is_top_level else self.PARAMS["global_aspect"])  # type: ignore
        current_scale = self.global_scale if is_top_level else self.PARAMS["global_scale"]  # type: ignore
        model_inherent_scale = current_scale  # type: ignore
        step_num = 1
        for i, block_raw in enumerate(step_blocks):
            if i == 0 and not block_raw.strip() and len(step_blocks) > 1:
                continue
            meta = get_meta_commands(block_raw)
            aspect_changed = False
            for cmd in meta:
                if "scale" in cmd:
                    current_scale = float(cmd["scale"]["values"][0])
                elif "model_scale" in cmd:
                    model_inherent_scale = float(cmd["model_scale"]["values"][0])
                elif "rotation_abs" in cmd:
                    v = [float(x) for x in cmd["rotation_abs"]["values"]]
                    current_aspect = [-v[0], v[1], v[2]]
                    aspect_changed = True
                elif "rotation_rel" in cmd:
                    v_rel = tuple(float(x) for x in cmd["rotation_rel"]["values"])
                    current_aspect[0] -= v_rel[0]
                    current_aspect[1] += v_rel[1]
                    current_aspect[2] += v_rel[2]
                    current_aspect = list(norm_aspect(tuple(current_aspect)))  # type: ignore
                    aspect_changed = True
                elif "rotation_pre" in cmd:
                    current_aspect = list(preset_aspect(tuple(current_aspect), cmd["rotation_pre"]["values"]))  # type: ignore
                    aspect_changed = True
            part_dicts = get_parts_from_model(block_raw)
            added_ldr_parts: List[LDRPart] = []
            recursive_parse_model(
                part_dicts,
                self.sub_models,
                added_ldr_parts,
                reset_parts_list_on_call=True,
            )
            if not added_ldr_parts and not meta and (i > 0 or not block_raw.strip()):
                continue
            pli_step = self.transform_parts_to(
                added_ldr_parts,
                origin=(0, 0, 0),
                aspect=self.pli_aspect,
                use_exceptions=True,
            )
            cumulative_parts.extend(added_ldr_parts)
            snap_model = self.transform_parts(
                cumulative_parts, aspect=tuple(current_aspect)
            )
            snap_step_parts = self.transform_parts(
                added_ldr_parts, aspect=tuple(current_aspect)
            )
            pli_bom_step: Optional[BOM] = None
            if BOM and BOMPart:
                pli_bom_step = BOM()
                if self.bom and hasattr(self.bom, "ignore_parts"):
                    pli_bom_step.ignore_parts = self.bom.ignore_parts  # type: ignore
                for p in pli_step:
                    pli_bom_step.add_part(BOMPart(1, p.name, p.attrib.colour))  # type: ignore
                if is_top_level and self.bom:
                    for p in pli_step:
                        self.bom.add_part(BOMPart(1, p.name, p.attrib.colour))  # type: ignore
            sub_models_in_block = [
                pd["partname"] for pd in part_dicts if pd["partname"] in self.sub_models
            ]
            sub_parts_snap: Dict[str, List[LDRPart]] = {}
            for sub_n in unique_set(sub_models_in_block):
                sub_pds = [pd for pd in part_dicts if pd["partname"] == sub_n]
                temp_l: List[LDRPart] = []
                recursive_parse_model(
                    sub_pds, self.sub_models, temp_l, reset_parts_list_on_call=True
                )
                transformed_sub_parts = self.transform_parts(
                    temp_l, aspect=tuple(current_aspect)
                )
                if all(isinstance(tsp, LDRPart) for tsp in transformed_sub_parts):
                    sub_parts_snap[sub_n] = transformed_sub_parts  # type: ignore
            steps_dict[step_num] = {
                "parts": list(snap_model) if isinstance(snap_model, list) else [],
                "step_parts": (
                    list(snap_step_parts) if isinstance(snap_step_parts, list) else []
                ),
                "sub_models": sub_models_in_block,
                "aspect": tuple(current_aspect),
                "scale": current_scale,
                "model_scale": model_inherent_scale,
                "raw_ldraw": block_raw,
                "pli_bom": pli_bom_step if pli_bom_step else (BOM() if BOM else []),
                "meta": meta,
                "aspect_change": aspect_changed,
                "sub_parts": sub_parts_snap,
            }
            if pli_step:
                pli_dict[step_num] = pli_step
            step_num += 1
            if is_top_level:
                progress_bar(i + 1, len(step_blocks), "Parsing Model:", length=50)
        return pli_dict, steps_dict
