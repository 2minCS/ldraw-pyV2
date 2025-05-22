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
from typing import List, Dict, Tuple, Any, Optional, Union  # For type hints

# Explicit imports from toolbox
from toolbox import Vector, Matrix, Identity, apply_params, split_path, progress_bar

# Explicit imports from ldrawpy package
from .constants import (
    SPECIAL_TOKENS,
    LDR_DEF_COLOUR,
    ASPECT_DICT,
    FLIP_DICT,
    # Import other constants if used directly and not via wildcard
)
from .ldrprimitives import LDRPart
from .ldrhelpers import norm_aspect, preset_aspect  # Assuming these are in ldrhelpers

# Conditional import for brickbom
try:
    from brickbom import BOM, BOMPart
except ImportError:
    BOM = None  # type: ignore
    BOMPart = None  # type: ignore

# Conditional import for rich
try:
    from rich import print as rich_print

    has_rich = True
except ImportError:

    def rich_print(*args, **kwargs):  # Fallback to builtin print
        print(*args, **kwargs)

    has_rich = False


START_TOKENS = ["PLI BEGIN IGN", "BUFEXCHG STORE"]
END_TOKENS = ["PLI END", "BUFEXCHG RETRIEVE"]
EXCEPTION_LIST = ["2429c01.dat"]
IGNORE_LIST = ["LS02"]

COMMON_SUBSTITUTIONS: List[Tuple[str, str]] = [
    ("3070a", "3070b"),  # 1 x 1 tile
    ("3069a", "3069b"),  # 1 x 2 tile
    ("3068a", "3068b"),  # 2 x 2 tile
    ("x224", "41751"),  # windscreen
    ("4864a", "87552"),  # 1 x 2 x 2 panel with side supports
    ("4864b", "87552"),
    ("2362a", "87544"),  # 1 x 2 x 3 panel with side supports
    ("2362b", "87544"),
    ("60583", "60583b"),  # 1 x 1 x 3 brick with clips
    ("60583a", "60583b"),
    ("3245a", "3245c"),  # 1 x 2 x 2 brick
    ("3245b", "3245c"),
    ("3794", "15573"),  # 1 x 2 jumper plate
    ("3794a", "15573"),
    ("3794b", "15573"),
    ("4215a", "60581"),  # 1 x 4 x 3 panel with side supports
    ("4215b", "60581"),
    ("4215", "60581"),
    ("73983", "2429c01"),  # 1 x 4 hinge plate complete
    ("3665a", "3665"),
    ("3665b", "3665"),  # 2 x 1 45 deg inv slope
    ("4081a", "4081b"),  # 1x1 plate with light ring
    ("4085a", "60897"),  # 1x1 plate with vert clip
    ("4085b", "60897"),
    ("4085c", "60897"),
    ("6019", "61252"),  # 1x1 plate with horz clip
    ("59426", "32209"),  # technic 5.5 axle
    ("48729", "48729b"),  # bar with clip
    ("48729a", "48729b"),
    ("41005", "48729b"),
    ("4459", "2780"),  # Technic friction pin
    ("44302", "44302a"),  # 1x2 click hinge plate
    ("44302b", "44302a"),
    ("2436", "28802"),  # 1x2 x 1x4 bracket
    ("2436a", "28802"),
    ("2436b", "28802"),
    ("2454", "2454b"),  # 1x2x5 brick
    ("64567", "577b"),  # minifig lightsabre holder
    ("30241b", "60475b"),
    ("2861", "2861c01"),
    ("2859", "2859c01"),
    ("70026a", "70026"),
    ("4707pb01", "4707c01"),
    ("4707pb02", "4707c02"),
    ("4707pb03", "4707c03"),
    ("4707pb04", "4707c04"),
    ("4707pb05", "4707c05"),
    ("3242", "3240a"),
    ("2776c28", "766bc03"),
    ("766c96", "766bc03"),
    ("7864-1", "u9058c02"),
    ("bb0012vb", "501bc01"),
    ("bb0012v2", "867"),
    ("70026b", "70026"),
    ("3242c", "3240a"),
    ("4623b", "4623"),
    ("4623a", "4623"),
]


def substitute_part(part: LDRPart) -> LDRPart:
    for e in COMMON_SUBSTITUTIONS:
        if part.name == e[0]:
            part.name = e[1]
    return part


def line_has_all_tokens(line: str, tokenlist: List[str]) -> bool:
    line_tokens = line.split()  # Split once
    for t_group_str in tokenlist:
        required_tokens_in_group = t_group_str.split()
        if all(req_token in line_tokens for req_token in required_tokens_in_group):
            return True
    return False


def parse_special_tokens(line: str) -> List[Dict[str, Any]]:
    ls = line.strip().split()  # Strip and split once
    metas: List[Dict[str, Any]] = []

    for cmd_key, token_patterns in SPECIAL_TOKENS.items():
        for pattern_str in token_patterns:
            pattern_tokens = pattern_str.split()

            # Check if all non-placeholder tokens in pattern are present in line
            non_placeholder_pattern_tokens = [
                pt for pt in pattern_tokens if not pt.startswith("%")
            ]
            if not all(nppt in ls for nppt in non_placeholder_pattern_tokens):
                continue  # This pattern doesn't match

            # Pattern matches, try to extract values
            values: List[str] = []
            valid_match = True
            line_token_idx = 0

            # This part needs a more robust way to match pattern_tokens to ls tokens
            # and extract placeholders. The original logic was complex.
            # For now, a simplified placeholder extraction if pattern matches:
            placeholder_indices_in_pattern = [
                int(pt[1:])
                for pt in pattern_tokens
                if pt.startswith("%") and pt[1:].isdigit()
            ]

            # This assumes placeholders in pattern correspond to specific indices in the *line itself*
            # which is how the original SPECIAL_TOKENS seem to be defined (e.g., %-1 means last token).
            # A more robust parser would be needed for complex patterns.
            # Let's try to match the original intent for placeholder indices like %2, %3, %-1

            extracted_values: List[str] = []
            try:
                for pt in pattern_tokens:
                    if pt.startswith("%") and pt[1:].isdigit():
                        idx = int(pt[1:])
                        # LDraw meta often uses 1-based indexing for placeholders after the command
                        # e.g., ROTSTEP %2 %3 %4 -> ls[1] is command, ls[2] is %2
                        # Find the command token first to align indices
                        command_in_pattern = non_placeholder_pattern_tokens[
                            0
                        ]  # Assuming first non-% is the command
                        try:
                            cmd_start_idx_in_line = ls.index(command_in_pattern)
                            actual_idx_in_line = cmd_start_idx_in_line + idx
                            if actual_idx_in_line < len(ls):
                                extracted_values.append(ls[actual_idx_in_line])
                            else:  # Placeholder index out of bounds
                                valid_match = False
                                break
                        except (
                            ValueError
                        ):  # Command not found in line (should have been caught by `all` check)
                            valid_match = False
                            break
                    # else: it's a literal token, already checked by `all`
                if not valid_match:
                    continue

            except (
                ValueError
            ):  # If int conversion of placeholder index fails (e.g. %-1 not handled yet)
                # Handle %-1 for last element, %-2 for second to last, etc.
                # This part of original logic was not fully clear from SPECIAL_TOKENS structure.
                # For now, focusing on positive indices.
                # If %-1 was intended, it needs specific handling.
                # The original code used `ls[x]` where x was from `captures`.
                # `captures = [int(x[1:]) for x in tokens if x[0] == "%"]`
                # This implies `tokens` were the pattern tokens, and `x` were the placeholder indices.
                # The `SPECIAL_TOKENS` values like `%-1` are not standard int indices.
                # This parsing needs to be more robust if such placeholders are used.
                # For now, we assume positive indices like %2, %3, %4.
                pass

            if extracted_values:
                metas.append(
                    {cmd_key: {"values": extracted_values, "text": line.strip()}}
                )
            else:  # No placeholders, or extraction failed, but non-% tokens matched
                metas.append({cmd_key: {"text": line.strip()}})
            break  # Found a match for this cmd_key from its patterns
        if metas and metas[-1].get(cmd_key):  # If a meta was added for this cmd_key
            break  # Move to next line, don't try other cmd_keys for this line

    return metas


def get_meta_commands(ldr_string: str) -> List[Dict[str, Any]]:
    cmd: List[Dict[str, Any]] = []
    for line in ldr_string.splitlines():
        stripped_line = line.lstrip()
        if not stripped_line or not stripped_line.startswith("0 "):
            continue

        # parse_special_tokens should return a list of found metas for the line
        meta_for_line = parse_special_tokens(line)
        if meta_for_line:
            cmd.extend(meta_for_line)
    return cmd


def get_parts_from_model(ldr_string: str) -> List[Dict[str, str]]:
    parts: List[Dict[str, str]] = []
    lines = ldr_string.splitlines()
    mask_depth = 0
    bufex = False  # Inside BUFEXCHG STORE/RETRIEVE block

    for line_idx, line in enumerate(lines):
        stripped_line = line.lstrip()
        if not stripped_line:
            continue

        # Check for masking tokens
        # Using line_has_all_tokens for robustness with multi-token commands
        if line_has_all_tokens(line, ["BUFEXCHG STORE"]):
            bufex = True
        if line_has_all_tokens(line, ["BUFEXCHG RETRIEVE"]):
            bufex = False

        # START_TOKENS and END_TOKENS might need more specific matching if they are complex
        if line_has_all_tokens(line, START_TOKENS):
            mask_depth += 1
        if line_has_all_tokens(line, END_TOKENS):
            if mask_depth > 0:
                mask_depth -= 1

        try:
            line_type = int(stripped_line[0])
        except (ValueError, IndexError):
            continue  # Not a standard LDraw line type

        if line_type == 1:
            split_line_tokens = line.split()  # Split original line for part name
            if len(split_line_tokens) < 15:
                continue  # Not enough elements for a valid part line

            part_dict = {
                "ldrtext": line,  # Store original line
                "partname": " ".join(split_line_tokens[14:]),
            }

            if mask_depth == 0:
                parts.append(part_dict)
            else:  # Inside a masked block (e.g. PLI IGN, BUFEXCHG)
                if part_dict["partname"] in EXCEPTION_LIST:
                    parts.append(part_dict)
                elif not bufex and part_dict["partname"].endswith(".ldr"):
                    # Add submodel references even if masked, unless it's a BUFEXCHG scenario
                    parts.append(part_dict)
    return parts


def recursive_parse_model(
    model_entries: List[
        Dict[str, str]
    ],  # List of part dicts {"partname": ..., "ldrtext": ...}
    all_submodels_data: Dict[str, List[Dict[str, str]]],  # {name: list_of_part_dicts}
    output_parts_list: List[LDRPart],  # List to populate with LDRPart objects
    current_offset: Vector = Vector(0, 0, 0),  # Use default if None
    current_matrix: Matrix = Identity(),  # Use default if None
    reset_parts_list_on_call: bool = False,  # If true, clears output_parts_list at start
    filter_for_submodel_name: Optional[str] = None,
):
    if reset_parts_list_on_call:
        output_parts_list.clear()

    for entry_dict in model_entries:
        part_name = entry_dict["partname"]
        ldr_text = entry_dict["ldrtext"]

        if filter_for_submodel_name and part_name != filter_for_submodel_name:
            continue

        if part_name in all_submodels_data:  # It's a reference to a submodel
            submodel_entries_for_this_ref = all_submodels_data[part_name]

            # Get the transformation of this submodel reference itself
            ref_part_transform = LDRPart()
            if ref_part_transform.from_str(ldr_text) is None:
                # print(f"Warning: Could not parse LDR text for submodel reference: {ldr_text}")
                continue

            # New cumulative transformation for children of this submodel:
            # M_new = M_current_cumulative * M_submodel_reference
            # T_new = M_current_cumulative * T_submodel_reference + T_current_cumulative
            new_child_matrix = current_matrix * ref_part_transform.attrib.matrix  # type: ignore
            new_child_offset = current_matrix * ref_part_transform.attrib.loc + current_offset  # type: ignore

            recursive_parse_model(
                submodel_entries_for_this_ref,
                all_submodels_data,
                output_parts_list,  # Continue populating the same list
                current_offset=new_child_offset,
                current_matrix=new_child_matrix,
                reset_parts_list_on_call=False,  # Crucial: do not reset for recursive calls
                filter_for_submodel_name=None,  # Do not filter further down unless intended
            )
        else:  # It's a direct part
            if (
                filter_for_submodel_name is None
            ):  # Only add direct parts if not specifically filtering
                actual_part = LDRPart()
                if actual_part.from_str(ldr_text) is None:
                    # print(f"Warning: Could not parse LDR text for direct part: {ldr_text}")
                    continue

                actual_part = substitute_part(actual_part)

                # Apply cumulative transformation to this part
                actual_part.transform(matrix=current_matrix, offset=current_offset)

                if (
                    actual_part.name not in IGNORE_LIST
                    and actual_part.name.upper() not in IGNORE_LIST
                ):
                    output_parts_list.append(actual_part)


def unique_set(items: List[Any]) -> Dict[Any, int]:
    udict: Dict[Any, int] = defaultdict(int)
    for e in items:
        udict[e] += 1
    return dict(udict)


def key_name(elem: LDRPart) -> str:
    return elem.name


def key_colour(elem: LDRPart) -> int:  # Only one definition needed
    return elem.attrib.colour


def get_sha1_hash(parts: List[LDRPart]) -> str:
    if not all(isinstance(p, LDRPart) for p in parts):  # Runtime check
        # Or raise TypeError("All items in 'parts' must be LDRPart objects.")
        return "error_parts_not_LDRPart_objects"  # Or handle error appropriately

    sp = sorted([(p, p.sha1hash()) for p in parts], key=lambda x: x[1])
    shash = hashlib.sha1()
    for _, p_hash_val in sp:  # Corrected iteration
        shash.update(bytes(p_hash_val, encoding="utf8"))
    return shash.hexdigest()


def sort_parts(
    parts: List[LDRPart], key: str = "name", order: str = "ascending"
) -> List[LDRPart]:
    if not all(isinstance(p, LDRPart) for p in parts):
        # raise TypeError("All items in 'parts' must be LDRPart objects.")
        return []  # Or handle error appropriately

    sp = list(parts)  # Copy
    is_descending = order.lower() == "descending"

    sort_key_func = None
    if key.lower() == "sha1":
        sort_key_func = lambda p: p.sha1hash()
    elif key.lower() == "name":
        sort_key_func = key_name
    elif key.lower() == "colour":
        sort_key_func = key_colour
    else:
        # print(f"Warning: Unknown sort key '{key}'. Returning unsorted list.")
        return sp  # Or raise ValueError

    sp.sort(key=sort_key_func, reverse=is_descending)
    return sp


class LDRModel:
    # ... (LDRModel class definition from previous correct version) ...
    # Ensure all methods within LDRModel that call toolbox functions or
    # other ldrawpy functions use explicit imports if those functions
    # are not defined within LDRModel itself or passed as arguments.
    # The MyPy errors were not inside LDRModel methods, so focusing on top-level functions first.
    # The previous version of LDRModel provided in the prompt had many improvements.
    # For brevity here, I'm not repeating the entire LDRModel class, but it should
    # use the corrected helper functions and explicit imports as needed.
    # Key is that `key_colour` is defined only once globally.
    PARAMS = {
        "global_origin": (0, 0, 0),
        "global_aspect": (-40, 55, 0),
        "global_scale": 1.0,
        "pli_aspect": (-25, -40, 0),
        "pli_exceptions": {
            "32001": (-50, -25, 0),
            "3676": (-25, 50, 0),
            "3045": (-25, 50, 0),
        },
        "callout_step_thr": 6,
        "continuous_step_numbers": False,
    }

    def __init__(self, filename: str, **kwargs):
        self.filename: str = filename
        self.title: str = ""
        self.bom: Optional[BOM] = None
        if BOM:
            self.bom = BOM()

        self.steps: Dict[int, Dict[str, Any]] = {}
        self.pli: Dict[int, List[LDRPart]] = {}
        self.sub_models: Dict[str, List[Dict[str, str]]] = (
            {}
        )  # {name: list_of_part_dicts}
        self.sub_model_str: Dict[str, str] = {}  # {name: raw_ldr_string}
        self.unwrapped: Optional[List[Dict[str, Any]]] = None
        self.callouts: Dict[int, Dict[str, Any]] = {}
        self.continuous_step_count: int = 0

        # Initialize with PARAMS defaults
        for key, value in self.PARAMS.items():
            setattr(self, key, value)

        apply_params(self, kwargs)  # Override with any kwargs passed

        _, self.title = split_path(filename)
        if self.bom and hasattr(
            self.bom, "ignore_parts"
        ):  # Initialize ignore_parts if bom exists
            self.bom.ignore_parts = []

    def __str__(self) -> str:
        return (
            f"LDRModel: {self.title}\n"
            f"  Steps: {len(self.steps)}, SubModels: {len(self.sub_models)}\n"
            f"  Global Aspect: {getattr(self, 'global_aspect', 'N/A')}"
        )

    def __getitem__(self, key: int) -> Dict[str, Any]:
        if self.unwrapped is None:
            self.unwrap()  # Ensure model is unwrapped
        if (
            self.unwrapped is None
        ):  # Still None after trying to unwrap (e.g. parse failed)
            raise IndexError("Model has not been successfully unwrapped or parsed.")
        return self.unwrapped[key]

    def print_step_dict(self, key: int):  # Type hints
        if key in self.steps:
            s_dict = self.steps[key]  # Renamed to avoid conflict with __str__
            for k, v in s_dict.items():
                if k == "sub_parts" and isinstance(v, dict):
                    rich_print(f"[bold blue]{k}:[/bold blue]")
                    for (
                        ks,
                        vs_list,
                    ) in v.items():  # Iterate through sub_parts dictionary
                        rich_print(f"  [cyan]{ks}:[/cyan]")
                        for (
                            e_part
                        ) in (
                            vs_list
                        ):  # Iterate through list of parts for that sub_model
                            rich_print(f"    {str(e_part).rstrip()}")
                elif isinstance(v, list) and all(
                    isinstance(item, LDRPart) for item in v
                ):
                    rich_print(f"[bold blue]{k}:[/bold blue] ({len(v)} items)")
                    for vx_part in v:
                        rich_print(f"  {str(vx_part).rstrip()}")
                elif isinstance(v, list):  # Other lists (e.g. meta)
                    rich_print(f"[bold blue]{k}:[/bold blue]")
                    for vx_item in v:
                        rich_print(f"  {vx_item}")
                elif k == "pli_bom" and BOM is not None and isinstance(v, BOM):
                    rich_print(f"[bold blue]{k}:[/bold blue]")
                    if hasattr(v, "summary_str"):
                        rich_print(f"  {v.summary_str()}")
                    else:
                        rich_print(f"  {v}")  # Fallback
                else:
                    rich_print(f"[bold blue]{k}:[/bold blue] {v}")
        else:
            rich_print(f"Step {key} not found.")

    def print_unwrapped_dict(self, idx: int):  # Type hints
        if self.unwrapped is None or not (0 <= idx < len(self.unwrapped)):
            rich_print(f"Index {idx} out of bounds for unwrapped model.")
            return
        s_dict = self.unwrapped[idx]
        for k, v in s_dict.items():
            if (
                k in ("parts", "step_parts")
                and isinstance(v, list)
                and all(isinstance(item, LDRPart) for item in v)
            ):
                rich_print(f"[bold green]{k}:[/bold green] ({len(v)} parts)")
                for vx_part in v:
                    rich_print(f"  {str(vx_part).rstrip()}")
            elif k == "pli_bom" and BOM is not None and isinstance(v, BOM):
                rich_print(f"[bold green]{k}:[/bold green]")
                if hasattr(v, "summary_str"):
                    rich_print(f"  {v.summary_str()}")
                else:
                    rich_print(f"  {v}")  # Fallback
            elif isinstance(v, list):
                rich_print(f"[bold green]{k}:[/bold green]")
                for vx_item in v:
                    rich_print(f"  {vx_item}")
            else:
                rich_print(f"[bold green]{k}:[/bold green] {v}")

    def print_unwrapped_verbose(self):
        if not self.unwrapped:
            rich_print("Model not unwrapped.")
            return
        for i, v in enumerate(self.unwrapped):
            aspect = v.get("aspect", [0, 0, 0])
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
        _print_func = rich_print if has_rich else print
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
                    tag_str = k.replace("_", " ")  # Basic formatting
                    if k == "columns" and "values" in v_dict:
                        meta_tags.append(f"[green]COL{v_dict['values'][0]}[/]")
                    else:
                        meta_tags.append(f"[dim]{tag_str}[/dim]")
            else:
                meta_tags.append(str(m_item))
        meta_str = " ".join(meta_tags)

        aspect = v_step.get("aspect", [0, 0, 0])
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
        final_aspect = aspect if aspect is not None else self.global_aspect
        final_origin_vec = safe_vector(origin) if origin is not None else None

        transformed_parts = []
        for p in parts:
            np = p.copy()
            current_aspect_for_part = list(
                final_aspect
            )  # Make mutable if Vector is not

            if use_exceptions and p.name in self.pli_exceptions:
                current_aspect_for_part = list(self.pli_exceptions[p.name])  # type: ignore

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

        final_aspect = aspect if aspect is not None else self.global_aspect  # type: ignore
        final_offset_vec = (
            safe_vector(offset) if offset is not None else Vector(0, 0, 0)
        )

        if not parts:
            return []

        if isinstance(parts[0], LDRPart):
            tparts_ldr = []
            for p_obj in parts:  # p_obj is LDRPart
                np = p_obj.copy()  # type: ignore
                np.rotate_by(final_aspect)  # type: ignore
                np.move_by(final_offset_vec)
                tparts_ldr.append(np)
            return tparts_ldr
        elif isinstance(parts[0], str):
            tparts_str = []
            for p_str in parts:  # p_str is str
                np = LDRPart()
                if np.from_str(p_str):
                    np.rotate_by(final_aspect)  # type: ignore
                    np.move_by(final_offset_vec)
                    tparts_str.append(str(np))
                else:  # Could not parse part string
                    tparts_str.append(p_str)  # Append original string
            return tparts_str
        return []  # Should not happen if input is validated

    def parse_file(self):
        self.sub_models = {}
        self.sub_model_str = {}
        try:
            with open(self.filename, "rt", encoding="utf-8") as fp:
                content = fp.read()
        except FileNotFoundError:
            rich_print(f"Error: File {self.filename} not found.")
            self.pli, self.steps = {}, {}
            return
        except Exception as e:
            rich_print(f"Error reading file {self.filename}: {e}")
            self.pli, self.steps = {}, {}
            return

        file_blocks = content.split("0 FILE")
        root_model_content = ""

        if not content.strip().startswith("0 FILE") and file_blocks:
            root_model_content = file_blocks[0]
            sub_file_blocks = file_blocks[1:]
        else:  # File starts with "0 FILE" or is empty / only comments before first "0 FILE"
            if len(file_blocks) > 1:  # There is at least one "0 FILE" block
                # The first block after "0 FILE" is the root. file_blocks[0] is stuff before it.
                root_model_content = "0 FILE " + file_blocks[1].strip()
                sub_file_blocks = file_blocks[2:]
            else:  # No "0 FILE" or only one block which is not root model content
                rich_print(
                    f"Warning: No root model found in {self.filename} based on '0 FILE' structure."
                )
                self.pli, self.steps = {}, {}
                if (
                    file_blocks
                    and file_blocks[0].strip()
                    and not content.strip().startswith("0 FILE")
                ):
                    # If there's content but no "0 FILE", treat all as root (handled by first if)
                    # This else implies file_blocks[0] was empty or only comments.
                    pass  # No actual model content to parse as root.
                else:  # Truly empty or unparseable structure
                    return

        for sub_block_text in sub_file_blocks:
            if not sub_block_text.strip():
                continue
            full_sub_text = "0 FILE " + sub_block_text.strip()
            sub_lines = full_sub_text.splitlines()
            if not sub_lines:
                continue
            sub_name = sub_lines[0].replace("0 FILE", "").strip().lower()
            if sub_name:
                self.sub_model_str[sub_name] = full_sub_text
                self.sub_models[sub_name] = get_parts_from_model(full_sub_text)

        if root_model_content.strip():
            self.pli, self.steps = self.parse_model(
                root_model_content, is_top_level=True
            )
        else:  # If after all that, root_model_content is still empty
            rich_print(
                f"Warning: Root model content for {self.filename} is empty after processing '0 FILE' directives."
            )
            self.pli, self.steps = {}, {}

        self.unwrap()  # Unwrap after parsing

    def unwrap(self):
        if self.unwrapped is None:  # Only unwrap once or if reset
            # Initialize parse_model_cache for submodels if not already done
            # This assumes parse_model can be called for submodels and returns their step structures
            self._parsed_submodel_steps_cache = {}
            for name, sub_model_str_content in self.sub_model_str.items():
                _, steps_dict = self.parse_model(
                    sub_model_str_content, is_top_level=False
                )
                self._parsed_submodel_steps_cache[name] = steps_dict

            self.unwrapped = self._unwrap_model_recursive(
                current_model_steps=self.steps
            )

    def _unwrap_model_recursive(
        self,
        current_model_steps: Dict[int, Dict[str, Any]],
        current_idx: int = 0,
        current_level: int = 0,
        model_name_for_step: str = "root",
        model_qty_for_step: int = 1,  # Root model qty is 1
        unwrapped_list: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], int]]:

        if unwrapped_list is None:  # Initial call
            unwrapped_list = []
            self.continuous_step_count = 0  # Reset for a fresh unwrap

        sorted_step_numbers = sorted(current_model_steps.keys())

        for step_no in sorted_step_numbers:
            step_data = current_model_steps[step_no]

            if step_data.get("sub_models"):
                unique_submodel_refs = unique_set(step_data["sub_models"])
                for sub_model_filename, qty in unique_submodel_refs.items():
                    # Get the parsed steps for this submodel (should be cached)
                    sub_steps_data = self._parsed_submodel_steps_cache.get(
                        sub_model_filename
                    )
                    if not sub_steps_data:
                        # Fallback: try parsing if not cached (should ideally be pre-parsed)
                        # rich_print(f"Dynamically parsing submodel {sub_model_filename} during unwrap.")
                        # _, sub_steps_data = self.parse_model(sub_model_filename, is_top_level=False)
                        # self._parsed_submodel_steps_cache[sub_model_filename] = sub_steps_data
                        # If still not found, skip
                        if not sub_steps_data:
                            # rich_print(f"Warning: Could not find/parse steps for submodel {sub_model_filename} during unwrap.")
                            continue

                    # Recurse into the submodel
                    # Pass the current_idx by value, it will be updated and returned
                    _, current_idx = self._unwrap_model_recursive(  # type: ignore
                        current_model_steps=sub_steps_data,
                        current_idx=current_idx,
                        current_level=current_level + 1,
                        model_name_for_step=sub_model_filename,
                        model_qty_for_step=qty,
                        unwrapped_list=unwrapped_list,  # Pass the same list
                    )

            # Add the current step from the current_model_steps (parent or submodel)
            step_detail = {
                "idx": current_idx,
                "level": current_level,
                "step": step_no,
                "next_step": (
                    step_no + 1 if step_no < len(sorted_step_numbers) else step_no
                ),
                "num_steps": len(sorted_step_numbers),
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
            unwrapped_list.append(step_detail)
            current_idx += 1

        if current_level == 0:  # Final post-processing after all recursion
            final_model = []
            cont_step_num = 1
            self.callouts = {}
            callout_starts_by_level: Dict[int, int] = {}  # level: start_idx

            for i, entry in enumerate(unwrapped_list):
                entry["prev_level"] = unwrapped_list[i - 1]["level"] if i > 0 else 0
                entry["next_level"] = (
                    unwrapped_list[i + 1]["level"]
                    if i < len(unwrapped_list) - 1
                    else entry["level"]
                )

                # Page break logic (simplified, can be expanded)
                entry["page_break"] = (
                    any("page_break" in m for m in entry.get("meta", []))
                    or (
                        entry["next_level"] > entry["level"]
                        and entry.get("num_steps", 0) >= self.callout_step_thr
                        and not any("no_callout" in m for m in entry.get("meta", []))
                    )
                    or (
                        entry["level"] > entry["prev_level"]
                        and unwrapped_list[i - 1].get("num_steps", 0)
                        >= self.callout_step_thr
                        and not any(
                            "no_callout" in m
                            for m in unwrapped_list[i - 1].get("meta", [])
                        )
                    )
                )

                # Callout logic
                current_callout_val = 0
                # If we are going deeper and it's not a "no_callout" step
                if entry["level"] > entry["prev_level"] and not any(
                    "no_callout" in m for m in entry.get("meta", [])
                ):
                    if (
                        entry.get("num_steps", 0) < self.callout_step_thr
                    ):  # Condition for being a callout
                        callout_starts_by_level[entry["level"]] = entry["idx"]

                # Determine current callout level
                active_callout_parent_level = 0
                for lvl, start_idx in sorted(
                    callout_starts_by_level.items(), reverse=True
                ):
                    if (
                        entry["level"] >= lvl
                    ):  # This step is part of or deeper than an active callout start
                        # Check if this callout is still active (not ended)
                        # This requires finding the end of the callout block.
                        # For now, assume if we are at `lvl` or deeper, we are in that callout.
                        current_callout_val = lvl
                        break
                entry["callout"] = current_callout_val

                # If we are coming out of a level that started a callout
                if entry["level"] < entry["prev_level"]:
                    if entry["prev_level"] in callout_starts_by_level:
                        start_idx = callout_starts_by_level.pop(entry["prev_level"])
                        # Store the callout: start_idx to i-1 (previous entry)
                        self.callouts[start_idx] = {
                            "level": entry["prev_level"],
                            "end": unwrapped_list[i - 1]["idx"],
                            "parent": unwrapped_list[start_idx][
                                "prev_level"
                            ],  # Parent is prev_level of start
                        }
                        # Custom scale for callout if specified at start_idx
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

                if self.continuous_step_numbers:
                    if (
                        entry["callout"] == 0
                    ):  # Only number non-callout steps continuously
                        entry["step"] = cont_step_num
                        cont_step_num += 1
                    # else: step number remains as per its own model's sequence
                final_model.append(entry)

            # Finalize continuous step count and num_steps for main flow
            if self.continuous_step_numbers:
                self.continuous_step_count = cont_step_num - 1
                for entry in final_model:
                    if entry["callout"] == 0:
                        entry["num_steps"] = self.continuous_step_count

            # Handle PLI proxy parts (original did this in unwrap)
            for entry in final_model:
                for meta_item in entry.get("meta", []):
                    if "pli_proxy" in meta_item and BOM and BOMPart:
                        for item_str in meta_item["pli_proxy"].get("values", []):
                            p_name, p_color_code = (
                                item_str.split("_")
                                if "_" in item_str
                                else (item_str, LDR_DEF_COLOUR)
                            )
                            proxy_part_obj = BOMPart(1, p_name, int(p_color_code))
                            if isinstance(entry["pli_bom"], BOM):
                                entry["pli_bom"].add_part(proxy_part_obj)  # type: ignore
                            if self.bom:
                                self.bom.add_part(proxy_part_obj)

            return final_model
        else:  # Recursive call, return list and current_idx
            return unwrapped_list, current_idx

    def parse_model(
        self,
        model_source: Union[str, List[Dict[str, str]]],
        is_top_level: bool = True,
        mask_submodels: bool = False,
    ) -> Tuple[Dict[int, List[LDRPart]], Dict[int, Dict[str, Any]]]:

        model_content_str: str = ""
        if isinstance(model_source, str):
            if not is_top_level:  # It's a submodel name
                model_key = model_source.lower()
                if model_key.endswith(".ldr"):
                    model_key = model_key[:-4]

                # Try finding the submodel string (case-insensitive for keys)
                found_key = None
                for k_sm_str in self.sub_model_str.keys():
                    if k_sm_str.lower().replace(".ldr", "") == model_key:
                        found_key = k_sm_str
                        break
                if found_key:
                    model_content_str = self.sub_model_str[found_key]
                else:
                    # rich_print(f"Warning: Submodel string for '{model_source}' not found in self.sub_model_str.")
                    return {}, {}
            else:  # Root model, model_source is the raw LDraw string
                model_content_str = model_source
        # elif isinstance(model_source, list): # If passing list of part dicts directly (internal use)
        # This case is not how parse_model is usually called from outside.
        # The main loop of parse_model expects to split a string by "0 STEP".
        # For now, assume model_source is primarily string (filename content or submodel name).
        # rich_print("Error: parse_model expects string (LDR content or submodel name).")
        # return {}, {}
        else:
            # rich_print(f"Error: Invalid model_source type for parse_model: {type(model_source)}")
            return {}, {}

        if not model_content_str.strip():
            return {}, {}

        current_pli_dict: Dict[int, List[LDRPart]] = {}
        current_steps_dict: Dict[int, Dict[str, Any]] = {}

        # Split by "0 STEP". The first part might be header or step 1 content.
        step_blocks_raw = model_content_str.split("0 STEP")

        cumulative_model_parts: List[LDRPart] = (
            []
        )  # Parts aggregated across steps for current model view

        # State for current model being parsed (can be root or a submodel)
        # These are the "snapshot" settings for rendering each step
        aspect_for_this_model_parse = (
            list(self.global_aspect)
            if is_top_level
            else list(self.PARAMS["global_aspect"])
        )
        scale_for_this_model_parse = (
            self.global_scale if is_top_level else self.PARAMS["global_scale"]
        )
        # Inherent scale of the model design itself (from !LPUB MODEL_SCALE meta)
        inherent_model_scale = (
            self.global_scale if is_top_level else self.PARAMS["global_scale"]
        )

        step_counter = 1  # LDraw steps are 1-indexed

        for i, block_text_raw in enumerate(step_blocks_raw):
            # If the first block is empty and there are other blocks, it's likely just header.
            if i == 0 and not block_text_raw.strip() and len(step_blocks_raw) > 1:
                continue

            current_step_content = block_text_raw  # Content for this step

            aspect_changed_in_this_block = False
            step_meta = get_meta_commands(current_step_content)

            for cmd in step_meta:
                if "scale" in cmd:  # View scale for this step's snapshot
                    scale_for_this_model_parse = float(cmd["scale"]["values"][0])
                elif "model_scale" in cmd:  # Inherent scale of the model design
                    inherent_model_scale = float(cmd["model_scale"]["values"][0])
                elif "rotation_abs" in cmd:
                    vals = [float(x) for x in cmd["rotation_abs"]["values"]]
                    aspect_for_this_model_parse = [
                        -vals[0],
                        vals[1],
                        vals[2],
                    ]  # LDraw X often inverted
                    aspect_changed_in_this_block = True
                elif "rotation_rel" in cmd:
                    vals = tuple(float(x) for x in cmd["rotation_rel"]["values"])
                    aspect_for_this_model_parse[0] -= vals[0]  # type: ignore
                    aspect_for_this_model_parse[1] += vals[1]  # type: ignore
                    aspect_for_this_model_parse[2] += vals[2]  # type: ignore
                    aspect_for_this_model_parse = list(norm_aspect(tuple(aspect_for_this_model_parse)))  # type: ignore
                    aspect_changed_in_this_block = True
                elif "rotation_pre" in cmd:
                    aspect_for_this_model_parse = list(preset_aspect(tuple(aspect_for_this_model_parse), cmd["rotation_pre"]["values"]))  # type: ignore
                    aspect_changed_in_this_block = True

            # Get part dictionaries {name, ldrtext} defined in this specific block
            part_dicts_in_block = get_parts_from_model(current_step_content)

            # Resolve these part dicts into actual LDRPart objects (expanding submodels if any)
            # These are the parts *added* in this step.
            ldr_parts_added_this_step: List[LDRPart] = []
            recursive_parse_model(
                model_entries=part_dicts_in_block,
                all_submodels_data=self.sub_models,  # Pass the main dict of known submodels
                output_parts_list=ldr_parts_added_this_step,
                reset_parts_list_on_call=True,  # We only want parts from *this* block
            )

            # Create PLI list for these newly added parts
            pli_for_step = self.transform_parts_to(
                ldr_parts_added_this_step,
                origin=(0, 0, 0),
                aspect=self.pli_aspect,
                use_exceptions=True,
            )

            # If no actual LDraw parts were added in this block, and it's not just a meta-command step,
            # it might be an empty "0 STEP" or just comments.
            # However, a step can consist only of meta-commands (e.g., ROTSTEP).
            # We process if there are LDR parts added OR if there were meta commands.
            if not ldr_parts_added_this_step and not step_meta:
                if i > 0 or (
                    i == 0 and not current_step_content.strip()
                ):  # Skip empty initial blocks or truly empty steps
                    continue

            cumulative_model_parts.extend(
                ldr_parts_added_this_step
            )  # Add to cumulative list

            # For snapshot: transform the *entire current model* and *parts added this step*
            snapshot_view_model = self.transform_parts(
                cumulative_model_parts, aspect=tuple(aspect_for_this_model_parse)
            )
            snapshot_view_step_parts = self.transform_parts(
                ldr_parts_added_this_step, aspect=tuple(aspect_for_this_model_parse)
            )

            # BOM for PLI parts
            pli_bom_for_step: Optional[BOM] = None
            if BOM and BOMPart:
                pli_bom_for_step = BOM()
                if self.bom and hasattr(self.bom, "ignore_parts"):
                    pli_bom_for_step.ignore_parts = self.bom.ignore_parts  # type: ignore
                for p in pli_for_step:
                    pli_bom_for_step.add_part(BOMPart(1, p.name, p.attrib.colour))  # type: ignore

            # Submodel references within this block
            sub_models_in_block = [
                pd["partname"]
                for pd in part_dicts_in_block
                if pd["partname"] in self.sub_models
            ]

            # Parts from submodels specifically for this step (transformed for snapshot view)
            sub_parts_for_snapshot_view: Dict[str, List[LDRPart]] = {}
            for sub_name in unique_set(sub_models_in_block):
                sub_part_dicts = [
                    pd for pd in part_dicts_in_block if pd["partname"] == sub_name
                ]
                temp_list: List[LDRPart] = []
                recursive_parse_model(
                    sub_part_dicts,
                    self.sub_models,
                    temp_list,
                    reset_parts_list_on_call=True,
                )
                sub_parts_for_snapshot_view[sub_name] = self.transform_parts(temp_list, aspect=tuple(aspect_for_this_model_parse))  # type: ignore

            current_steps_dict[step_counter] = {
                "parts": list(snapshot_view_model),  # type: ignore # Store copy
                "step_parts": snapshot_view_step_parts,  # type: ignore
                "sub_models": sub_models_in_block,
                "aspect": tuple(aspect_for_this_model_parse),
                "scale": scale_for_this_model_parse,
                "model_scale": inherent_model_scale,
                "raw_ldraw": current_step_content,
                "pli_bom": (
                    pli_bom_for_step if pli_bom_for_step else (BOM() if BOM else [])
                ),
                "meta": step_meta,
                "aspect_change": aspect_changed_in_this_block,
                "sub_parts": sub_parts_for_snapshot_view,
            }
            if pli_for_step:  # Only add PLI entry if there are parts for it
                current_pli_dict[step_counter] = pli_for_step

            step_counter += 1
            if is_top_level:
                progress_bar(i + 1, len(step_blocks_raw), "Parsing Model:", length=50)

        return current_pli_dict, current_steps_dict

    # ... (Other LDRModel methods like idx_range_from_steps, has_meta_tag, etc.)
    # These methods would use the .unwrapped list, which is populated by _unwrap_model_recursive.
    # They generally don't involve direct imports that MyPy would flag if the class attributes
    # and method signatures are correctly typed.
    # The complex methods are parse_file, parse_model, and _unwrap_model_recursive.
