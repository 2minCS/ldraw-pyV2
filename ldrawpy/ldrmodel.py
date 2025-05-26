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
from pathlib import Path  # ADDED
from typing import List, Dict, Tuple, Any, Optional, Union, cast

# Explicit imports from toolbox
from toolbox import Vector, Matrix, Identity, apply_params, progress_bar, safe_vector  # type: ignore

# split_path will be removed as its usage is replaced by pathlib

# Explicit imports from ldrawpy package
from .constants import SPECIAL_TOKENS, LDR_DEF_COLOUR, ASPECT_DICT, FLIP_DICT
from .ldrprimitives import LDRPart
from .ldrhelpers import norm_aspect, preset_aspect

# Conditional import for brickbom
try:
    from brickbom import BOM, BOMPart  # type: ignore
except ImportError:
    BOM = None  # type: ignore
    BOMPart = None  # type: ignore

from rich import print as rich_print  # type: ignore

START_TOKENS = ["PLI BEGIN IGN", "BUFEXCHG STORE"]
END_TOKENS = ["PLI END", "BUFEXCHG RETRIEVE"]
EXCEPTION_LIST = ["2429c01.dat"]
IGNORE_LIST = ["LS02"]
COMMON_SUBSTITUTIONS: List[Tuple[str, str]] = [
    ("3070a", "3070b"),
    ("3069a", "3069b"),
    ("3068a", "3068b"),
    ("x224", "41751"),
    ("4864a", "87552"),
    ("4864b", "87552"),
    ("2362a", "87544"),
    ("2362b", "87544"),
    ("60583", "60583b"),
    ("60583a", "60583b"),
    ("3245a", "3245c"),
    ("3245b", "3245c"),
    ("3794", "15573"),
    ("3794a", "15573"),
    ("3794b", "15573"),
    ("4215a", "60581"),
    ("4215b", "60581"),
    ("4215", "60581"),
    ("73983", "2429c01"),
    ("3665a", "3665"),
    ("3665b", "3665"),
    ("4081a", "4081b"),
    ("4085a", "60897"),
    ("4085b", "60897"),
    ("4085c", "60897"),
    ("6019", "61252"),
    ("59426", "32209"),
    ("48729", "48729b"),
    ("48729a", "48729b"),
    ("41005", "48729b"),
    ("4459", "2780"),
    ("44302", "44302a"),
    ("44302b", "44302a"),
    ("2436", "28802"),
    ("2436a", "28802"),
    ("2436b", "28802"),
    ("2454", "2454b"),
    ("64567", "577b"),
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
    """Applies common part substitutions to an LDRPart object."""
    for e in COMMON_SUBSTITUTIONS:
        if part.name == e[0]:
            part.name = e[1]
    return part


def line_has_all_tokens(line: str, tokenlist: List[str]) -> bool:
    """Checks if a line contains all tokens from any group in a list of token groups."""
    line_tokens = line.split()
    for t_group_str in tokenlist:
        # Check if all required tokens in the current group are present in the line
        if all(req_token in line_tokens for req_token in t_group_str.split()):
            return True
    return False


def parse_special_tokens(line: str) -> List[Dict[str, Any]]:
    """Parses a line for special LDraw meta-commands defined in SPECIAL_TOKENS."""
    ls = line.strip().split()
    metas: List[Dict[str, Any]] = []
    for cmd_key, token_patterns in SPECIAL_TOKENS.items():
        for pattern_str in token_patterns:
            pattern_tokens = pattern_str.split()
            # Extract non-placeholder tokens from the pattern (those not starting with '%')
            non_placeholder_pattern_tokens = [
                pt for pt in pattern_tokens if not pt.startswith("%")
            ]
            # Check if all non-placeholder tokens are in the line
            if not all(nppt in ls for nppt in non_placeholder_pattern_tokens):
                continue

            extracted_values: List[str] = []
            valid_match_for_values = True
            try:
                # Assume the first non-placeholder token is the command itself
                cmd_in_pattern = non_placeholder_pattern_tokens[0]
                cmd_start_idx_in_line = ls.index(cmd_in_pattern)

                for pt_idx, pt_token in enumerate(pattern_tokens):
                    if pt_token.startswith("%") and pt_token[1:].isdigit():
                        placeholder_num_in_pattern = int(pt_token[1:])
                        idx_of_last_cmd_token_in_line = ls.index(
                            non_placeholder_pattern_tokens[-1]
                        )
                        actual_value_idx_in_line = (
                            idx_of_last_cmd_token_in_line + placeholder_num_in_pattern
                        )

                        if actual_value_idx_in_line < len(ls):
                            extracted_values.append(ls[actual_value_idx_in_line])
                        else:
                            valid_match_for_values = False
                            break
            except (ValueError, IndexError):  # If .index fails or out of bounds
                valid_match_for_values = False

            if not valid_match_for_values and any(
                pt.startswith("%") for pt in pattern_tokens
            ):  # If placeholders were expected but not found
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
    """Extracts all recognized meta commands from an LDraw string."""
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
    """
    Extracts type 1 LDraw lines (parts/submodels) from an LDraw string,
    respecting masking contexts like PLI and BUFEXCHG.
    """
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
            if len(split_line_tokens) < 14:
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
    current_offset: Vector = Vector(0, 0, 0),  # type: ignore
    current_matrix: Matrix = Identity(),  # type: ignore
    reset_parts_list_on_call: bool = False,
    filter_for_submodel_name: Optional[str] = None,
):
    """
    Recursively parses a model structure, resolving submodels into their constituent parts.
    Applies transformations down the hierarchy.
    """
    if reset_parts_list_on_call:
        output_parts_list.clear()

    for entry_dict in model_entries:
        part_name, ldr_text = entry_dict["partname"], entry_dict["ldrtext"]

        if filter_for_submodel_name and part_name != filter_for_submodel_name:
            continue

        if part_name in all_submodels_data:
            submodel_definition_entries = all_submodels_data[part_name]
            ref_part_instance = LDRPart()
            if ref_part_instance.from_str(ldr_text) is None:
                continue

            new_matrix = current_matrix * ref_part_instance.attrib.matrix
            new_offset = current_matrix * ref_part_instance.attrib.loc + current_offset  # type: ignore

            recursive_parse_model(
                submodel_definition_entries,
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
    """Returns a dictionary of unique items from a list and their counts."""
    return dict(defaultdict(int, {k: items.count(k) for k in set(items)}))


def key_name(elem: LDRPart) -> str:
    """Sort key for LDRPart based on name."""
    return elem.name


def key_colour(elem: LDRPart) -> int:
    """Sort key for LDRPart based on colour code."""
    return elem.attrib.colour


def get_sha1_hash(parts: List[LDRPart]) -> str:
    """Generates a SHA1 hash from a list of LDRPart objects."""
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
    """Sorts a list of LDRPart objects by a given key and order."""
    if not all(isinstance(p, LDRPart) for p in parts):
        return []
    sp = list(parts)
    is_desc = order.lower() == "descending"
    key_func_map = {
        "sha1": lambda p_item: p_item.sha1hash(),
        "name": key_name,
        "colour": key_colour,
    }
    sort_key_func = key_func_map.get(key.lower())
    if sort_key_func is None:
        return sp
    sp.sort(key=sort_key_func, reverse=is_desc)
    return sp


class LDRModel:
    """
    Represents an LDraw model, capable of parsing LDraw files (including MPD),
    managing steps, submodels, and generating various views or BOMs.
    """

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
    filepath: Path
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

    def __init__(self, filename_str: str, **kwargs: Any):
        self.filepath = Path(filename_str).expanduser().resolve()
        self.title = self.filepath.name

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

        if self.bom and hasattr(self.bom, "ignore_parts"):
            self.bom.ignore_parts = []

    def __str__(self) -> str:
        return (
            f"LDRModel: {self.title}\n"
            f"  File: {str(self.filepath)}\n"
            f"  Global origin: {self.global_origin} Global aspect: {self.global_aspect}\n"
            f"  Number of steps: {len(self.steps)}\n"
            f"  Number of sub-models: {len(self.sub_models)}"
        )

    def __getitem__(self, key: int) -> Dict[str, Any]:
        """Allows accessing unwrapped steps by index like a list."""
        if self.unwrapped is None:
            self.unwrap()

        # Assign to a local variable for MyPy to narrow the type effectively
        current_unwrapped_steps = self.unwrapped
        if current_unwrapped_steps is None:
            raise IndexError("Model not unwrapped or unwrap failed.")
        return current_unwrapped_steps[key]  # Index the local variable

    def print_step_dict(self, key: int):
        """Prints the detailed dictionary for a specific parsed step of the main model."""
        if key in self.steps:
            s_dict = self.steps[key]
            for k, v_item in s_dict.items():
                if k == "sub_parts" and isinstance(v_item, dict):
                    rich_print(f"[bold blue]{k}:[/bold blue]")
                    for ks_sub, vs_list_sub in v_item.items():  # type: ignore
                        rich_print(f"  [cyan]{ks_sub}:[/cyan]")
                        for e_part_sub in vs_list_sub:  # type: ignore
                            rich_print(f"    {str(e_part_sub).rstrip()}")
                elif isinstance(v_item, list) and all(
                    isinstance(item, LDRPart) for item in v_item  # type: ignore
                ):
                    rich_print(f"[bold blue]{k}:[/bold blue] ({len(v_item)} items)")  # type: ignore
                    for vx_part_item in v_item:  # type: ignore
                        rich_print(f"  {str(vx_part_item).rstrip()}")
                elif k == "pli_bom" and BOM is not None and isinstance(v_item, BOM):
                    rich_print(f"[bold blue]{k}:[/bold blue]")
                    if hasattr(v_item, "summary_str"):
                        rich_print(f"  {v_item.summary_str()}")  # type: ignore
                    else:
                        rich_print(f"  {v_item}")
                else:
                    rich_print(f"[bold blue]{k}:[/bold blue] {v_item}")
        else:
            rich_print(f"Step {key} not found in main model steps.")

    def print_unwrapped_dict(self, idx: int):
        """Prints the detailed dictionary for a specific step from the unwrapped model list."""
        if self.unwrapped is None or not (0 <= idx < len(self.unwrapped)):
            rich_print(f"Index {idx} out of bounds for unwrapped model.")
            return
        s_dict = self.unwrapped[idx]
        for k, v_item in s_dict.items():
            if (
                k in ("parts", "step_parts")
                and isinstance(v_item, list)
                and all(isinstance(item, LDRPart) for item in v_item)  # type: ignore
            ):
                rich_print(f"[bold green]{k}:[/bold green] ({len(v_item)} parts)")  # type: ignore
                for vx_part_item in v_item:  # type: ignore
                    rich_print(f"  {str(vx_part_item).rstrip()}")
            elif k == "pli_bom" and BOM is not None and isinstance(v_item, BOM):
                rich_print(f"[bold green]{k}:[/bold green]")
                if hasattr(v_item, "summary_str"):
                    rich_print(f"  {v_item.summary_str()}")  # type: ignore
                else:
                    rich_print(f"  {v_item}")
            else:
                rich_print(f"[bold green]{k}:[/bold green] {v_item}")

    def print_unwrapped_verbose(self):
        """Prints a verbose summary of each step in the unwrapped model."""
        if not self.unwrapped:
            rich_print("Model not unwrapped.")
            return
        for i, v_step_item in enumerate(self.unwrapped):
            aspect = v_step_item.get("aspect", (0.0, 0.0, 0.0))
            rich_print(
                f"{i:3d}. idx:{v_step_item.get('idx','N/A'):3} "
                f"[pl:{v_step_item.get('prev_level', 'N/A')} l:{v_step_item.get('level','N/A')} nl:{v_step_item.get('next_level', 'N/A')}] "
                f"[s:{v_step_item.get('step','N/A'):2} ns:{v_step_item.get('next_step','N/A'):2} sc:{v_step_item.get('num_steps','N/A'):2}] "
                f"{str(v_step_item.get('model','N/A'))[:16]:<16} q:{v_step_item.get('qty',0):d} sc:{v_step_item.get('scale',0.0):.2f} "
                f"asp:({aspect[0]:3.0f},{aspect[1]:4.0f},{aspect[2]:3.0f})"
            )

    def print_unwrapped(self):
        """Prints a formatted summary of each step in the unwrapped model."""
        if not self.unwrapped:
            rich_print("Model not unwrapped.")
            return
        for v_step_item in self.unwrapped:
            self.print_step(v_step_item)

    def print_step(self, v_step_dict: dict):
        """Helper to print a single step from the unwrapped model list, formatted."""
        pb = "break" if v_step_dict.get("page_break") else ""
        co = str(v_step_dict.get("callout", 0))
        model_name = str(v_step_dict.get("model", "")).replace(".ldr", "")[:16]
        qty = (
            f"({v_step_dict.get('qty',0):2d}x)"
            if v_step_dict.get("qty", 0) > 0
            else "     "
        )
        level_str = (
            " " * v_step_dict.get("level", 0) + f"Level {v_step_dict.get('level',0)}"
        )
        level_str_padded = f"{level_str:<11}"

        pli_bom_obj = v_step_dict.get("pli_bom")
        parts_count = 0
        if BOM and isinstance(pli_bom_obj, BOM) and hasattr(pli_bom_obj, "parts"):
            parts_count = len(pli_bom_obj.parts)  # type: ignore
        elif isinstance(pli_bom_obj, list):
            parts_count = len(pli_bom_obj)
        parts_str = f"({parts_count:2d}x pcs)"

        meta_tags = []
        for m_item in v_step_dict.get("meta", []):
            if isinstance(m_item, dict):
                for k_meta, v_meta_dict in m_item.items():
                    tag_str = k_meta.replace("_", " ")
                    if k_meta == "columns" and "values" in v_meta_dict:  # type: ignore
                        meta_tags.append(f"[green]COL{v_meta_dict['values'][0]}[/]")  # type: ignore
                    else:
                        meta_tags.append(f"[dim]{tag_str}[/dim]")
            else:
                meta_tags.append(str(m_item))
        meta_str = " ".join(meta_tags)
        aspect = v_step_dict.get("aspect", (0.0, 0.0, 0.0))

        fmt_base = (
            f"{v_step_dict.get('idx','N/A'):3}. {level_str_padded} Step "
            f"{'[yellow]' if co != '0' else '[green]'}{v_step_dict.get('step','N/A'):3}/{v_step_dict.get('num_steps','N/A'):3}{'[/]'} "
            f"Model: {'[red]' if co != '0' and model_name != 'root' else '[green]'}{model_name:<16}{'[/]'} "
            f"{qty} {parts_str} scale: {v_step_dict.get('scale',0.0):.2f} "
            f"({aspect[0]:3.0f},{aspect[1]:4.0f},{aspect[2]:3.0f})"
        )
        fmt_co = f" {'[yellow]' if co != '0' else '[dim]'}{co}{'[/]'}"
        fmt_pb = f" {'[magenta]BR[/]' if pb else ''}"
        rich_print(f"{fmt_base}{fmt_co}{fmt_pb} {meta_str}")

    def transform_parts_to(
        self,
        parts: List[LDRPart],
        origin: Optional[Union[Tuple[float, float, float], Vector]] = None,  # type: ignore
        aspect: Optional[Union[Tuple[float, float, float], Vector]] = None,  # type: ignore
        use_exceptions: bool = False,
    ) -> List[LDRPart]:
        """Transforms a list of parts to a specific origin and aspect (for PLI)."""
        final_aspect_tuple = aspect if aspect is not None else self.pli_aspect
        final_origin_vec = safe_vector(origin) if origin is not None else None

        transformed_parts = []
        for p_part in parts:
            np = p_part.copy()
            current_aspect_for_part = list(final_aspect_tuple)
            if use_exceptions and p_part.name in self.pli_exceptions:
                current_aspect_for_part = list(self.pli_exceptions[p_part.name])
            np.set_rotation(tuple(current_aspect_for_part))
            if final_origin_vec:
                np.move_to(final_origin_vec)
            transformed_parts.append(np)
        return transformed_parts

    def transform_parts(
        self,
        parts: Union[List[LDRPart], List[str]],
        offset: Optional[Union[Tuple[float, float, float], Vector]] = None,  # type: ignore
        aspect: Optional[Union[Tuple[float, float, float], Vector]] = None,  # type: ignore
    ) -> Union[List[LDRPart], List[str]]:
        """Transforms a list of parts (LDRPart objects or LDraw strings) by an offset and aspect."""
        final_aspect_tuple = aspect if aspect is not None else self.global_aspect
        final_offset_vec = (
            safe_vector(offset) if offset is not None else Vector(0, 0, 0)  # type: ignore
        )

        if not parts:
            return []

        if all(isinstance(p_item, LDRPart) for p_item in parts):
            processed_ldr_parts: List[LDRPart] = []
            for p_obj_item in cast(List[LDRPart], parts):
                np = p_obj_item.copy()
                np.rotate_by(final_aspect_tuple)
                np.move_by(final_offset_vec)
                processed_ldr_parts.append(np)
            return processed_ldr_parts
        elif all(isinstance(p_item, str) for p_item in parts):
            processed_str_parts: List[str] = []
            for p_str_item in cast(List[str], parts):
                np = LDRPart()
                if np.from_str(p_str_item):
                    np.rotate_by(final_aspect_tuple)
                    np.move_by(final_offset_vec)
                    processed_str_parts.append(str(np))
                else:
                    processed_str_parts.append(p_str_item)
            return processed_str_parts
        return []

    def parse_file(self):
        """
        Parses the LDraw model file specified during initialization.
        Handles MPD (Multi-Part Document) files by identifying the root model
        and submodels. Populates self.steps, self.pli, self.sub_models, etc.
        """
        self.sub_models = {}
        self.sub_model_str = {}
        try:
            with open(self.filepath, "rt", encoding="utf-8") as fp:
                content = fp.read()
        except FileNotFoundError:
            rich_print(f"Error: File {str(self.filepath)} not found.")
            self.pli, self.steps = {}, {}
            return
        except Exception as e:
            rich_print(f"Error reading file {str(self.filepath)}: {e}")
            self.pli, self.steps = {}, {}
            return

        file_blocks = content.split("0 FILE")
        root_model_content = ""
        sub_file_blocks = []

        if not content.strip().startswith("0 FILE") and file_blocks:
            root_model_content = file_blocks[0]
            sub_file_blocks = file_blocks[1:]
        elif len(file_blocks) > 1 and content.strip().startswith("0 FILE"):
            root_model_content = "0 FILE " + file_blocks[1].strip()
            sub_file_blocks = file_blocks[2:]
        else:
            root_model_content = content
            sub_file_blocks = []

        for sub_block_text in sub_file_blocks:
            if not sub_block_text.strip():
                continue
            full_sub_model_definition_str = "0 FILE " + sub_block_text.strip()
            lines_in_submodel = full_sub_model_definition_str.splitlines()
            if not lines_in_submodel:
                continue

            sub_name_key = lines_in_submodel[0].replace("0 FILE", "").strip().lower()
            if sub_name_key:
                self.sub_model_str[sub_name_key] = full_sub_model_definition_str
                self.sub_models[sub_name_key] = get_parts_from_model(
                    full_sub_model_definition_str
                )

        if root_model_content.strip():
            self.pli, self.steps = self.parse_model_steps(
                root_model_content, is_top_level=True
            )
        else:
            rich_print(
                f"Warning: Root model content for {str(self.filepath)} is empty after processing '0 FILE' directives."
            )
            self.pli, self.steps = {}, {}

        self.unwrap()

    def unwrap(self):
        """
        Flattens the hierarchical model structure (main model and its submodels)
        into a single list of steps (self.unwrapped). This involves recursively
        processing each step and its submodels.
        """
        if self.unwrapped is not None:
            return

        self._parsed_submodel_steps_cache = {}
        for sub_name, sub_model_raw_str_content in self.sub_model_str.items():
            _, steps_dict_for_submodel = self.parse_model_steps(
                sub_model_raw_str_content, is_top_level=False
            )
            self._parsed_submodel_steps_cache[sub_name] = steps_dict_for_submodel

        self.unwrapped = self._unwrap_model_recursive(
            current_model_steps_dict=self.steps
        )

    def _unwrap_model_recursive(
        self,
        current_model_steps_dict: Dict[int, Dict[str, Any]],
        current_idx_offset: int = 0,
        current_level: int = 0,
        model_name_for_these_steps: str = "root",
        model_qty_for_this_instance: int = 1,
        unwrapped_list_accumulator: Optional[List[Dict[str, Any]]] = None,
    ) -> Union[List[Dict[str, Any]], Tuple[List[Dict[str, Any]], int]]:
        """
        Recursive helper for unwrap. Processes steps of current_model_steps_dict.
        If a step contains submodels, calls itself for each submodel.
        Returns the populated list and the next available global index.
        """
        is_top_level_call = unwrapped_list_accumulator is None
        if is_top_level_call:
            unwrapped_list_accumulator = []

        next_global_idx = current_idx_offset

        sorted_step_numbers = sorted(current_model_steps_dict.keys())

        for step_no_in_current_model in sorted_step_numbers:
            step_data_from_parent = current_model_steps_dict[step_no_in_current_model]

            if step_data_from_parent.get("sub_models"):
                unique_submodel_refs_in_this_step = unique_set(
                    step_data_from_parent["sub_models"]
                )
                for (
                    sub_name_ref,
                    qty_of_this_sub_ref,
                ) in unique_submodel_refs_in_this_step.items():
                    submodel_definition_steps = self._parsed_submodel_steps_cache.get(
                        sub_name_ref
                    )
                    if not submodel_definition_steps:
                        continue

                    recursive_result = self._unwrap_model_recursive(
                        submodel_definition_steps,
                        next_global_idx,
                        current_level + 1,
                        sub_name_ref,
                        qty_of_this_sub_ref,
                        unwrapped_list_accumulator,
                    )
                    if (
                        isinstance(recursive_result, tuple)
                        and len(recursive_result) == 2
                    ):
                        next_global_idx = recursive_result[1]

            entry_for_unwrapped_list = {
                "idx": next_global_idx,
                "level": current_level,
                "step": step_no_in_current_model,
                "next_step": (
                    step_no_in_current_model + 1
                    if step_no_in_current_model < len(sorted_step_numbers)
                    else step_no_in_current_model
                ),
                "num_steps": len(sorted_step_numbers),
                "model": model_name_for_these_steps,
                "qty": model_qty_for_this_instance,
                "scale": step_data_from_parent.get("scale", self.global_scale),
                "model_scale": step_data_from_parent.get(
                    "model_scale", self.global_scale
                ),
                "aspect": step_data_from_parent.get("aspect", self.global_aspect),
                "parts": step_data_from_parent.get("parts", []),
                "step_parts": step_data_from_parent.get("step_parts", []),
                "pli_bom": step_data_from_parent.get("pli_bom", BOM() if BOM else []),
                "meta": step_data_from_parent.get("meta", []),
                "aspect_change": step_data_from_parent.get("aspect_change", False),
                "raw_ldraw": step_data_from_parent.get("raw_ldraw", ""),
                "sub_parts": step_data_from_parent.get("sub_parts", {}),
            }
            unwrapped_list_accumulator.append(entry_for_unwrapped_list)  # type: ignore
            next_global_idx += 1

        if is_top_level_call:
            final_processed_list: List[Dict[str, Any]] = []
            continuous_step_counter_for_main_flow = 1
            self.callouts = {}
            active_callout_start_indices: Dict[int, int] = {}

            for i, entry_item_dict in enumerate(unwrapped_list_accumulator):  # type: ignore
                entry_item_dict["prev_level"] = unwrapped_list_accumulator[i - 1]["level"] if i > 0 else 0  # type: ignore
                entry_item_dict["next_level"] = (
                    unwrapped_list_accumulator[i + 1]["level"]  # type: ignore
                    if i < len(unwrapped_list_accumulator) - 1  # type: ignore
                    else entry_item_dict["level"]
                )

                has_page_break_meta_cmd = any(
                    "page_break" in m_cmd for m_cmd in entry_item_dict.get("meta", [])
                )
                is_no_callout_meta_cmd = any(
                    "no_callout" in m_cmd for m_cmd in entry_item_dict.get("meta", [])
                )

                entry_item_dict["page_break"] = has_page_break_meta_cmd or (
                    entry_item_dict["next_level"] > entry_item_dict["level"]
                    and entry_item_dict.get("num_steps", 0) >= self.callout_step_thr
                    and not is_no_callout_meta_cmd
                )

                if (
                    entry_item_dict["level"] > entry_item_dict["prev_level"]
                    and not is_no_callout_meta_cmd
                    and entry_item_dict.get("num_steps", 0) < self.callout_step_thr
                ):
                    active_callout_start_indices[entry_item_dict["level"]] = (
                        entry_item_dict["idx"]
                    )

                current_callout_level_for_this_step = 0
                for active_lvl_cs_key in sorted(
                    active_callout_start_indices.keys(), reverse=True
                ):
                    if entry_item_dict["level"] >= active_lvl_cs_key:
                        current_callout_level_for_this_step = active_lvl_cs_key
                        break
                entry_item_dict["callout"] = current_callout_level_for_this_step

                if entry_item_dict["level"] < entry_item_dict["prev_level"]:
                    if entry_item_dict["prev_level"] in active_callout_start_indices:
                        start_idx_of_completed_callout = (
                            active_callout_start_indices.pop(
                                entry_item_dict["prev_level"]
                            )
                        )
                        self.callouts[start_idx_of_completed_callout] = {
                            "level": entry_item_dict["prev_level"],
                            "end": unwrapped_list_accumulator[i - 1]["idx"],  # type: ignore
                            "parent": unwrapped_list_accumulator[start_idx_of_completed_callout]["prev_level"],  # type: ignore
                        }
                        scale_meta_at_callout_start = next(
                            (m_cmd for m_cmd in unwrapped_list_accumulator[start_idx_of_completed_callout].get("meta", []) if "model_scale" in m_cmd),  # type: ignore
                            None,
                        )
                        if scale_meta_at_callout_start:
                            self.callouts[start_idx_of_completed_callout][
                                "scale"
                            ] = float(
                                scale_meta_at_callout_start["model_scale"]["values"][0]  # type: ignore
                            )

                if self.continuous_step_numbers and entry_item_dict["callout"] == 0:
                    entry_item_dict["step"] = continuous_step_counter_for_main_flow
                    continuous_step_counter_for_main_flow += 1

                final_processed_list.append(entry_item_dict)

            if self.continuous_step_numbers:
                self.continuous_step_count = continuous_step_counter_for_main_flow - 1
                for e_fm_item_update in final_processed_list:
                    if e_fm_item_update["callout"] == 0:
                        e_fm_item_update["num_steps"] = self.continuous_step_count

            for e_fm_pli_item_update in final_processed_list:
                for meta_item_cmd in e_fm_pli_item_update.get("meta", []):
                    if (
                        "pli_proxy" in meta_item_cmd
                        and BOM
                        and BOMPart
                        and isinstance(e_fm_pli_item_update["pli_bom"], BOM)
                    ):
                        for item_str_val_proxy in meta_item_cmd["pli_proxy"].get("values", []):  # type: ignore
                            pname_val_proxy, pcol_val_str_proxy = (
                                item_str_val_proxy.split("_")  # type: ignore
                                if "_" in item_str_val_proxy  # type: ignore
                                else (item_str_val_proxy, str(LDR_DEF_COLOUR))
                            )
                            pcol_val_proxy = int(pcol_val_str_proxy)

                            bom_part_instance_proxy = BOMPart(1, pname_val_proxy, pcol_val_proxy)  # type: ignore
                            e_fm_pli_item_update["pli_bom"].add_part(bom_part_instance_proxy)  # type: ignore
                            if self.bom:
                                self.bom.add_part(bom_part_instance_proxy)
            return final_processed_list  # type: ignore

        return unwrapped_list_accumulator, next_global_idx  # type: ignore

    def parse_model_steps(
        self,
        model_source_str_content: str,
        is_top_level: bool = True,
    ) -> Tuple[Dict[int, List[LDRPart]], Dict[int, Dict[str, Any]]]:
        """
        Parses the LDraw string content of a single model (or submodel) into steps.
        """
        if not model_source_str_content.strip():
            return {}, {}

        pli_dict_for_this_model: Dict[int, List[LDRPart]] = {}
        steps_dict_for_this_model: Dict[int, Dict[str, Any]] = {}

        step_blocks_raw = model_source_str_content.split("0 STEP")

        cumulative_parts_for_snapshot: List[LDRPart] = []

        current_aspect_list_for_model: List[float] = list(
            self.global_aspect if is_top_level else self.PARAMS["global_aspect"]
        )
        current_scale_for_model_view = (
            self.global_scale if is_top_level else self.PARAMS["global_scale"]
        )
        model_inherent_scale_preference = current_scale_for_model_view

        current_step_number = 1

        for i, single_step_raw_content in enumerate(step_blocks_raw):
            if (
                i == 0
                and not single_step_raw_content.strip()
                and len(step_blocks_raw) > 1
            ):
                continue

            meta_commands_in_step = get_meta_commands(single_step_raw_content)
            aspect_was_changed_by_meta_in_step = False

            for cmd_dict_item in meta_commands_in_step:
                if "scale" in cmd_dict_item:
                    current_scale_for_model_view = float(
                        cmd_dict_item["scale"]["values"][0]
                    )
                elif "model_scale" in cmd_dict_item:
                    model_inherent_scale_preference = float(
                        cmd_dict_item["model_scale"]["values"][0]
                    )
                elif "rotation_abs" in cmd_dict_item:
                    v_abs_rot = [
                        float(x_val)
                        for x_val in cmd_dict_item["rotation_abs"]["values"]
                    ]
                    current_aspect_list_for_model = [
                        -v_abs_rot[0],
                        v_abs_rot[1],
                        v_abs_rot[2],
                    ]
                    aspect_was_changed_by_meta_in_step = True
                elif "rotation_rel" in cmd_dict_item:
                    v_rel_rot_vals = tuple(
                        float(x_val)
                        for x_val in cmd_dict_item["rotation_rel"]["values"]
                    )
                    current_aspect_list_for_model[0] -= v_rel_rot_vals[0]
                    current_aspect_list_for_model[1] += v_rel_rot_vals[1]
                    current_aspect_list_for_model[2] += v_rel_rot_vals[2]
                    current_aspect_list_for_model = list(
                        norm_aspect(
                            cast(
                                Tuple[float, float, float],
                                tuple(current_aspect_list_for_model),
                            )
                        )
                    )
                    aspect_was_changed_by_meta_in_step = True
                elif "rotation_pre" in cmd_dict_item:
                    current_aspect_list_for_model = list(
                        preset_aspect(
                            cast(
                                Tuple[float, float, float],
                                tuple(current_aspect_list_for_model),
                            ),
                            cmd_dict_item["rotation_pre"]["values"],  # type: ignore
                        )
                    )
                    aspect_was_changed_by_meta_in_step = True

            part_reference_dicts_in_step = get_parts_from_model(single_step_raw_content)

            ldr_parts_added_in_this_step: List[LDRPart] = []
            recursive_parse_model(
                part_reference_dicts_in_step,
                self.sub_models,
                ldr_parts_added_in_this_step,
                reset_parts_list_on_call=True,
            )

            if (
                not ldr_parts_added_in_this_step
                and not meta_commands_in_step
                and (i > 0 or not single_step_raw_content.strip())
            ):
                continue

            pli_parts_for_this_step = self.transform_parts_to(
                ldr_parts_added_in_this_step,
                origin=(0, 0, 0),
                aspect=self.pli_aspect,
                use_exceptions=True,
            )

            cumulative_parts_for_snapshot.extend(ldr_parts_added_in_this_step)

            final_current_aspect_tuple_for_model = cast(
                Tuple[float, float, float], tuple(current_aspect_list_for_model)
            )

            snapshot_model_parts_transformed_union = self.transform_parts(
                cumulative_parts_for_snapshot,
                aspect=final_current_aspect_tuple_for_model,
            )
            snapshot_model_parts_transformed: List[LDRPart] = []
            if isinstance(snapshot_model_parts_transformed_union, list) and all(
                isinstance(p_item, LDRPart)
                for p_item in snapshot_model_parts_transformed_union
            ):
                snapshot_model_parts_transformed = cast(
                    List[LDRPart], snapshot_model_parts_transformed_union
                )

            snapshot_step_added_parts_transformed_union = self.transform_parts(
                ldr_parts_added_in_this_step,
                aspect=final_current_aspect_tuple_for_model,
            )
            snapshot_step_added_parts_transformed: List[LDRPart] = []
            if isinstance(snapshot_step_added_parts_transformed_union, list) and all(
                isinstance(p_item, LDRPart)
                for p_item in snapshot_step_added_parts_transformed_union
            ):
                snapshot_step_added_parts_transformed = cast(
                    List[LDRPart], snapshot_step_added_parts_transformed_union
                )

            pli_bom_for_this_step: Optional[BOM] = None
            if BOM and BOMPart:
                pli_bom_for_this_step = BOM()
                if self.bom and hasattr(self.bom, "ignore_parts"):
                    pli_bom_for_this_step.ignore_parts = self.bom.ignore_parts
                for p_part_item in pli_parts_for_this_step:
                    pli_bom_for_this_step.add_part(BOMPart(1, p_part_item.name, p_part_item.attrib.colour))  # type: ignore

                if is_top_level and self.bom:
                    for p_part_item in pli_parts_for_this_step:
                        self.bom.add_part(BOMPart(1, p_part_item.name, p_part_item.attrib.colour))  # type: ignore

            submodel_names_referenced_in_step_lines = [
                pd_item["partname"]
                for pd_item in part_reference_dicts_in_step
                if pd_item["partname"] in self.sub_models
            ]

            sub_parts_snapshot_view_dict: Dict[str, List[LDRPart]] = {}
            for sub_name_key_ref in unique_set(submodel_names_referenced_in_step_lines):
                submodel_definition_part_dicts = self.sub_models.get(
                    sub_name_key_ref, []
                )

                temp_list_for_submodel_content_parts: List[LDRPart] = []
                recursive_parse_model(
                    submodel_definition_part_dicts,
                    self.sub_models,
                    temp_list_for_submodel_content_parts,
                    reset_parts_list_on_call=True,
                )
                transformed_sub_content_parts_union = self.transform_parts(
                    temp_list_for_submodel_content_parts,
                    aspect=final_current_aspect_tuple_for_model,
                )
                if isinstance(transformed_sub_content_parts_union, list) and all(
                    isinstance(tsp_item, LDRPart)
                    for tsp_item in transformed_sub_content_parts_union
                ):
                    sub_parts_snapshot_view_dict[sub_name_key_ref] = cast(
                        List[LDRPart], transformed_sub_content_parts_union
                    )

            steps_dict_for_this_model[current_step_number] = {
                "parts": snapshot_model_parts_transformed,
                "step_parts": snapshot_step_added_parts_transformed,
                "sub_models": submodel_names_referenced_in_step_lines,
                "aspect": final_current_aspect_tuple_for_model,
                "scale": current_scale_for_model_view,
                "model_scale": model_inherent_scale_preference,
                "raw_ldraw": single_step_raw_content,
                "pli_bom": (
                    pli_bom_for_this_step
                    if pli_bom_for_this_step
                    else (BOM() if BOM else [])
                ),
                "meta": meta_commands_in_step,
                "aspect_change": aspect_was_changed_by_meta_in_step,
                "sub_parts": sub_parts_snapshot_view_dict,
            }

            if pli_parts_for_this_step:
                pli_dict_for_this_model[current_step_number] = pli_parts_for_this_step

            current_step_number += 1

            if is_top_level:
                progress_bar(
                    i + 1, len(step_blocks_raw), "Parsing Model Steps:", length=50
                )

        return pli_dict_for_this_model, steps_dict_for_this_model
