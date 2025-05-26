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
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union, cast, TypedDict

# Explicit imports from toolbox
from toolbox import Vector, Matrix, Identity, apply_params, progress_bar, safe_vector  # type: ignore

# Explicit imports from ldrawpy package
from .constants import SPECIAL_TOKENS, LDR_DEF_COLOUR, ASPECT_DICT, FLIP_DICT
from .ldrprimitives import LDRPart
from .ldrhelpers import norm_aspect, preset_aspect

# Conditional import for brickbom
try:
    from brickbom import BOM, BOMPart  # type: ignore

    BOM_AVAILABLE = True
except ImportError:
    BOM = object  # type: ignore
    BOMPart = object  # type: ignore
    BOM_AVAILABLE = False

from rich import print as rich_print  # type: ignore

# --- Constants for parsing ---
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


# --- TypedDict Definitions ---
class MetaCommandData(TypedDict, total=False):
    values: List[str]
    text: str


ParsedMetaCommandItem = Dict[str, MetaCommandData]


class ModelPartEntry(TypedDict):
    ldrtext: str
    partname: str


class CalloutData(TypedDict, total=False):
    level: int
    end: int
    parent: int
    scale: float


PLIBomType = Union[BOM, List[Any]]


class StepData(TypedDict):
    step: int  # ADDED: Original step number in its parent model
    parts: List[LDRPart]
    step_parts: List[LDRPart]
    sub_models: List[str]
    aspect: Tuple[float, float, float]
    scale: float
    model_scale: float
    raw_ldraw: str
    pli_bom: PLIBomType
    meta: List[ParsedMetaCommandItem]
    aspect_change: bool
    sub_parts: Dict[str, List[LDRPart]]


class UnwrappedStepEntry(StepData):
    idx: int
    level: int
    # 'step' is inherited from StepData
    next_step: int
    num_steps: int
    model: str
    qty: int
    prev_level: int
    next_level: int
    page_break: bool
    callout: int


# --- Helper Functions ---
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


def parse_special_tokens(line: str) -> List[ParsedMetaCommandItem]:
    ls = line.strip().split()
    metas: List[ParsedMetaCommandItem] = []
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
                # cmd_start_idx_in_line = ls.index(cmd_in_pattern) # Not directly used with new logic

                idx_of_last_cmd_keyword_in_pattern = -1
                for idx, token_in_pattern in enumerate(pattern_tokens):
                    if not token_in_pattern.startswith("%"):
                        idx_of_last_cmd_keyword_in_pattern = idx

                # Find the index of the last command keyword of the pattern within the line
                # This assumes the command keywords appear contiguously in the line
                idx_of_first_cmd_keyword_in_line = ls.index(
                    non_placeholder_pattern_tokens[0]
                )
                idx_of_last_cmd_keyword_in_line = idx_of_first_cmd_keyword_in_line + (
                    len(non_placeholder_pattern_tokens) - 1
                )

                for pt_token in pattern_tokens:
                    if pt_token.startswith("%") and pt_token[1:].isdigit():
                        placeholder_num_in_pattern = int(pt_token[1:])
                        actual_value_idx_in_line = (
                            idx_of_last_cmd_keyword_in_line + placeholder_num_in_pattern
                        )

                        if actual_value_idx_in_line < len(ls):
                            extracted_values.append(ls[actual_value_idx_in_line])
                        else:
                            valid_match_for_values = False
                            break
            except (ValueError, IndexError):
                valid_match_for_values = False

            if not valid_match_for_values and any(
                pt.startswith("%") for pt in pattern_tokens
            ):
                continue

            current_meta_data: MetaCommandData = {"text": line.strip()}
            if extracted_values:
                current_meta_data["values"] = extracted_values

            # MyPy might struggle here if it can't infer that cmd_key is a valid key for ParsedMetaCommandItem
            # This usually happens if ParsedMetaCommandItem was defined as a Union of specific TypedDicts.
            # Since it's Dict[str, MetaCommandData], this should be okay.
            metas.append({cmd_key: current_meta_data})
            break
        if metas and metas[-1].get(cmd_key):
            break
    return metas


def get_meta_commands(ldr_string: str) -> List[ParsedMetaCommandItem]:
    cmd: List[ParsedMetaCommandItem] = []
    for line in ldr_string.splitlines():
        stripped_line = line.lstrip()
        if not stripped_line or not stripped_line.startswith("0 "):
            continue
        meta_for_line = parse_special_tokens(line)
        if meta_for_line:
            cmd.extend(meta_for_line)
    return cmd


def get_parts_from_model(ldr_string: str) -> List[ModelPartEntry]:
    parts: List[ModelPartEntry] = []
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
            part_dict: ModelPartEntry = {
                "ldrtext": line,
                "partname": " ".join(split_line_tokens[14:]),
            }
            if mask_depth == 0:
                parts.append(part_dict)
            else:
                if part_dict["partname"] in EXCEPTION_LIST:
                    parts.append(part_dict)
                elif not bufex and part_dict["partname"].endswith(".ldr"):
                    parts.append(part_dict)
    return parts


def recursive_parse_model(
    model_entries: List[ModelPartEntry],
    all_submodels_data: Dict[str, List[ModelPartEntry]],
    output_parts_list: List[LDRPart],
    current_offset: Vector = Vector(0, 0, 0),  # type: ignore
    current_matrix: Matrix = Identity(),  # type: ignore
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
    steps: Dict[int, StepData]
    pli: Dict[int, List[LDRPart]]
    sub_models: Dict[str, List[ModelPartEntry]]
    sub_model_str: Dict[str, str]
    unwrapped: Optional[List[UnwrappedStepEntry]]
    callouts: Dict[int, CalloutData]
    continuous_step_count: int
    _parsed_submodel_steps_cache: Dict[str, Dict[int, StepData]]

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
        self.bom = BOM() if BOM_AVAILABLE else None
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

    def __getitem__(self, key: int) -> UnwrappedStepEntry:
        if self.unwrapped is None:
            self.unwrap()
        current_unwrapped_steps = self.unwrapped
        if current_unwrapped_steps is None:
            raise IndexError("Model not unwrapped or unwrap failed.")
        return current_unwrapped_steps[key]

    def print_step_dict(self, key: int):
        if key in self.steps:
            s_dict = self.steps[key]  # s_dict is StepData
            for k, v_item in s_dict.items():
                # k is a key of StepData, v_item is its value
                if k == "sub_parts" and isinstance(v_item, dict):
                    rich_print(f"[bold blue]{k}:[/bold blue]")
                    for ks_sub, vs_list_sub in v_item.items():
                        rich_print(f"  [cyan]{ks_sub}:[/cyan]")
                        for e_part_sub in vs_list_sub:
                            rich_print(f"    {str(e_part_sub).rstrip()}")
                elif (
                    (k == "parts" or k == "step_parts")
                    and isinstance(v_item, list)
                    and all(isinstance(item, LDRPart) for item in v_item)
                ):
                    rich_print(f"[bold blue]{k}:[/bold blue] ({len(v_item)} items)")
                    for vx_part_item in v_item:
                        rich_print(f"  {str(vx_part_item).rstrip()}")
                elif k == "pli_bom" and BOM_AVAILABLE and isinstance(v_item, BOM):
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
        if self.unwrapped is None or not (0 <= idx < len(self.unwrapped)):
            rich_print(f"Index {idx} out of bounds for unwrapped model.")
            return
        s_dict = self.unwrapped[idx]  # s_dict is UnwrappedStepEntry
        for k, v_item in s_dict.items():
            if (
                (k == "parts" or k == "step_parts")
                and isinstance(v_item, list)
                and all(isinstance(item, LDRPart) for item in v_item)
            ):
                rich_print(f"[bold green]{k}:[/bold green] ({len(v_item)} parts)")
                for vx_part_item in v_item:
                    rich_print(f"  {str(vx_part_item).rstrip()}")
            elif k == "pli_bom" and BOM_AVAILABLE and isinstance(v_item, BOM):
                rich_print(f"[bold green]{k}:[/bold green]")
                if hasattr(v_item, "summary_str"):
                    rich_print(f"  {v_item.summary_str()}")  # type: ignore
                else:
                    rich_print(f"  {v_item}")
            else:
                rich_print(f"[bold green]{k}:[/bold green] {v_item}")

    def print_unwrapped_verbose(self):
        if not self.unwrapped:
            rich_print("Model not unwrapped.")
            return
        for i, v_step_item in enumerate(
            self.unwrapped
        ):  # v_step_item is UnwrappedStepEntry
            aspect = v_step_item[
                "aspect"
            ]  # Direct access, key is required in UnwrappedStepEntry (via StepData)
            rich_print(
                f"{i:3d}. idx:{v_step_item['idx']:3} "
                f"[pl:{v_step_item['prev_level']} l:{v_step_item['level']} nl:{v_step_item['next_level']}] "
                f"[s:{v_step_item['step']:2} ns:{v_step_item['next_step']:2} sc:{v_step_item['num_steps']:2}] "
                f"{str(v_step_item['model'])[:16]:<16} q:{v_step_item['qty']:d} sc:{v_step_item['scale']:.2f} "
                f"asp:({aspect[0]:3.0f},{aspect[1]:4.0f},{aspect[2]:3.0f})"
            )

    def print_unwrapped(self):
        if not self.unwrapped:
            rich_print("Model not unwrapped.")
            return
        for v_step_item in self.unwrapped:  # v_step_item is UnwrappedStepEntry
            self.print_step(v_step_item)

    def print_step(self, v_step_dict: UnwrappedStepEntry):
        pb = "break" if v_step_dict["page_break"] else ""
        co = str(v_step_dict["callout"])
        model_name = str(v_step_dict["model"]).replace(".ldr", "")[:16]
        qty = f"({v_step_dict['qty']:2d}x)" if v_step_dict["qty"] > 0 else "     "
        level_str = " " * v_step_dict["level"] + f"Level {v_step_dict['level']}"
        level_str_padded = f"{level_str:<11}"
        pli_bom_obj = v_step_dict["pli_bom"]
        parts_count = 0
        if (
            BOM_AVAILABLE
            and isinstance(pli_bom_obj, BOM)
            and hasattr(pli_bom_obj, "parts")
        ):
            parts_count = len(pli_bom_obj.parts)  # type: ignore
        elif isinstance(pli_bom_obj, list):
            parts_count = len(pli_bom_obj)
        parts_str = f"({parts_count:2d}x pcs)"
        meta_tags = []
        for m_item_dict in v_step_dict["meta"]:
            for k_meta, v_meta_data in m_item_dict.items():
                tag_str = k_meta.replace("_", " ")
                if k_meta == "columns" and "values" in v_meta_data:
                    meta_tags.append(f"[green]COL{v_meta_data['values'][0]}[/]")
                else:
                    meta_tags.append(f"[dim]{tag_str}[/dim]")
        meta_str = " ".join(meta_tags)
        aspect = v_step_dict["aspect"]
        fmt_base = (
            f"{v_step_dict['idx']:3}. {level_str_padded} Step "
            f"{'[yellow]' if co != '0' else '[green]'}{v_step_dict['step']:3}/{v_step_dict['num_steps']:3}{'[/]'} "
            f"Model: {'[red]' if co != '0' and model_name != 'root' else '[green]'}{model_name:<16}{'[/]'} "
            f"{qty} {parts_str} scale: {v_step_dict['scale']:.2f} "
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
        if self.unwrapped is not None:
            return
        self._parsed_submodel_steps_cache = {}
        for sub_name, sub_model_raw_str_content in self.sub_model_str.items():
            _, steps_dict_for_submodel = self.parse_model_steps(
                sub_model_raw_str_content, is_top_level=False
            )
            self._parsed_submodel_steps_cache[sub_name] = steps_dict_for_submodel

        # Initialize unwrapped to an empty list before starting the recursive process
        # This ensures that _unwrap_model_recursive receives a list, not None, for its accumulator.
        initial_unwrapped_list: List[UnwrappedStepEntry] = []
        self.unwrapped = self._unwrap_model_recursive(
            current_model_steps_dict=self.steps,
            unwrapped_list_accumulator=initial_unwrapped_list,  # Pass the initialized list
        )

    def _unwrap_model_recursive(
        self,
        current_model_steps_dict: Dict[int, StepData],
        current_idx_offset: int = 0,
        current_level: int = 0,
        model_name_for_these_steps: str = "root",
        model_qty_for_this_instance: int = 1,
        unwrapped_list_accumulator: List[
            UnwrappedStepEntry
        ] = [],  # Now defaults to empty list
    ) -> Union[List[UnwrappedStepEntry], Tuple[List[UnwrappedStepEntry], int]]:

        is_top_level_call = (
            current_level == 0
            and current_idx_offset == 0
            and model_name_for_these_steps == "root"
        )
        # If it's the top-level call and unwrapped_list_accumulator was passed as the default [], it's fine.
        # For recursive calls, it will be passed the accumulating list.

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

                    # Pass the same unwrapped_list_accumulator for recursive calls
                    recursive_result = self._unwrap_model_recursive(
                        submodel_definition_steps,
                        next_global_idx,
                        current_level + 1,
                        sub_name_ref,
                        qty_of_this_sub_ref,
                        unwrapped_list_accumulator,
                    )
                    # The recursive call appends to the list passed; we only need the updated index
                    if (
                        isinstance(recursive_result, tuple)
                        and len(recursive_result) == 2
                    ):
                        next_global_idx = recursive_result[1]

            entry_for_unwrapped_list: UnwrappedStepEntry = {
                "idx": next_global_idx,
                "level": current_level,
                "step": step_no_in_current_model,  # This is now correctly expected via StepData
                "next_step": (
                    step_no_in_current_model + 1
                    if step_no_in_current_model < len(sorted_step_numbers)
                    else step_no_in_current_model
                ),
                "num_steps": len(sorted_step_numbers),
                "model": model_name_for_these_steps,
                "qty": model_qty_for_this_instance,
                "scale": step_data_from_parent["scale"],
                "model_scale": step_data_from_parent["model_scale"],
                "aspect": step_data_from_parent["aspect"],
                "parts": step_data_from_parent["parts"],
                "step_parts": step_data_from_parent["step_parts"],
                "sub_models": step_data_from_parent[
                    "sub_models"
                ],  # ADDED: Ensure this is populated
                "pli_bom": step_data_from_parent["pli_bom"],
                "meta": step_data_from_parent["meta"],
                "aspect_change": step_data_from_parent["aspect_change"],
                "raw_ldraw": step_data_from_parent["raw_ldraw"],
                "sub_parts": step_data_from_parent["sub_parts"],
                "prev_level": 0,
                "next_level": 0,
                "page_break": False,
                "callout": 0,
            }
            unwrapped_list_accumulator.append(entry_for_unwrapped_list)
            next_global_idx += 1

        if is_top_level_call:
            # Post-processing loop now operates on unwrapped_list_accumulator which is a List
            final_processed_list: List[UnwrappedStepEntry] = (
                unwrapped_list_accumulator  # Alias for clarity
            )
            continuous_step_counter_for_main_flow = 1
            self.callouts = {}
            active_callout_start_indices: Dict[int, int] = {}

            for i, entry_item_dict in enumerate(final_processed_list):
                entry_item_dict["prev_level"] = (
                    final_processed_list[i - 1]["level"] if i > 0 else 0
                )
                entry_item_dict["next_level"] = (
                    final_processed_list[i + 1]["level"]
                    if i < len(final_processed_list) - 1
                    else entry_item_dict["level"]
                )
                has_page_break_meta_cmd = any(
                    "page_break" in m_cmd for m_cmd in entry_item_dict["meta"]
                )
                is_no_callout_meta_cmd = any(
                    "no_callout" in m_cmd for m_cmd in entry_item_dict["meta"]
                )
                entry_item_dict["page_break"] = has_page_break_meta_cmd or (
                    entry_item_dict["next_level"] > entry_item_dict["level"]
                    and entry_item_dict["num_steps"] >= self.callout_step_thr
                    and not is_no_callout_meta_cmd
                )
                if (
                    entry_item_dict["level"] > entry_item_dict["prev_level"]
                    and not is_no_callout_meta_cmd
                    and entry_item_dict["num_steps"] < self.callout_step_thr
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
                        callout_entry: CalloutData = {
                            "level": entry_item_dict["prev_level"],
                            "end": final_processed_list[i - 1]["idx"],
                            "parent": final_processed_list[
                                start_idx_of_completed_callout
                            ]["prev_level"],
                        }
                        scale_meta_at_callout_start = next(
                            (
                                m_cmd
                                for m_cmd in final_processed_list[
                                    start_idx_of_completed_callout
                                ]["meta"]
                                if "model_scale" in m_cmd
                            ),
                            None,
                        )
                        if scale_meta_at_callout_start:
                            meta_data = cast(
                                MetaCommandData,
                                scale_meta_at_callout_start.get("model_scale"),
                            )
                            if (
                                meta_data
                                and "values" in meta_data
                                and meta_data["values"]
                            ):
                                callout_entry["scale"] = float(meta_data["values"][0])
                        self.callouts[start_idx_of_completed_callout] = callout_entry
                if self.continuous_step_numbers and entry_item_dict["callout"] == 0:
                    entry_item_dict["step"] = (
                        continuous_step_counter_for_main_flow  # 'step' is part of UnwrappedStepEntry via StepData
                    )
                    continuous_step_counter_for_main_flow += 1
                # No need to append to final_processed_list, it's the same list object
            if self.continuous_step_numbers:
                self.continuous_step_count = continuous_step_counter_for_main_flow - 1
                for e_fm_item_update in final_processed_list:
                    if e_fm_item_update["callout"] == 0:
                        e_fm_item_update["num_steps"] = self.continuous_step_count
            for e_fm_pli_item_update in final_processed_list:
                for meta_item_cmd_dict in e_fm_pli_item_update["meta"]:
                    if "pli_proxy" in meta_item_cmd_dict:
                        pli_proxy_data = meta_item_cmd_dict["pli_proxy"]
                        if BOM_AVAILABLE and isinstance(
                            e_fm_pli_item_update["pli_bom"], BOM
                        ):
                            for item_str_val_proxy in pli_proxy_data.get("values", []):
                                pname_val_proxy, pcol_val_str_proxy = (
                                    item_str_val_proxy.split("_")
                                    if "_" in item_str_val_proxy
                                    else (item_str_val_proxy, str(LDR_DEF_COLOUR))
                                )
                                pcol_val_proxy = int(pcol_val_str_proxy)
                                bom_part_instance_proxy = BOMPart(1, pname_val_proxy, pcol_val_proxy)  # type: ignore
                                cast(BOM, e_fm_pli_item_update["pli_bom"]).add_part(
                                    bom_part_instance_proxy
                                )
                                if self.bom:
                                    self.bom.add_part(bom_part_instance_proxy)
            return final_processed_list
        return unwrapped_list_accumulator, next_global_idx

    def parse_model_steps(
        self,
        model_source_str_content: str,
        is_top_level: bool = True,
    ) -> Tuple[Dict[int, List[LDRPart]], Dict[int, StepData]]:
        if not model_source_str_content.strip():
            return {}, {}
        pli_dict_for_this_model: Dict[int, List[LDRPart]] = {}
        steps_dict_for_this_model: Dict[int, StepData] = {}
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
                meta_cmd_data = next(
                    iter(cmd_dict_item.values())
                )  # Get the MetaCommandData
                if (
                    "scale" in cmd_dict_item
                    and "values" in meta_cmd_data
                    and meta_cmd_data["values"]
                ):
                    current_scale_for_model_view = float(meta_cmd_data["values"][0])
                elif (
                    "model_scale" in cmd_dict_item
                    and "values" in meta_cmd_data
                    and meta_cmd_data["values"]
                ):
                    model_inherent_scale_preference = float(meta_cmd_data["values"][0])
                elif (
                    "rotation_abs" in cmd_dict_item
                    and "values" in meta_cmd_data
                    and meta_cmd_data["values"]
                ):
                    v_abs_rot = [float(x_val) for x_val in meta_cmd_data["values"]]
                    current_aspect_list_for_model = [
                        -v_abs_rot[0],
                        v_abs_rot[1],
                        v_abs_rot[2],
                    ]
                    aspect_was_changed_by_meta_in_step = True
                elif (
                    "rotation_rel" in cmd_dict_item
                    and "values" in meta_cmd_data
                    and meta_cmd_data["values"]
                ):
                    v_rel_rot_vals = tuple(
                        float(x_val) for x_val in meta_cmd_data["values"]
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
                elif (
                    "rotation_pre" in cmd_dict_item
                    and "values" in meta_cmd_data
                    and meta_cmd_data["values"]
                ):
                    current_aspect_list_for_model = list(
                        preset_aspect(
                            cast(
                                Tuple[float, float, float],
                                tuple(current_aspect_list_for_model),
                            ),
                            meta_cmd_data["values"],
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

            pli_bom_for_this_step: PLIBomType = []
            if BOM_AVAILABLE:
                current_pli_bom = BOM()  # type: ignore
                if self.bom and hasattr(self.bom, "ignore_parts"):
                    current_pli_bom.ignore_parts = self.bom.ignore_parts  # type: ignore
                for p_part_item in pli_parts_for_this_step:
                    current_pli_bom.add_part(BOMPart(1, p_part_item.name, p_part_item.attrib.colour))  # type: ignore
                pli_bom_for_this_step = current_pli_bom
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

            current_step_data: StepData = {
                "step": current_step_number,  # Added step number here
                "parts": snapshot_model_parts_transformed,
                "step_parts": snapshot_step_added_parts_transformed,
                "sub_models": submodel_names_referenced_in_step_lines,
                "aspect": final_current_aspect_tuple_for_model,
                "scale": current_scale_for_model_view,
                "model_scale": model_inherent_scale_preference,
                "raw_ldraw": single_step_raw_content,
                "pli_bom": pli_bom_for_this_step,
                "meta": meta_commands_in_step,
                "aspect_change": aspect_was_changed_by_meta_in_step,
                "sub_parts": sub_parts_snapshot_view_dict,
            }
            steps_dict_for_this_model[current_step_number] = current_step_data

            if pli_parts_for_this_step:
                pli_dict_for_this_model[current_step_number] = pli_parts_for_this_step
            current_step_number += 1
            if is_top_level:
                progress_bar(
                    i + 1, len(step_blocks_raw), "Parsing Model Steps:", length=50
                )
        return pli_dict_for_this_model, steps_dict_for_this_model
