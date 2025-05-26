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
ldrmodel.py: LDraw model parsing, processing, and manipulation.

This module provides the LDRModel class, which is responsible for parsing
LDraw files (including Multi-Part Document - MPD - files), understanding their
hierarchical structure of main models and submodels, processing steps,
and handling LDraw meta-commands. It can "unwrap" a hierarchical model into
a flat, sequential list of building steps, generate Bill of Materials (BOM),
and apply transformations. Various helper functions for parsing and list
manipulation related to LDraw models are also included.
"""

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

# Conditional import for brickbom for Bill of Materials generation
try:
    from brickbom import BOM, BOMPart  # type: ignore

    BOM_AVAILABLE = True  # Flag to indicate BOM functionality is available
except ImportError:
    BOM = object  # type: ignore # Define BOM as a fallback type if brickbom is not installed
    BOMPart = object  # type: ignore
    BOM_AVAILABLE = False  # Flag to indicate BOM functionality is unavailable

from rich import print as rich_print  # type: ignore # For rich console output

# --- Constants for parsing LDraw meta lines ---
# Tokens indicating the start of blocks that might mask parts from normal view (e.g., PLI previews)
START_TOKENS = ["PLI BEGIN IGN", "BUFEXCHG STORE"]
# Tokens indicating the end of such masking blocks
END_TOKENS = ["PLI END", "BUFEXCHG RETRIEVE"]

# List of part names that are exceptions and might be included even if inside masked blocks
EXCEPTION_LIST = ["2429c01.dat"]
# List of part names to generally ignore during parsing or processing
IGNORE_LIST = ["LS02"]  # Example: LEGO Sort&Store figure, often not part of model

# List of common part substitutions (old_name, new_name) for standardization
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


# --- TypedDict Definitions for Structured Data ---
class MetaCommandData(TypedDict, total=False):
    """
    Structure for data associated with a parsed LDraw meta command.
    'total=False' means 'values' key is optional.
    """

    values: List[str]  # Optional: values extracted from the command parameters
    text: str  # Required: the full text of the meta command line


ParsedMetaCommandItem = Dict[str, MetaCommandData]


class ModelPartEntry(TypedDict):
    """Structure for an entry representing a part line (type 1) in a model definition."""

    ldrtext: str
    partname: str


class CalloutData(TypedDict, total=False):
    """
    Structure for data associated with a generated callout (a sub-assembly shown separately).
    'total=False' means 'scale' key is optional.
    """

    level: int
    end: int
    parent: int
    scale: float


PLIBomType = Union[BOM, List[Any]]


class StepData(TypedDict):
    """
    Structure for data representing a single parsed step within a model's definition.
    This holds the state and content for one "0 STEP" block.
    """

    step: int
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
    """
    Structure for an entry in the unwrapped (flattened) list of model steps.
    Inherits all fields from StepData and adds context specific to the flattened view.
    """

    idx: int
    level: int
    next_step: int
    num_steps: int
    model: str
    qty: int
    prev_level: int
    next_level: int
    page_break: bool
    callout: int


# --- Module-Level Helper Functions ---
def substitute_part(part: LDRPart) -> LDRPart:
    """
    Applies common part substitutions to an LDRPart object based on COMMON_SUBSTITUTIONS.
    Modifies the part's name in place if a substitution is found.
    """
    for old_name, new_name in COMMON_SUBSTITUTIONS:
        if part.name == old_name:
            part.name = new_name
            break
    return part


def line_has_all_tokens(line: str, tokenlist: List[str]) -> bool:
    """
    Checks if a given line (string) contains all space-separated tokens
    from any of the token groups provided in `tokenlist`.
    """
    line_tokens = line.upper().split()
    for t_group_str in tokenlist:
        required_tokens_in_group = t_group_str.upper().split()
        if all(req_token in line_tokens for req_token in required_tokens_in_group):
            return True
    return False


def parse_special_tokens(line: str) -> List[ParsedMetaCommandItem]:
    """
    Parses a single LDraw meta line (line type 0) for recognized special commands
    defined in the `ldrawpy.constants.SPECIAL_TOKENS` dictionary.
    """
    line_stripped_lower = line.strip().lower()
    line_tokens_original_case = line.strip().split()
    metas: List[ParsedMetaCommandItem] = []

    for cmd_key, token_patterns_for_cmd in SPECIAL_TOKENS.items():
        for pattern_str in token_patterns_for_cmd:
            pattern_tokens_lower = pattern_str.lower().split()
            command_keywords_in_pattern = [
                pt for pt in pattern_tokens_lower if not pt.startswith("%")
            ]
            line_starts_with_cmd = True
            if len(line_tokens_original_case) < len(command_keywords_in_pattern) + 1:
                line_starts_with_cmd = False
            else:
                for i, pattern_cmd_token in enumerate(command_keywords_in_pattern):
                    if line_tokens_original_case[i + 1].lower() != pattern_cmd_token:
                        line_starts_with_cmd = False
                        break
            if not line_starts_with_cmd:
                continue
            extracted_values: List[str] = []
            idx_last_cmd_keyword_in_pattern = -1
            for i, token in enumerate(pattern_tokens_lower):
                if not token.startswith("%"):
                    idx_last_cmd_keyword_in_pattern = i
            num_cmd_keywords = len(command_keywords_in_pattern)
            value_start_index_in_line = 1 + num_cmd_keywords
            valid_match_for_values = True
            for pattern_token_idx, pattern_token_val in enumerate(pattern_tokens_lower):
                if (
                    pattern_token_val.startswith("%")
                    and pattern_token_val[1:].isdigit()
                ):
                    placeholder_num = int(pattern_token_val[1:])
                    actual_value_idx_in_line = value_start_index_in_line + (
                        placeholder_num - 1
                    )
                    if actual_value_idx_in_line < len(line_tokens_original_case):
                        extracted_values.append(
                            line_tokens_original_case[actual_value_idx_in_line]
                        )
                    else:
                        valid_match_for_values = False
                        break
            if not valid_match_for_values and any(
                pt.startswith("%") for pt in pattern_tokens_lower
            ):
                continue
            current_meta_data: MetaCommandData = {"text": line.strip()}
            if extracted_values:
                current_meta_data["values"] = extracted_values
            metas.append({cmd_key: current_meta_data})  # type: ignore
            break
        if metas and metas[-1].get(cmd_key):
            break
    return metas


def get_meta_commands(ldr_string_content: str) -> List[ParsedMetaCommandItem]:
    """
    Extracts all recognized LDraw meta commands (type 0 lines) from a multi-line
    LDraw string content.
    """
    parsed_commands_list: List[ParsedMetaCommandItem] = []
    for line_str in ldr_string_content.splitlines():
        stripped_line_str = line_str.lstrip()
        if not stripped_line_str or not stripped_line_str.startswith("0 "):
            continue
        meta_commands_from_line = parse_special_tokens(line_str)
        if meta_commands_from_line:
            parsed_commands_list.extend(meta_commands_from_line)
    return parsed_commands_list


def get_parts_from_model(ldr_string_content: str) -> List[ModelPartEntry]:
    """
    Extracts type 1 LDraw lines (representing parts or submodel references)
    from an LDraw string.
    """
    parts_found: List[ModelPartEntry] = []
    lines_in_content = ldr_string_content.splitlines()
    masking_block_depth = 0
    is_in_bufexchg_store_block = False
    for line_str in lines_in_content:
        stripped_line_str = line_str.lstrip()
        if not stripped_line_str:
            continue
        if line_has_all_tokens(line_str, ["BUFEXCHG STORE"]):
            is_in_bufexchg_store_block = True
        if line_has_all_tokens(line_str, ["BUFEXCHG RETRIEVE"]):
            is_in_bufexchg_store_block = False
        if line_has_all_tokens(line_str, START_TOKENS):
            masking_block_depth += 1
        if line_has_all_tokens(line_str, END_TOKENS):
            if masking_block_depth > 0:
                masking_block_depth -= 1
        try:
            line_type_char = stripped_line_str[0]
            if not line_type_char.isdigit():
                continue
            line_type = int(line_type_char)
        except (ValueError, IndexError):
            continue
        if line_type == 1:
            tokens_in_part_line = line_str.split()
            if len(tokens_in_part_line) < 14:
                continue
            part_entry_dict: ModelPartEntry = {
                "ldrtext": line_str,
                "partname": " ".join(tokens_in_part_line[14:]),
            }
            if masking_block_depth == 0:
                parts_found.append(part_entry_dict)
            else:
                if part_entry_dict["partname"] in EXCEPTION_LIST:
                    parts_found.append(part_entry_dict)
                elif not is_in_bufexchg_store_block and part_entry_dict[
                    "partname"
                ].endswith(".ldr"):
                    parts_found.append(part_entry_dict)
    return parts_found


def recursive_parse_model(
    model_entries_list: List[ModelPartEntry],
    all_submodels_definitions: Dict[str, List[ModelPartEntry]],
    output_resolved_parts_list: List[LDRPart],
    current_cumulative_offset: Vector = Vector(0, 0, 0),  # type: ignore
    current_cumulative_matrix: Matrix = Identity(),  # type: ignore
    reset_output_list_on_call: bool = False,  # CORRECTED parameter name
    filter_for_specific_submodel_name: Optional[str] = None,
):
    """
    Recursively parses a list of model entries, resolving submodels.
    """
    if reset_output_list_on_call:  # CORRECTED usage
        output_resolved_parts_list.clear()

    for entry_dict_item in model_entries_list:
        part_or_submodel_name = entry_dict_item["partname"]
        ldr_text_line = entry_dict_item["ldrtext"]
        if (
            filter_for_specific_submodel_name
            and part_or_submodel_name != filter_for_specific_submodel_name
        ):
            continue
        if part_or_submodel_name in all_submodels_definitions:
            submodel_definition_entries_list = all_submodels_definitions[
                part_or_submodel_name
            ]
            submodel_reference_part_instance = LDRPart()
            if submodel_reference_part_instance.from_str(ldr_text_line) is None:
                continue
            new_cumulative_matrix_for_submodel = (
                current_cumulative_matrix
                * submodel_reference_part_instance.attrib.matrix
            )
            new_cumulative_offset_for_submodel = (
                current_cumulative_matrix * submodel_reference_part_instance.attrib.loc + current_cumulative_offset  # type: ignore
            )
            recursive_parse_model(
                submodel_definition_entries_list,
                all_submodels_definitions,
                output_resolved_parts_list,
                new_cumulative_offset_for_submodel,
                new_cumulative_matrix_for_submodel,
                False,
                None,
            )
        elif filter_for_specific_submodel_name is None:
            actual_part_object = LDRPart()
            if actual_part_object.from_str(ldr_text_line) is None:
                continue
            actual_part_object = substitute_part(actual_part_object)
            actual_part_object.transform(
                matrix=current_cumulative_matrix, offset=current_cumulative_offset
            )
            if (
                actual_part_object.name not in IGNORE_LIST
                and actual_part_object.name.upper() not in IGNORE_LIST
            ):
                output_resolved_parts_list.append(actual_part_object)


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


# --- Class LDRModel ---
class LDRModel:
    """
    Represents an LDraw model, loaded from an .ldr or .mpd file.
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
        """
        Initializes the LDRModel by loading and preparing to parse an LDraw file.
        """
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
        """Returns a string summary of the LDRModel."""
        return (
            f"LDRModel: {self.title}\n"
            f"  File: {str(self.filepath)}\n"
            f"  Global origin: {self.global_origin} Global aspect: {self.global_aspect}\n"
            f"  Number of steps in root model: {len(self.steps)}\n"
            f"  Number of defined sub-models: {len(self.sub_models)}"
        )

    def __getitem__(self, key: int) -> UnwrappedStepEntry:
        """
        Allows dictionary-like access to the unwrapped steps by their global index.
        """
        if self.unwrapped is None:
            self.unwrap()
        current_unwrapped_steps = self.unwrapped
        if current_unwrapped_steps is None:
            raise IndexError(
                "Model not unwrapped or unwrap process failed to produce a list."
            )
        return current_unwrapped_steps[key]

    def print_step_dict(self, step_key: int):
        """
        Prints a detailed, formatted representation of a specific parsed step
        from the main model's `self.steps` dictionary. (For debugging).
        """
        if step_key in self.steps:
            step_data_dict = self.steps[step_key]
            rich_print(
                f"[bold magenta]--- Step {step_key} (Raw Parsed Data) ---[/bold magenta]"
            )
            for key_name_in_step, value_item in step_data_dict.items():
                if key_name_in_step == "sub_parts" and isinstance(value_item, dict):
                    rich_print(f"[bold blue]{key_name_in_step}:[/bold blue]")
                    for sub_model_name, sub_model_parts_list in value_item.items():
                        rich_print(
                            f"  [cyan]{sub_model_name}:[/cyan] ({len(sub_model_parts_list)} parts)"
                        )
                elif (
                    (key_name_in_step == "parts" or key_name_in_step == "step_parts")
                    and isinstance(value_item, list)
                    and all(isinstance(p, LDRPart) for p in value_item)
                ):
                    rich_print(
                        f"[bold blue]{key_name_in_step}:[/bold blue] ({len(value_item)} LDRParts)"
                    )
                elif (
                    key_name_in_step == "pli_bom"
                    and BOM_AVAILABLE
                    and isinstance(value_item, BOM)
                ):
                    rich_print(f"[bold blue]{key_name_in_step}:[/bold blue]")
                    if hasattr(value_item, "summary_str"):
                        rich_print(f"  {value_item.summary_str()}")  # type: ignore
                    else:
                        rich_print(f"  {value_item}")
                elif key_name_in_step == "meta" and isinstance(value_item, list):
                    rich_print(
                        f"[bold blue]{key_name_in_step}:[/bold blue] ({len(value_item)} meta commands)"
                    )
                else:
                    rich_print(
                        f"[bold blue]{key_name_in_step}:[/bold blue] {value_item}"
                    )
        else:
            rich_print(f"Step {step_key} not found in main model steps (self.steps).")

    def print_unwrapped_dict(self, global_step_idx: int):
        """
        Prints a detailed, formatted representation of a specific step
        from the `self.unwrapped` list using its global index. (For debugging).
        """
        if self.unwrapped is None or not (0 <= global_step_idx < len(self.unwrapped)):
            rich_print(f"Index {global_step_idx} out of bounds for unwrapped model.")
            return
        unwrapped_step_data_dict = self.unwrapped[global_step_idx]
        rich_print(
            f"[bold magenta]--- Unwrapped Step (Global Index {global_step_idx}) ---[/bold magenta]"
        )
        for key_name_in_step, value_item in unwrapped_step_data_dict.items():
            if (
                (key_name_in_step == "parts" or key_name_in_step == "step_parts")
                and isinstance(value_item, list)
                and all(isinstance(p, LDRPart) for p in value_item)
            ):
                rich_print(
                    f"[bold green]{key_name_in_step}:[/bold green] ({len(value_item)} LDRParts)"
                )
            elif (
                key_name_in_step == "pli_bom"
                and BOM_AVAILABLE
                and isinstance(value_item, BOM)
            ):
                rich_print(f"[bold green]{key_name_in_step}:[/bold green]")
                if hasattr(value_item, "summary_str"):
                    rich_print(f"  {value_item.summary_str()}")  # type: ignore
                else:
                    rich_print(f"  {value_item}")
            elif key_name_in_step == "meta" and isinstance(value_item, list):
                rich_print(
                    f"[bold green]{key_name_in_step}:[/bold green] ({len(value_item)} meta commands)"
                )
            else:
                rich_print(f"[bold green]{key_name_in_step}:[/bold green] {value_item}")

    def print_unwrapped_verbose(self):
        """Prints a verbose, single-line summary for each step in the unwrapped model."""
        if not self.unwrapped:
            rich_print("Model not unwrapped yet. Call parse_file() first.")
            return
        rich_print("[bold]--- Unwrapped Model (Verbose Summary) ---[/bold]")
        for i, unwrapped_step_item in enumerate(self.unwrapped):
            aspect_tuple = unwrapped_step_item["aspect"]
            rich_print(
                f"{i:3d}. idx:{unwrapped_step_item['idx']:3} "
                f"[pl:{unwrapped_step_item['prev_level']} l:{unwrapped_step_item['level']} nl:{unwrapped_step_item['next_level']}] "
                f"[s:{unwrapped_step_item['step']:2} ns:{unwrapped_step_item['next_step']:2} sc:{unwrapped_step_item['num_steps']:2}] "
                f"{str(unwrapped_step_item['model'])[:16]:<16} q:{unwrapped_step_item['qty']:d} sc:{unwrapped_step_item['scale']:.2f} "
                f"asp:({aspect_tuple[0]:3.0f},{aspect_tuple[1]:4.0f},{aspect_tuple[2]:3.0f})"
            )

    def print_unwrapped(self):
        """Prints a formatted, human-readable summary for each step in the unwrapped model."""
        if not self.unwrapped:
            rich_print("Model not unwrapped yet. Call parse_file() first.")
            return
        rich_print("[bold]--- Unwrapped Model Steps (Formatted) ---[/bold]")
        for unwrapped_step_item in self.unwrapped:
            self.print_step(unwrapped_step_item)

    def print_step(self, step_data_entry: UnwrappedStepEntry):
        """
        Helper method to print a single step from the unwrapped model list in a
        formatted, human-readable way.
        """
        is_page_break = "break" if step_data_entry["page_break"] else ""
        callout_level_str = str(step_data_entry["callout"])
        model_display_name = str(step_data_entry["model"]).replace(".ldr", "")[:16]
        quantity_str = (
            f"({step_data_entry['qty']:2d}x)" if step_data_entry["qty"] > 0 else "     "
        )
        level_indent_str = " " * step_data_entry["level"]
        level_display_str = f"{level_indent_str}Level {step_data_entry['level']}"
        level_display_str_padded = f"{level_display_str:<11}"
        pli_bom_data = step_data_entry["pli_bom"]
        num_parts_in_pli = 0
        if (
            BOM_AVAILABLE
            and isinstance(pli_bom_data, BOM)
            and hasattr(pli_bom_data, "parts")
        ):
            num_parts_in_pli = len(pli_bom_data.parts)  # type: ignore
        elif isinstance(pli_bom_data, list):
            num_parts_in_pli = len(pli_bom_data)
        parts_count_display_str = f"({num_parts_in_pli:2d}x pcs)"
        meta_tags_display_list = []
        for meta_command_item_dict in step_data_entry["meta"]:
            for command_name_key, meta_command_detail in meta_command_item_dict.items():
                display_tag_str = command_name_key.replace("_", " ")
                if (
                    command_name_key == "columns"
                    and "values" in meta_command_detail
                    and meta_command_detail["values"]
                ):
                    meta_tags_display_list.append(
                        f"[green]COL{meta_command_detail['values'][0]}[/]"
                    )
                else:
                    meta_tags_display_list.append(f"[dim]{display_tag_str}[/dim]")
        meta_commands_display_str = " ".join(meta_tags_display_list)
        aspect_tuple_for_display = step_data_entry["aspect"]
        base_info_str = (
            f"{step_data_entry['idx']:3}. {level_display_str_padded} Step "
            f"{'[yellow]' if callout_level_str != '0' else '[green]'}{step_data_entry['step']:3}/{step_data_entry['num_steps']:3}{'[/]'} "
            f"Model: {'[red]' if callout_level_str != '0' and model_display_name != 'root' else '[green]'}{model_display_name:<16}{'[/]'} "
            f"{quantity_str} {parts_count_display_str} scale: {step_data_entry['scale']:.2f} "
            f"({aspect_tuple_for_display[0]:3.0f},{aspect_tuple_for_display[1]:4.0f},{aspect_tuple_for_display[2]:3.0f})"
        )
        callout_info_str = f" {'[yellow]' if callout_level_str != '0' else '[dim]'}{callout_level_str}{'[/]'}"
        page_break_info_str = f" {'[magenta]BR[/]' if is_page_break else ''}"
        rich_print(
            f"{base_info_str}{callout_info_str}{page_break_info_str} {meta_commands_display_str}"
        )

    def transform_parts_to(
        self,
        parts_list: List[LDRPart],
        target_origin: Optional[Union[Tuple[float, float, float], Vector]] = None,  # type: ignore
        target_aspect: Optional[Union[Tuple[float, float, float], Vector]] = None,  # type: ignore
        use_pli_exceptions: bool = False,
    ) -> List[LDRPart]:
        """
        Transforms a list of LDRPart objects to a specific target origin and aspect.
        """
        final_target_aspect = (
            target_aspect if target_aspect is not None else self.pli_aspect
        )
        final_target_origin_vector = (
            safe_vector(target_origin) if target_origin is not None else None
        )
        transformed_parts_list = []
        for p_part_obj in parts_list:
            new_part_obj = p_part_obj.copy()
            current_part_aspect_list = list(final_target_aspect)
            if use_pli_exceptions and p_part_obj.name in self.pli_exceptions:
                current_part_aspect_list = list(self.pli_exceptions[p_part_obj.name])
            new_part_obj.set_rotation(tuple(current_part_aspect_list))
            if final_target_origin_vector:
                new_part_obj.move_to(final_target_origin_vector)
            transformed_parts_list.append(new_part_obj)
        return transformed_parts_list

    def transform_parts(
        self,
        parts_input_list: Union[List[LDRPart], List[str]],
        relative_offset: Optional[Union[Tuple[float, float, float], Vector]] = None,  # type: ignore
        relative_aspect: Optional[Union[Tuple[float, float, float], Vector]] = None,  # type: ignore # CORRECTED: Changed from aspect=
    ) -> Union[List[LDRPart], List[str]]:
        """
        Applies a relative transformation (offset and aspect/rotation) to a list of parts.
        """
        final_relative_aspect = (
            relative_aspect if relative_aspect is not None else self.global_aspect
        )  # CORRECTED: Changed from aspect
        final_relative_offset_vector = (
            safe_vector(relative_offset) if relative_offset is not None else Vector(0, 0, 0)  # type: ignore
        )
        if not parts_input_list:
            return []
        if all(isinstance(p_item, LDRPart) for p_item in parts_input_list):
            processed_ldr_part_objects: List[LDRPart] = []
            for p_obj_item in cast(List[LDRPart], parts_input_list):
                new_part_obj = p_obj_item.copy()
                new_part_obj.rotate_by(final_relative_aspect)
                new_part_obj.move_by(final_relative_offset_vector)
                processed_ldr_part_objects.append(new_part_obj)
            return processed_ldr_part_objects
        elif all(isinstance(p_item, str) for p_item in parts_input_list):
            processed_ldr_string_lines: List[str] = []
            for p_str_item_line in cast(List[str], parts_input_list):
                part_obj_from_string = LDRPart()
                if part_obj_from_string.from_str(p_str_item_line):
                    part_obj_from_string.rotate_by(final_relative_aspect)
                    part_obj_from_string.move_by(final_relative_offset_vector)
                    processed_ldr_string_lines.append(str(part_obj_from_string))
                else:
                    processed_ldr_string_lines.append(p_str_item_line)
            return processed_ldr_string_lines
        return []

    def parse_file(self):
        """
        Parses the LDraw model file specified during initialization.
        """
        self.sub_models = {}
        self.sub_model_str = {}
        try:
            with open(self.filepath, "rt", encoding="utf-8") as fp_handle:
                full_file_content = fp_handle.read()
        except FileNotFoundError:
            rich_print(f"Error: File {str(self.filepath)} not found.")
            self.pli, self.steps = {}, {}
            return
        except Exception as e:
            rich_print(f"Error reading file {str(self.filepath)}: {e}")
            self.pli, self.steps = {}, {}
            return
        file_blocks_list = full_file_content.split("0 FILE")
        root_model_string_content = ""
        sub_file_blocks_list = []
        if not full_file_content.strip().startswith("0 FILE") and file_blocks_list:
            root_model_string_content = file_blocks_list[0]
            sub_file_blocks_list = file_blocks_list[1:]
        elif len(file_blocks_list) > 1 and full_file_content.strip().startswith(
            "0 FILE"
        ):
            root_model_string_content = "0 FILE " + file_blocks_list[1].strip()
            sub_file_blocks_list = file_blocks_list[2:]
        else:
            root_model_string_content = full_file_content
            sub_file_blocks_list = []
        for sub_block_text_content in sub_file_blocks_list:
            if not sub_block_text_content.strip():
                continue
            full_sub_model_definition_str = "0 FILE " + sub_block_text_content.strip()
            lines_in_submodel_def = full_sub_model_definition_str.splitlines()
            if not lines_in_submodel_def:
                continue
            submodel_name_key = (
                lines_in_submodel_def[0].replace("0 FILE", "").strip().lower()
            )
            if submodel_name_key:
                self.sub_model_str[submodel_name_key] = full_sub_model_definition_str
                self.sub_models[submodel_name_key] = get_parts_from_model(
                    full_sub_model_definition_str
                )
        if root_model_string_content.strip():
            self.pli, self.steps = self.parse_model_steps(
                root_model_string_content, is_top_level=True
            )
        else:
            rich_print(
                f"Warning: Root model content for {str(self.filepath)} is empty after MPD processing. "
                "Check file structure."
            )
            self.pli, self.steps = {}, {}
        self.unwrap()

    def unwrap(self):
        """
        Flattens the hierarchical model structure into `self.unwrapped`.
        """
        if self.unwrapped is not None:
            return
        self._parsed_submodel_steps_cache = {}
        for sub_name_key, sub_model_raw_str in self.sub_model_str.items():
            _, steps_dict_for_this_submodel = self.parse_model_steps(
                sub_model_raw_str, is_top_level=False
            )
            self._parsed_submodel_steps_cache[sub_name_key] = (
                steps_dict_for_this_submodel
            )
        initial_unwrapped_accumulator: List[UnwrappedStepEntry] = []
        self.unwrapped = self._unwrap_model_recursive(
            current_model_steps_dict=self.steps,
            unwrapped_list_accumulator=initial_unwrapped_accumulator,
        )

    def _unwrap_model_recursive(
        self,
        current_model_steps_dict: Dict[int, StepData],
        current_idx_offset: int = 0,
        current_level: int = 0,
        model_name_for_these_steps: str = "root",
        model_qty_for_this_instance: int = 1,
        unwrapped_list_accumulator: List[UnwrappedStepEntry] = [],
    ) -> Union[List[UnwrappedStepEntry], Tuple[List[UnwrappedStepEntry], int]]:
        """
        Recursive helper for `unwrap()`.
        """
        is_initial_top_level_call = (
            current_level == 0
            and current_idx_offset == 0
            and model_name_for_these_steps == "root"
        )

        next_global_idx_counter = current_idx_offset
        sorted_step_numbers_in_current_model = sorted(current_model_steps_dict.keys())

        for step_number_in_this_model in sorted_step_numbers_in_current_model:
            step_data_from_parent_model = current_model_steps_dict[
                step_number_in_this_model
            ]
            if step_data_from_parent_model.get("sub_models"):
                unique_submodel_references_in_this_step = unique_set(
                    step_data_from_parent_model["sub_models"]
                )  # Ensure unique_set is available
                for (
                    submodel_name_referenced,
                    quantity_of_this_submodel,
                ) in unique_submodel_references_in_this_step.items():
                    submodel_definition_steps_dict = (
                        self._parsed_submodel_steps_cache.get(submodel_name_referenced)
                    )
                    if not submodel_definition_steps_dict:
                        continue
                    recursive_call_result = self._unwrap_model_recursive(
                        submodel_definition_steps_dict,
                        next_global_idx_counter,
                        current_level + 1,
                        submodel_name_referenced,
                        quantity_of_this_submodel,
                        unwrapped_list_accumulator,
                    )
                    if (
                        isinstance(recursive_call_result, tuple)
                        and len(recursive_call_result) == 2
                    ):
                        next_global_idx_counter = recursive_call_result[1]

            unwrapped_step_entry: UnwrappedStepEntry = {
                "idx": next_global_idx_counter,
                "level": current_level,
                "step": step_number_in_this_model,
                "next_step": (
                    step_number_in_this_model + 1
                    if step_number_in_this_model
                    < len(sorted_step_numbers_in_current_model)
                    else step_number_in_this_model
                ),
                "num_steps": len(sorted_step_numbers_in_current_model),
                "model": model_name_for_these_steps,
                "qty": model_qty_for_this_instance,
                "scale": step_data_from_parent_model["scale"],
                "model_scale": step_data_from_parent_model["model_scale"],
                "aspect": step_data_from_parent_model["aspect"],
                "parts": step_data_from_parent_model["parts"],
                "step_parts": step_data_from_parent_model["step_parts"],
                "sub_models": step_data_from_parent_model["sub_models"],
                "pli_bom": step_data_from_parent_model["pli_bom"],
                "meta": step_data_from_parent_model["meta"],
                "aspect_change": step_data_from_parent_model["aspect_change"],
                "raw_ldraw": step_data_from_parent_model["raw_ldraw"],
                "sub_parts": step_data_from_parent_model["sub_parts"],
                "prev_level": 0,
                "next_level": 0,
                "page_break": False,
                "callout": 0,
            }
            unwrapped_list_accumulator.append(unwrapped_step_entry)
            next_global_idx_counter += 1

        if current_level == 0:  # Post-processing for the top-level call
            final_processed_list: List[UnwrappedStepEntry] = unwrapped_list_accumulator
            continuous_step_counter_main_flow = 1
            self.callouts = {}
            active_callout_start_global_indices: Dict[int, int] = {}
            for i, current_entry_dict in enumerate(final_processed_list):
                current_entry_dict["prev_level"] = (
                    final_processed_list[i - 1]["level"] if i > 0 else 0
                )
                current_entry_dict["next_level"] = (
                    final_processed_list[i + 1]["level"]
                    if i < len(final_processed_list) - 1
                    else current_entry_dict["level"]
                )
                has_page_break_meta = any(
                    "page_break" in m_cmd for m_cmd in current_entry_dict["meta"]
                )
                is_no_callout_meta = any(
                    "no_callout" in m_cmd for m_cmd in current_entry_dict["meta"]
                )
                current_entry_dict["page_break"] = has_page_break_meta or (
                    current_entry_dict["next_level"] > current_entry_dict["level"]
                    and current_entry_dict["num_steps"] >= self.callout_step_thr
                    and not is_no_callout_meta
                )
                if (
                    current_entry_dict["level"] > current_entry_dict["prev_level"]
                    and not is_no_callout_meta
                    and current_entry_dict["num_steps"] < self.callout_step_thr
                ):
                    active_callout_start_global_indices[current_entry_dict["level"]] = (
                        current_entry_dict["idx"]
                    )
                current_callout_level_for_this_step = 0
                for active_level_key in sorted(
                    active_callout_start_global_indices.keys(), reverse=True
                ):
                    if current_entry_dict["level"] >= active_level_key:
                        current_callout_level_for_this_step = active_level_key
                        break
                current_entry_dict["callout"] = current_callout_level_for_this_step
                if current_entry_dict["level"] < current_entry_dict["prev_level"]:
                    if (
                        current_entry_dict["prev_level"]
                        in active_callout_start_global_indices
                    ):
                        global_start_idx_of_completed_callout = (
                            active_callout_start_global_indices.pop(
                                current_entry_dict["prev_level"]
                            )
                        )
                        callout_entry_data: CalloutData = {
                            "level": current_entry_dict["prev_level"],
                            "end": final_processed_list[i - 1]["idx"],
                            "parent": final_processed_list[
                                global_start_idx_of_completed_callout
                            ]["prev_level"],
                        }
                        callout_start_step_meta = final_processed_list[
                            global_start_idx_of_completed_callout
                        ]["meta"]
                        scale_meta_command = next(
                            (
                                m_cmd
                                for m_cmd in callout_start_step_meta
                                if "model_scale" in m_cmd
                            ),
                            None,
                        )
                        if scale_meta_command:
                            meta_data_value = cast(
                                MetaCommandData, scale_meta_command.get("model_scale")
                            )
                            if (
                                meta_data_value
                                and "values" in meta_data_value
                                and meta_data_value["values"]
                            ):
                                callout_entry_data["scale"] = float(
                                    meta_data_value["values"][0]
                                )
                        self.callouts[global_start_idx_of_completed_callout] = (
                            callout_entry_data
                        )
                if self.continuous_step_numbers and current_entry_dict["callout"] == 0:
                    current_entry_dict["step"] = continuous_step_counter_main_flow
                    continuous_step_counter_main_flow += 1
            if self.continuous_step_numbers:
                self.continuous_step_count = continuous_step_counter_main_flow - 1
                for entry_to_update_numsteps in final_processed_list:
                    if entry_to_update_numsteps["callout"] == 0:
                        entry_to_update_numsteps["num_steps"] = (
                            self.continuous_step_count
                        )
            for entry_for_pli_proxy in final_processed_list:
                for meta_command_item in entry_for_pli_proxy["meta"]:
                    if "pli_proxy" in meta_command_item:
                        pli_proxy_data_values = meta_command_item["pli_proxy"].get(
                            "values", []
                        )
                        if BOM_AVAILABLE and isinstance(
                            entry_for_pli_proxy["pli_bom"], BOM
                        ):
                            pli_bom_instance = cast(BOM, entry_for_pli_proxy["pli_bom"])
                            for item_value_str in pli_proxy_data_values:
                                part_name_str, colour_code_str = (
                                    item_value_str.split("_")
                                    if "_" in item_value_str
                                    else (item_value_str, str(LDR_DEF_COLOUR))
                                )
                                colour_code_int = int(colour_code_str)
                                bom_part_proxy_instance = BOMPart(1, part_name_str, colour_code_int)  # type: ignore
                                pli_bom_instance.add_part(bom_part_proxy_instance)
                                if self.bom:
                                    self.bom.add_part(bom_part_proxy_instance)
            return final_processed_list
        return unwrapped_list_accumulator, next_global_idx_counter

    def parse_model_steps(
        self,
        model_source_str_content: str,
        is_top_level: bool = True,
    ) -> Tuple[Dict[int, List[LDRPart]], Dict[int, StepData]]:
        """
        Parses the LDraw string content of a single model (or submodel) into discrete steps.
        """
        if not model_source_str_content.strip():
            return {}, {}
        pli_dict_for_this_model_parse: Dict[int, List[LDRPart]] = {}
        steps_dict_for_this_model_parse: Dict[int, StepData] = {}
        step_blocks_raw_text = model_source_str_content.split("0 STEP")
        cumulative_parts_for_snapshot_view: List[LDRPart] = []
        current_aspect_for_model_view: List[float] = list(
            self.global_aspect if is_top_level else self.PARAMS["global_aspect"]
        )
        current_scale_for_model_render = (
            self.global_scale if is_top_level else self.PARAMS["global_scale"]
        )
        model_inherent_display_scale = current_scale_for_model_render
        step_counter = 1

        for i, single_step_raw_ldr_content in enumerate(step_blocks_raw_text):
            if (
                i == 0
                and not single_step_raw_ldr_content.strip()
                and len(step_blocks_raw_text) > 1
            ):
                continue
            parsed_meta_commands_for_step = get_meta_commands(
                single_step_raw_ldr_content
            )
            aspect_was_changed_by_meta_this_step = False
            for meta_cmd_item_dict in parsed_meta_commands_for_step:
                for command_name, command_data in meta_cmd_item_dict.items():
                    command_values = command_data.get("values")
                    if command_values:
                        if command_name == "scale":
                            current_scale_for_model_render = float(command_values[0])
                        elif command_name == "model_scale":
                            model_inherent_display_scale = float(command_values[0])
                        elif command_name == "rotation_abs":
                            abs_rotation_values = [
                                float(x_val) for x_val in command_values
                            ]
                            current_aspect_for_model_view = [
                                -abs_rotation_values[0],
                                abs_rotation_values[1],
                                abs_rotation_values[2],
                            ]
                            aspect_was_changed_by_meta_this_step = True
                        elif command_name == "rotation_rel":
                            rel_rotation_values = tuple(
                                float(x_val) for x_val in command_values
                            )
                            current_aspect_for_model_view[0] -= rel_rotation_values[0]
                            current_aspect_for_model_view[1] += rel_rotation_values[1]
                            current_aspect_for_model_view[2] += rel_rotation_values[2]
                            current_aspect_for_model_view = list(
                                norm_aspect(
                                    cast(
                                        Tuple[float, float, float],
                                        tuple(current_aspect_for_model_view),
                                    )
                                )
                            )
                            aspect_was_changed_by_meta_this_step = True
                        elif command_name == "rotation_pre":
                            current_aspect_for_model_view = list(
                                preset_aspect(
                                    cast(
                                        Tuple[float, float, float],
                                        tuple(current_aspect_for_model_view),
                                    ),
                                    command_values,
                                )
                            )
                            aspect_was_changed_by_meta_this_step = True
            part_reference_entry_dicts_in_step = get_parts_from_model(
                single_step_raw_ldr_content
            )
            ldr_parts_added_in_this_step_definition: List[LDRPart] = []
            recursive_parse_model(
                part_reference_entry_dicts_in_step,
                self.sub_models,
                ldr_parts_added_in_this_step_definition,
                reset_output_list_on_call=True,  # CORRECTED keyword argument
            )
            if (
                not ldr_parts_added_in_this_step_definition
                and not parsed_meta_commands_for_step
                and (i > 0 or not single_step_raw_ldr_content.strip())
            ):
                continue
            pli_parts_for_this_step_view = self.transform_parts_to(
                ldr_parts_added_in_this_step_definition,
                target_origin=(0, 0, 0),
                target_aspect=self.pli_aspect,  # CORRECTED: Changed from aspect=
                use_pli_exceptions=True,
            )
            cumulative_parts_for_snapshot_view.extend(
                ldr_parts_added_in_this_step_definition
            )
            final_current_aspect_for_model_view_tuple = cast(
                Tuple[float, float, float], tuple(current_aspect_for_model_view)
            )
            snapshot_all_model_parts_transformed_union = self.transform_parts(
                cumulative_parts_for_snapshot_view,
                relative_aspect=final_current_aspect_for_model_view_tuple,  # CORRECTED: Changed from aspect=
            )
            snapshot_all_model_parts_transformed: List[LDRPart] = (
                cast(List[LDRPart], snapshot_all_model_parts_transformed_union)
                if isinstance(snapshot_all_model_parts_transformed_union, list)
                and all(
                    isinstance(p, LDRPart)
                    for p in snapshot_all_model_parts_transformed_union
                )
                else []
            )
            snapshot_step_only_added_parts_transformed_union = self.transform_parts(
                ldr_parts_added_in_this_step_definition,
                relative_aspect=final_current_aspect_for_model_view_tuple,  # CORRECTED: Changed from aspect=
            )
            snapshot_step_only_added_parts_transformed: List[LDRPart] = (
                cast(List[LDRPart], snapshot_step_only_added_parts_transformed_union)
                if isinstance(snapshot_step_only_added_parts_transformed_union, list)
                and all(
                    isinstance(p, LDRPart)
                    for p in snapshot_step_only_added_parts_transformed_union
                )
                else []
            )
            pli_bom_for_this_step_obj: PLIBomType = []
            if BOM_AVAILABLE:
                current_step_pli_bom = BOM()  # type: ignore
                if self.bom and hasattr(self.bom, "ignore_parts"):
                    current_step_pli_bom.ignore_parts = self.bom.ignore_parts  # type: ignore
                for p_part_obj_for_bom in pli_parts_for_this_step_view:
                    current_step_pli_bom.add_part(BOMPart(1, p_part_obj_for_bom.name, p_part_obj_for_bom.attrib.colour))  # type: ignore
                pli_bom_for_this_step_obj = current_step_pli_bom
                if is_top_level and self.bom:
                    for p_part_obj_for_bom in pli_parts_for_this_step_view:
                        self.bom.add_part(BOMPart(1, p_part_obj_for_bom.name, p_part_obj_for_bom.attrib.colour))  # type: ignore
            submodel_names_in_step_lines = [
                entry["partname"]
                for entry in part_reference_entry_dicts_in_step
                if entry["partname"] in self.sub_models
            ]
            sub_parts_content_for_snapshot_view: Dict[str, List[LDRPart]] = {}
            for submodel_name_key in unique_set(
                submodel_names_in_step_lines
            ):  # Ensure unique_set is available
                submodel_definition_entries = self.sub_models.get(submodel_name_key, [])
                temp_list_for_submodel_resolved_parts: List[LDRPart] = []
                recursive_parse_model(
                    submodel_definition_entries,
                    self.sub_models,
                    temp_list_for_submodel_resolved_parts,
                    reset_output_list_on_call=True,  # CORRECTED keyword argument
                )
                transformed_submodel_content_parts_union = self.transform_parts(
                    temp_list_for_submodel_resolved_parts,
                    relative_aspect=final_current_aspect_for_model_view_tuple,  # CORRECTED: Changed from aspect=
                )
                if isinstance(transformed_submodel_content_parts_union, list) and all(
                    isinstance(tsp_item, LDRPart)
                    for tsp_item in transformed_submodel_content_parts_union
                ):
                    sub_parts_content_for_snapshot_view[submodel_name_key] = cast(
                        List[LDRPart], transformed_submodel_content_parts_union
                    )
            current_step_data_dict: StepData = {
                "step": step_counter,
                "parts": snapshot_all_model_parts_transformed,
                "step_parts": snapshot_step_only_added_parts_transformed,
                "sub_models": submodel_names_in_step_lines,
                "aspect": final_current_aspect_for_model_view_tuple,
                "scale": current_scale_for_model_render,
                "model_scale": model_inherent_display_scale,
                "raw_ldraw": single_step_raw_ldr_content,
                "pli_bom": pli_bom_for_this_step_obj,
                "meta": parsed_meta_commands_for_step,
                "aspect_change": aspect_was_changed_by_meta_this_step,
                "sub_parts": sub_parts_content_for_snapshot_view,
            }
            steps_dict_for_this_model_parse[step_counter] = current_step_data_dict
            if pli_parts_for_this_step_view:
                pli_dict_for_this_model_parse[step_counter] = (
                    pli_parts_for_this_step_view
                )
            step_counter += 1
            if is_top_level:
                progress_bar(
                    i + 1, len(step_blocks_raw_text), "Parsing Model Steps:", length=50
                )
        return pli_dict_for_this_model_parse, steps_dict_for_this_model_parse
