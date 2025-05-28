"""
ldrawpy - A Python utility package for LDraw files.

This package provides classes and functions for creating, modifying, reading,
and processing LDraw files and their associated data structures. It includes
tools for handling LDraw primitives (parts, lines, triangles, quads), colours,
geometric shapes, model parsing, LDView rendering automation, arrow generation
for instructions, and pretty-printing LDraw content to the console.

The main components are organized into submodules:
- constants: Defines common LDraw constants, colour codes, and meta-command tokens.
- ldrcolour: Provides the LDRColour class for colour management and conversions.
- ldrcolourdict: Contains dictionaries mapping LDraw colour codes to names, RGB, etc.
- ldrhelpers: Utility functions for LDraw value formatting, geometry, and list manipulation.
- ldrprimitives: Defines core LDraw objects like LDRAttrib, LDRPart, LDRLine, etc.
- ldrshapes: Classes for generating common geometric shapes using LDraw primitives.
- ldrmodel: The LDRModel class for parsing and managing complex LDraw models and MPD files.
- ldvrender: The LDViewRender class for automating rendering with LDView.
- ldrarrows: Utilities for generating arrow annotations in LDraw.
- ldrpprint: Functions for pretty-printing LDraw lines with syntax highlighting.
- ldrawpy (module): Contains miscellaneous top-level utility functions.

Key classes and functions are re-exported here for easier access from the package root.
"""

# import os # Retained for potential use, though not directly used by __init__ itself.

# fmt: off
__project__ = 'ldrawpyV2' 
__version__ = '2.0.0a1' 
# fmt: on

VERSION = f"{__project__}-{__version__}"

# --- Selective and organized imports for the package's public API ---

# From .constants
from .constants import (
    ASPECT_DICT,
    FLIP_DICT,  # LDRAW_TOKENS, META_TOKENS, SPECIAL_TOKENS, # REMOVED from public API
    LDR_ALL_COLOUR,
    LDR_ANY_COLOUR,
    LDR_ANY_COLOUR_FILL,
    LDR_BLUBLU_COLOUR,
    LDR_BLUES_COLOUR,
    LDR_BLUGRN_COLOUR,
    LDR_BLUBRN_COLOUR,
    LDR_BLUYLW_COLOUR,
    LDR_BLKWHT_COLOUR,
    LDR_BRGREEN_COLOUR,
    LDR_DEFAULT_ASPECT,
    LDR_DEFAULT_SCALE,
    LDR_DEF_COLOUR,
    LDR_DKREDBLU_COLOUR,
    LDR_GRAY_COLOUR,
    LDR_GREENS_COLOUR,
    LDR_GRNBRN_COLOUR,
    LDR_LAVENDER_COLOUR,
    LDR_LTYLW_COLOUR,
    LDR_MONO_COLOUR,
    LDR_OPT_COLOUR,
    LDR_ORGBRN_COLOUR,
    LDR_ORGYLW_COLOUR,
    LDR_OTHER_COLOUR,
    LDR_PINK_COLOUR,
    LDR_PINKPURP_COLOUR,
    LDR_REDBLUYLW_COLOUR,
    LDR_REDORG_COLOUR,
    LDR_REDORGYLW_COLOUR,
    LDR_REDYLW_COLOUR,
    LDR_TAN_COLOUR,
    LDR_TANBRN_COLOUR,
    LDR_YELLOWS_COLOUR,
)

# From .ldrcolour and .ldrcolourdict
from .ldrcolour import LDRColour, FillColoursFromLDRCode, FillTitlesFromLDRCode
from .ldrcolourdict import (
    BL_TO_LDR_COLOUR,
    LDR_COLOUR_NAME,
    LDR_COLOUR_RGB,
    LDR_COLOUR_TITLE,
    LDR_FILL_CODES,
    LDR_FILL_TITLES,
)

# From .ldrhelpers
from .ldrhelpers import (
    GetCircleSegments,
    LDU2MM,
    MM2LDU,
    clean_file,
    clean_line,
    ldrlist_from_parts,
    ldrstring_from_list,
    mat_str,
    merge_same_parts,
    norm_angle,
    norm_aspect,
    preset_aspect,
    quantize,
    remove_parts_from_list,
    val_units,
    vector_str,
)

# From .ldrprimitives
from .ldrprimitives import LDRAttrib, LDRHeader, LDRLine, LDRPart, LDRQuad, LDRTriangle

# From .ldrshapes
from .ldrshapes import (
    LDRBox,
    LDRCircle,
    LDRCylinder,
    LDRDisc,
    LDRHole,
    LDRPolyWall,
    LDRRect,
)

# From .ldrmodel
# Aliasing imported constants from ldrmodel to avoid name clashes if they were also in constants.py
from .ldrmodel import (
    COMMON_SUBSTITUTIONS as MODEL_COMMON_SUBSTITUTIONS,
    END_TOKENS as MODEL_END_TOKENS,
    EXCEPTION_LIST as MODEL_EXCEPTION_LIST,
    IGNORE_LIST as MODEL_IGNORE_LIST,
    LDRModel,
    START_TOKENS as MODEL_START_TOKENS,
    # get_meta_commands, # Considered internal
    get_parts_from_model,
    get_sha1_hash,
    # key_colour, # Considered internal sort helper
    # key_name, # Considered internal sort helper
    # line_has_all_tokens, # Considered internal / too specific
    # parse_special_tokens, # Considered internal
    # recursive_parse_model, # Considered internal
    sort_parts,
    substitute_part,
    unique_set,
)

# From .ldvrender
from .ldvrender import LDViewRender, camera_distance

# From .ldrarrows
from .ldrarrows import (
    ARROW_MX,
    ARROW_MY,
    ARROW_MZ,
    ARROW_PARTS,
    ARROW_PLI,
    ARROW_PLI_SUFFIX,
    ARROW_PREFIX,
    ARROW_PX,
    ARROW_PY,
    ARROW_PZ,
    ARROW_SUFFIX,
    ArrowContext,
    arrows_for_step,
    norm_angle_arrow,
    remove_offset_parts as remove_arrow_offset_parts,
    value_after_token as arrow_value_after_token,
    vectorize_arrow,
)

# From .ldrpprint
from .ldrpprint import (
    is_hex_colour,
    pprint_coord_str,
    pprint_ldr_colour,
    pprint_line,
    pprint_line0,
    pprint_line1,
    pprint_line2345,
)

# From .ldrawpy (top-level utility module, often named same as package)
from .ldrawpy import brick_name_strip, mesh_to_ldr, xyz_to_ldr

# Define __all__ to specify the public API explicitly
# Sorted alphabetically for easier maintenance.
__all__ = [
    # .constants
    "ASPECT_DICT",
    "FLIP_DICT",
    "LDR_ALL_COLOUR",
    "LDR_ANY_COLOUR",
    "LDR_ANY_COLOUR_FILL",
    "LDR_BLUBLU_COLOUR",
    "LDR_BLUES_COLOUR",
    "LDR_BLUGRN_COLOUR",
    "LDR_BLUBRN_COLOUR",
    "LDR_BLUYLW_COLOUR",
    "LDR_BLKWHT_COLOUR",
    "LDR_BRGREEN_COLOUR",
    "LDR_DEFAULT_ASPECT",
    "LDR_DEFAULT_SCALE",
    "LDR_DEF_COLOUR",
    "LDR_DKREDBLU_COLOUR",
    "LDR_GRAY_COLOUR",
    "LDR_GREENS_COLOUR",
    "LDR_GRNBRN_COLOUR",
    "LDR_LAVENDER_COLOUR",
    "LDR_LTYLW_COLOUR",
    "LDR_MONO_COLOUR",
    "LDR_OPT_COLOUR",
    "LDR_ORGBRN_COLOUR",
    "LDR_ORGYLW_COLOUR",
    "LDR_OTHER_COLOUR",
    "LDR_PINK_COLOUR",
    "LDR_PINKPURP_COLOUR",
    "LDR_REDBLUYLW_COLOUR",
    "LDR_REDORG_COLOUR",
    "LDR_REDORGYLW_COLOUR",
    "LDR_REDYLW_COLOUR",
    "LDR_TAN_COLOUR",
    "LDR_TANBRN_COLOUR",
    "LDR_YELLOWS_COLOUR",
    # .ldrcolour & .ldrcolourdict
    "BL_TO_LDR_COLOUR",
    "FillColoursFromLDRCode",
    "FillTitlesFromLDRCode",
    "LDRColour",
    "LDR_COLOUR_NAME",
    "LDR_COLOUR_RGB",
    "LDR_COLOUR_TITLE",
    "LDR_FILL_CODES",
    "LDR_FILL_TITLES",
    # .ldrhelpers
    "GetCircleSegments",
    "LDU2MM",
    "MM2LDU",
    "clean_file",
    "clean_line",
    "ldrlist_from_parts",
    "ldrstring_from_list",
    "mat_str",
    "merge_same_parts",
    "norm_angle",
    "norm_aspect",
    "preset_aspect",
    "quantize",
    "remove_parts_from_list",
    "val_units",
    "vector_str",
    # .ldrprimitives
    "LDRAttrib",
    "LDRHeader",
    "LDRLine",
    "LDRPart",
    "LDRQuad",
    "LDRTriangle",
    # .ldrshapes
    "LDRBox",
    "LDRCircle",
    "LDRCylinder",
    "LDRDisc",
    "LDRHole",
    "LDRPolyWall",
    "LDRRect",
    # .ldrmodel
    "LDRModel",
    "MODEL_COMMON_SUBSTITUTIONS",
    "MODEL_END_TOKENS",
    "MODEL_EXCEPTION_LIST",
    "MODEL_IGNORE_LIST",
    "MODEL_START_TOKENS",
    "get_parts_from_model",
    "get_sha1_hash",
    "sort_parts",
    "substitute_part",
    "unique_set",
    # "get_meta_commands", # Removed from __all__
    # "key_colour", # Removed from __all__
    # "key_name", # Removed from __all__
    # "line_has_all_tokens", # Removed from __all__
    # "parse_special_tokens", # Removed from __all__
    # "recursive_parse_model", # Removed from __all__
    # .ldvrender
    "LDViewRender",
    "camera_distance",
    # .ldrarrows
    "ARROW_MX",
    "ARROW_MY",
    "ARROW_MZ",
    "ARROW_PARTS",
    "ARROW_PLI",
    "ARROW_PLI_SUFFIX",
    "ARROW_PREFIX",
    "ARROW_PX",
    "ARROW_PY",
    "ARROW_PZ",
    "ARROW_SUFFIX",
    "ArrowContext",
    "arrow_value_after_token",
    "arrows_for_step",
    "norm_angle_arrow",
    "remove_arrow_offset_parts",
    "vectorize_arrow",
    # .ldrpprint
    "is_hex_colour",
    "pprint_coord_str",
    "pprint_ldr_colour",
    "pprint_line",
    "pprint_line0",
    "pprint_line1",
    "pprint_line2345",
    # .ldrawpy (main module utilities)
    "brick_name_strip",
    "mesh_to_ldr",
    "xyz_to_ldr",
    # Package version info
    "__project__",
    "__version__",
    "VERSION",
]

# Clean up os import if it's truly not needed by any conditional logic within __init__
# import os # This line was commented out in the source, keeping it that way.
# If os was truly needed, it should be uncommented. For now, assume it's not.
# If it was used by a deleted line, `del os` is not necessary. If imported and not used, `del os` is fine.
# Since it's commented out in the source, no action needed for `os` here.
