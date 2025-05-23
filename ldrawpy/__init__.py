"""ldrawpy - A utility package for creating, modifying, and reading LDraw files and data structures."""

import os

# fmt: off
__project__ = 'ldrawpyV2' # Updated project name
# Version for V2, alpha. Ensure no inline comments here that parsing might pick up.
__version__ = '2.0.0a1'
# fmt: on

VERSION = __project__ + "-" + __version__

# Selective and organized imports for the package's public API

# From .constants
from .constants import (
    LDR_OPT_COLOUR, LDR_DEF_COLOUR, LDR_ALL_COLOUR, LDR_ANY_COLOUR, LDR_OTHER_COLOUR,
    LDR_MONO_COLOUR, LDR_BLKWHT_COLOUR, LDR_GRAY_COLOUR, LDR_REDYLW_COLOUR,
    LDR_BLUYLW_COLOUR, LDR_REDBLUYLW_COLOUR, LDR_GRNBRN_COLOUR, LDR_BLUBRN_COLOUR,
    LDR_BRGREEN_COLOUR, LDR_LAVENDER_COLOUR, LDR_PINK_COLOUR, LDR_LTYLW_COLOUR,
    LDR_BLUBLU_COLOUR, LDR_DKREDBLU_COLOUR, LDR_ORGYLW_COLOUR, LDR_ORGBRN_COLOUR,
    LDR_BLUES_COLOUR, LDR_GREENS_COLOUR, LDR_YELLOWS_COLOUR, LDR_REDORG_COLOUR,
    LDR_TANBRN_COLOUR, LDR_REDORGYLW_COLOUR, LDR_BLUGRN_COLOUR, LDR_TAN_COLOUR,
    LDR_PINKPURP_COLOUR, LDR_ANY_COLOUR_FILL, LDRAW_TOKENS, META_TOKENS, SPECIAL_TOKENS,
    LDR_DEFAULT_SCALE, LDR_DEFAULT_ASPECT, ASPECT_DICT, FLIP_DICT
)

# From .ldrcolour and .ldrcolourdict
from .ldrcolour import ( # Import LDRColour and the functions from here
    LDRColour,
    FillColoursFromLDRCode, # Moved from ldrcolourdict
    FillTitlesFromLDRCode   # Moved from ldrcolourdict
)
from .ldrcolourdict import ( # ldrcolourdict only contains dictionaries
    BL_TO_LDR_COLOUR, LDR_COLOUR_NAME, LDR_COLOUR_RGB, LDR_COLOUR_TITLE,
    LDR_FILL_CODES, LDR_FILL_TITLES
)

# From .ldrhelpers
from .ldrhelpers import (
    quantize, MM2LDU, LDU2MM, val_units, mat_str, vector_str,
    get_circle_segments, ldrlist_from_parts, ldrstring_from_list,
    merge_same_parts, remove_parts_from_list, norm_angle,
    norm_aspect, preset_aspect, clean_line, clean_file,
)

# From .ldrprimitives
from .ldrprimitives import (
    LDRAttrib, LDRHeader, LDRLine, LDRTriangle, LDRQuad, LDRPart,
)

# From .ldrshapes
from .ldrshapes import (
    LDRPolyWall, LDRRect, LDRCircle, LDRDisc, LDRHole, LDRCylinder, LDRBox,
)

# From .ldrmodel
from .ldrmodel import (
    LDRModel, substitute_part, line_has_all_tokens, parse_special_tokens,
    get_meta_commands, get_parts_from_model, recursive_parse_model,
    unique_set, key_name, key_colour, get_sha1_hash, sort_parts,
    MODEL_START_TOKENS,
    MODEL_END_TOKENS,
    MODEL_EXCEPTION_LIST,
    MODEL_IGNORE_LIST,
    MODEL_COMMON_SUBSTITUTIONS,
)

# From .ldvrender
from .ldvrender import LDViewRender, camera_distance

# From .ldrarrows
from .ldrarrows import (
    ArrowContext, arrows_for_step, remove_arrow_offset_parts,
    ARROW_PREFIX, ARROW_PLI, ARROW_SUFFIX, ARROW_PLI_SUFFIX, ARROW_PARTS,
    ARROW_MZ, ARROW_PZ, ARROW_MX, ARROW_PX, ARROW_MY, ARROW_PY,
    arrow_value_after_token,
    norm_angle_arrow, vectorize_arrow,
)

# From .ldrpprint
from .ldrpprint import (
    pprint_ldr_colour, pprint_coord_str, pprint_line1,
    pprint_line2345, pprint_line0, pprint_line, is_hex_colour
)

# From .ldrawpy (top-level module, often named same as package)
from .ldrawpy import (
    brick_name_strip, xyz_to_ldr, mesh_to_ldr,
)

# Define __all__ to specify the public API explicitly
__all__ = [
    # Constants
    "LDR_DEF_COLOUR", "LDR_OPT_COLOUR", "SPECIAL_TOKENS", # Add more as needed
    # ldrcolour & ldrcolourdict
    "LDRColour", "LDR_COLOUR_NAME", "LDR_COLOUR_RGB",
    "FillColoursFromLDRCode", "FillTitlesFromLDRCode", # Now correctly sourced
    "BL_TO_LDR_COLOUR", "LDR_COLOUR_TITLE", "LDR_FILL_CODES", "LDR_FILL_TITLES",
    # ldrhelpers
    "quantize", "clean_line", "clean_file", "vector_str", "mat_str", "norm_aspect",
    "ldrlist_from_parts", "ldrstring_from_list", "merge_same_parts", "remove_parts_from_list",
    "MM2LDU", "LDU2MM", "val_units", "get_circle_segments", "norm_angle", "preset_aspect",
    # ldrprimitives
    "LDRAttrib", "LDRHeader", "LDRLine", "LDRTriangle", "LDRQuad", "LDRPart",
    # ldrshapes
    "LDRBox", "LDRCircle", "LDRCylinder", "LDRDisc", "LDRHole", "LDRPolyWall", "LDRRect",
    # ldrmodel
    "LDRModel", "substitute_part", "get_parts_from_model", "sort_parts", "get_sha1_hash",
    "MODEL_COMMON_SUBSTITUTIONS", "line_has_all_tokens", "parse_special_tokens",
    "get_meta_commands", "recursive_parse_model", "unique_set", "key_name", "key_colour",
    "MODEL_START_TOKENS", "MODEL_END_TOKENS", "MODEL_EXCEPTION_LIST", "MODEL_IGNORE_LIST",
    # ldvrender
    "LDViewRender", "camera_distance",
    # ldrarrows
    "ArrowContext", "arrows_for_step", "remove_arrow_offset_parts", "ARROW_PARTS",
    "ARROW_PREFIX", "ARROW_PLI", "ARROW_SUFFIX", "ARROW_PLI_SUFFIX",
    "ARROW_MZ", "ARROW_PZ", "ARROW_MX", "ARROW_PX", "ARROW_MY", "ARROW_PY",
    "arrow_value_after_token", "norm_angle_arrow", "vectorize_arrow",
    # ldrpprint
    "pprint_line", "is_hex_colour", "pprint_ldr_colour", "pprint_coord_str",
    "pprint_line1", "pprint_line2345", "pprint_line0",
    # ldrawpy (main module)
    "brick_name_strip", "xyz_to_ldr", "mesh_to_ldr",
    # Version info
    "__project__", "__version__", "VERSION",
]
