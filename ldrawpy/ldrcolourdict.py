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
# Colour lookup dictionaries

from .constants import (
    LDR_ALL_COLOUR,
    LDR_ANY_COLOUR,
    LDR_OTHER_COLOUR,
    LDR_MONO_COLOUR,
    LDR_BLUES_COLOUR,
    LDR_GREENS_COLOUR,
    LDR_YELLOWS_COLOUR,
    LDR_PINKPURP_COLOUR,
    LDR_BLKWHT_COLOUR,
    LDR_GRAY_COLOUR,
    LDR_REDYLW_COLOUR,
    LDR_BLUYLW_COLOUR,
    LDR_REDBLUYLW_COLOUR,
    LDR_GRNBRN_COLOUR,
    LDR_BLUBRN_COLOUR,
    LDR_BRGREEN_COLOUR,
    LDR_LAVENDER_COLOUR,
    LDR_PINK_COLOUR,
    LDR_LTYLW_COLOUR,
    LDR_BLUBLU_COLOUR,
    LDR_ORGYLW_COLOUR,
    LDR_ORGBRN_COLOUR,
    LDR_REDORG_COLOUR,
    LDR_REDORGYLW_COLOUR,
    LDR_BLUGRN_COLOUR,
    LDR_TAN_COLOUR,
    LDR_ANY_COLOUR_FILL,
    LDR_TANBRN_COLOUR,
    LDR_DKREDBLU_COLOUR,
)

BL_TO_LDR_COLOUR = {
    1: 15,
    49: 503,
    99: 151,
    86: 71,
    9: 7,
    10: 8,
    85: 72,
    11: 0,
    59: 320,
    5: 4,
    27: 216,
    25: 12,
    26: 100,
    58: 335,
    88: 70,
    8: 6,
    120: 308,
    69: 28,
    2: 19,
    90: 78,
    28: 92,
    150: 84,
    91: 86,
    106: 450,
    29: 366,
    68: 484,
    4: 25,
    31: 462,
    110: 191,
    32: 125,
    96: 68,
    3: 14,
    103: 226,
    33: 18,
    35: 120,
    158: 326,
    34: 27,
    155: 330,
    80: 288,
    6: 2,
    36: 10,
    37: 74,
    38: 17,
    48: 378,
    39: 3,
    40: 11,
    41: 118,
    152: 323,
    63: 272,
    7: 1,
    153: 321,
    156: 322,
    42: 73,
    72: 313,
    105: 212,
    62: 9,
    87: 232,
    55: 379,
    97: 89,
    109: 23,
    43: 110,
    73: 112,
    44: 20,
    24: 22,
    157: 30,
    154: 31,
    54: 373,
    71: 26,
    94: 351,
    104: 29,
    23: 13,
    56: 77,
    12: 47,
    13: 40,
    17: 36,
    18: 38,
    98: 57,
    164: 231,
    121: 54,
    19: 46,
    16: 42,
    108: 35,
    20: 34,
    14: 33,
    74: 41,
    15: 43,
    113: 39,
    51: 52,
    50: 37,
    107: 45,
    21: 334,
    22: 383,
    57: 60,
    122: 64,
    52: 61,
    64: 62,
    82: 63,
    83: 183,
    119: 150,
    66: 135,
    95: 179,
    77: 148,
    78: 137,
    61: 142,
    115: 297,
    81: 178,
    84: 134,
    67: 80,
    70: 81,
    65: 82,
    89: 85,
    46: 21,
    353: 220,
    368: 236,
}

LDR_COLOUR_NAME = {
    0: "Black",
    1: "Blue",
    2: "Green",
    3: "Dark Turquoise",
    4: "Red",
    5: "Dark Pink",
    6: "Brown",
    7: "Old Light Grey",
    8: "Old Dark Grey",
    9: "Light Blue",
    10: "Bright Green",
    11: "Light Turquoise",
    12: "Salmon",
    13: "Pink",
    14: "Yellow",
    15: "White",
    16: "Default",
    17: "Light Green",
    18: "Light Yellow",
    19: "Tan",
    20: "Light Violet",
    21: "Glow in Dark Opaque",
    22: "Purple",
    23: "Dark Blue Violet",
    24: "Outline",
    25: "Orange",
    26: "Magenta",
    27: "Lime",
    28: "Dark Tan",
    29: "Bright Pink",
    30: "Medium Lavender",
    31: "Lavender",
    68: "Very Light Orange",
    69: "Bright Reddish Lilac",
    70: "Reddish Brown",
    71: "Light Bluish Gray",
    72: "Dark Bluish Gray",
    73: "Medium Blue",
    74: "Medium Green",
    77: "Light Pink",
    78: "Light Nougat",
    84: "Medium Nougat",
    85: "Dark Purple",
    86: "Dark Flesh",
    89: "Blue Violet",
    92: "Nougat",
    100: "Light Salmon",
    110: "Violet",
    112: "Medium Violet",
    115: "Medium Lime",
    118: "Aqua",
    120: "Light Lime",
    125: "Light Orange",
    151: "Very Light Bluish Grey",
    191: "Bright Light Orange",
    212: "Bright Light Blue",
    216: "Rust",
    226: "Bright Light Yellow",
    232: "Sky Blue",
    272: "Dark Blue",
    288: "Dark Green",
    308: "Dark Brown",
    313: "Maersk Blue",
    320: "Dark Red",
    321: "Dark Azur",
    322: "Medium Azur",
    323: "Light Aqua",
    326: "Yellowish Green",
    330: "Olive Green",
    335: "Sand Red",
    351: "Medium Dark Pink",
    353: "Coral",
    366: "Earth Orange",
    373: "Sand Purple",
    378: "Sand Green",
    379: "Sand Blue",
    450: "Fabuland Brown",
    462: "Medium Orange",
    484: "Dark Orange",
    503: "Very Light Grey",
    218: "Reddish Lilac",
    295: "Flamingo Pink",
    219: "Lilac",
    128: "Dark Nougat",
    47: "Trans Clear",
    40: "Trans Black",
    36: "Trans Red",
    38: "Trans Neon Orange",
    57: "Trans Orange",
    54: "Trans Neon Yellow",
    46: "Trans Yellow",
    42: "Trans Neon Green",
    35: "Trans Bright Green",
    34: "Trans Green",
    33: "Trans Dark Blue",
    41: "Trans Medium Blue",
    43: "Trans Light Blue",
    39: "Trans Very Light Blue",
    44: "Trans Bright Reddish Lilac",
    52: "Trans Purple",
    37: "Trans Dark Pink",
    45: "Trans Pink",
    285: "Trans Light Green",
    234: "Trans Fire Yellow",
    293: "Trans Light Blue Violet",
    231: "Trans Bright Light Orange",
    284: "Trans Reddish Lilac",
    334: "Chrome Gold",
    383: "Chrome Silver",
    60: "Chrome Antique Brass",
    64: "Chrome Black",
    61: "Chrome Blue",
    62: "Chrome Green",
    63: "Chrome Pink",
    183: "Pearl White",
    150: "Pearl Very Light Grey",
    135: "Pearl Light Grey",
    179: "Flat Silver",
    148: "Pearl Dark Grey",
    137: "Metal Blue",
    142: "Pearl Light Gold",
    297: "Pearl Gold",
    178: "Flat Dark Gold",
    134: "Copper",
    189: "Reddish Gold",
    80: "Metallic Silver",
    81: "Metallic Green",
    82: "Metallic Gold",
    83: "Metallic Black",
    87: "Metallic Dark Grey",
    300: "Metallic Copper",
    184: "Metallic Bright Red",
    186: "Metallic Dark Green",
    368: "Neon Yellow",
    801: "Arrow Blue",
    802: "Arrow Green",
    804: "Arrow Red",
}

LDR_COLOUR_RGB = {
    0: "05131D",  # 'Black',
    1: "0055bf",  #'Blue',
    2: "257a3e",  #'Green',
    3: "00838f",  #'Dark Turquoise',
    4: "c91a09",  #'Red',
    5: "c870a0",  #'Dark Pink',
    6: "583927",  #'Brown',
    7: "9ba19d",  #'Light Grey',
    8: "6d6e5c",  #'Dark Grey',
    9: "b4d2e3",  #'Light Blue',
    10: "4b9f4a",  #'Bright Green',
    11: "55a5af",  #'Light Turquoise',
    12: "f2705e",  #'Salmon',
    13: "fc97ac",  #'Pink',
    14: "f2cd37",  #'Yellow',
    15: "ffffff",  #'White',
    16: "101010",  #'Default',
    17: "c2dab8",  #'Light Green',
    18: "fbe696",  #'Light Yellow',
    19: "e4cd9e",  #'Tan',
    20: "c9cae2",  #'Light Violet',
    21: "ECE8DE",  #'Glow in Dark Opaque'
    22: "81007b",  #'Purple',
    23: "2032b0",  #'Dark Blue Violet',
    24: "101010",  #'Outline',
    25: "fe8a18",  #'Orange',
    26: "923978",  #'Magenta',
    27: "bbe90b",  #'Lime',
    28: "958a73",  #'Dark Tan',
    29: "e4adc8",  #'Bright Pink',
    30: "ac78ba",  #'Medium Lavender',
    31: "e1d5ed",  #'Lavender',
    68: "f3cf9b",  #'Very Light Orange',
    69: "cd6298",  #'Bright Reddish Lilac',
    70: "582a12",  #'Reddish Brown',
    71: "a0a5a9",  #'Light Bluish Gray',
    72: "6c6e68",  #'Dark Bluish Gray',
    73: "5c9dd1",  #'Medium Blue',
    74: "73dca1",  #'Medium Green',
    77: "fecccf",  #'Light Pink',
    78: "f6d7b3",  #'Light Flesh',
    84: "cc702a",  #'Medium Dark Flesh',
    85: "3f3691",  #'Medium Lilac', / Dark Purple
    86: "7c503a",  #'Dark Flesh',
    89: "4c61db",  #'Blue Violet',
    92: "d09168",  #'Flesh',
    100: "febabd",  #'Light Salmon',
    110: "4354a3",  #'Violet',
    112: "6874ca",  #'Medium Violet',
    115: "c7d23c",  #'Medium Lime',
    118: "b3d7d1",  #'Aqua',
    120: "d9e4a7",  #'Light Lime',
    125: "f9ba61",  #'Light Orange',
    151: "e6e3e0",  #'Very Light Bluish Grey',
    191: "f8bb3d",  #'Bright Light Orange',
    212: "86c1e1",  #'Bright Light Blue',
    216: "b31004",  #'Rust',
    226: "fff03a",  #'Bright Light Yellow',
    232: "56bed6",  #'Sky Blue',
    272: "0d325b",  #'Dark Blue',
    288: "184632",  #'Dark Green',
    308: "352100",  #'Dark Brown',
    313: "54a9c8",  #'Maersk Blue',
    320: "720e0f",  #'Dark Red',
    321: "1498d7",  #'Dark Azur',
    322: "3ec2dd",  #'Medium Azur',
    323: "bddcd8",  #'Light Aqua',
    326: "dfeea5",  #'Yellowish Green',
    330: "9b9a5a",  #'Olive Green',
    335: "d67572",  #'Sand Red',
    351: "f785b1",  #'Medium Dark Pink',
    353: "FF6D77",  # Coral
    366: "fa9c1c",  #'Earth Orange',
    373: "845e84",  #'Sand Purple',
    378: "a0bcac",  #'Sand Green',
    379: "597184",  #'Sand Blue',
    450: "b67b50",  #'Fabuland Brown',
    462: "ffa70b",  #'Medium Orange',
    484: "a95500",  #'Dark Orange',
    503: "e6e3da",  #'Very Light Grey',
    218: "8e5597",  #'Reddish Lilac',
    295: "ff94c2",  #'Flamingo Pink',
    219: "564e9d",  #'Lilac',
    128: "ad6140",  #'Dark Nougat',
    47: "fcfcfc",  #'Trans Clear',
    40: "635f52",  #'Trans Black',
    36: "c91a09",  #'Trans Red',
    38: "ff800d",  #'Trans Neon Orange',
    57: "f08f1c",  #'Trans Orange',
    54: "dab000",  #'Trans Neon Yellow',
    46: "f5cd2f",  #'Trans Yellow',
    42: "c0ff00",  #'Trans Neon Green',
    35: "56e646",  #'Trans Bright Green',
    34: "237841",  #'Trans Green',
    33: "0020a0",  #'Trans Dark Blue',
    41: "559ab7",  #'Trans Medium Blue',
    43: "aee9ef",  #'Trans Light Blue',
    39: "c1dff0",  #'Trans Very Light Blue',
    44: "96709f",  #'Trans Bright Reddish Lilac',
    52: "a5a5cb",  #'Trans Purple',
    37: "df6695",  #'Trans Dark Pink',
    45: "fc97ac",  #'Trans Pink',
    285: "7dc291",  #'Trans Light Green',
    234: "fbe890",  #'Trans Fire Yellow',
    293: "68abe4",  #'Trans Light Blue Violet',
    231: "fcb76d",  #'Trans Bright Light Orange',
    284: "c281a5",  #'Trans Reddish Lilac',
    334: "bba53d",  #'Chrome Gold',
    383: "e0e0e0",  #'Chrome Silver',
    60: "645a4c",  #'Chrome Antique Brass',
    64: "1b2a34",  #'Chrome Black',
    61: "6c96bf",  #'Chrome Blue',
    62: "3cb371",  #'Chrome Green',
    63: "aa4d8e",  #'Chrome Pink',
    183: "f2f3f2",  #'Pearl White',
    150: "bbbdbc",  #'Pearl Very Light Grey',
    135: "9ca3a8",  #'Pearl Light Grey',
    179: "898788",  #'Flat Silver',
    148: "575857",  #'Pearl Dark Grey',
    137: "5677ba",  #'Metal Blue',
    142: "dcbe61",  #'Pearl Light Gold',
    297: "cc9c2b",  #'Pearl Gold',
    178: "b4883e",  #'Flat Dark Gold',
    134: "964a27",  #'Copper',
    189: "ac8247",  #'Reddish Gold',
    80: "a5a9b4",  #'Metallic Silver',
    81: "899b5f",  #'Metallic Green',
    82: "dbac34",  #'Metallic Gold',
    83: "1a2831",  #'Metallic Black',
    87: "6d6e5c",  #'Metallic Dark Grey',
    300: "c27f53",  #'Metallic Copper',
    184: "d60026",  #'Metallic Bright Red',
    186: "008e3c",  #'Metallic Dark Green'
    368: "EBD800",  # Neon Yellow
    801: "0830FF",  # Arrow Blue
    802: "08B010",  # Arrow Green
    804: "FF0000",  # Arrow Red
}

LDR_COLOUR_TITLE = {
    LDR_ALL_COLOUR: "All Colours",
    LDR_ANY_COLOUR: "Any Colour",
    LDR_OTHER_COLOUR: "Other Colours",
    LDR_MONO_COLOUR: "Monochrome",
    LDR_BLUES_COLOUR: "Blues",
    LDR_GREENS_COLOUR: "Greens",
    LDR_YELLOWS_COLOUR: "Yellows",
    LDR_PINKPURP_COLOUR: "Pinks/Purples",
    LDR_BLKWHT_COLOUR: "Black/White",
    LDR_GRAY_COLOUR: "Light/Dark Gray",
    LDR_REDYLW_COLOUR: "Red/Yellow",
    LDR_BLUYLW_COLOUR: "Blue/Yellow",
    LDR_REDBLUYLW_COLOUR: "Red/Blue/Yellow",
    LDR_GRNBRN_COLOUR: "Green/Brown",
    LDR_BLUBRN_COLOUR: "Blue/Brown",
    LDR_BRGREEN_COLOUR: "Bright Green/Ylwish Green",
    LDR_LAVENDER_COLOUR: "Light/Med Lavender",
    LDR_PINK_COLOUR: "Light/Med Pink",
    LDR_LTYLW_COLOUR: "Br Light Yellow/Light Nougat",
    LDR_BLUBLU_COLOUR: "Blue/Med Blue",
    LDR_ORGYLW_COLOUR: "Orange/Yellow",
    LDR_ORGBRN_COLOUR: "Orange/Brown",
    LDR_REDORG_COLOUR: "Red/Orange",
    LDR_REDORGYLW_COLOUR: "Red/Org/Ylw",
    LDR_BLUGRN_COLOUR: "Blue/Green",
    LDR_TAN_COLOUR: "Light/Dark Tan",
}

LDR_FILL_CODES = {
    LDR_ALL_COLOUR: LDR_ANY_COLOUR_FILL,
    LDR_ANY_COLOUR: LDR_ANY_COLOUR_FILL,
    LDR_OTHER_COLOUR: LDR_ANY_COLOUR_FILL,
    LDR_MONO_COLOUR: [
        LDR_COLOUR_RGB[15],
        LDR_COLOUR_RGB[151],
        LDR_COLOUR_RGB[71],
        LDR_COLOUR_RGB[72],
    ],
    LDR_BLUES_COLOUR: [
        LDR_COLOUR_RGB[212],
        LDR_COLOUR_RGB[73],
        LDR_COLOUR_RGB[1],
        LDR_COLOUR_RGB[272],
    ],
    LDR_GREENS_COLOUR: [
        LDR_COLOUR_RGB[378],
        LDR_COLOUR_RGB[10],
        LDR_COLOUR_RGB[2],
        LDR_COLOUR_RGB[288],
    ],
    LDR_YELLOWS_COLOUR: [
        LDR_COLOUR_RGB[19],
        LDR_COLOUR_RGB[226],
        LDR_COLOUR_RGB[14],
        LDR_COLOUR_RGB[191],
    ],
    LDR_PINKPURP_COLOUR: [
        LDR_COLOUR_RGB[29],
        LDR_COLOUR_RGB[13],
        LDR_COLOUR_RGB[30],
        LDR_COLOUR_RGB[26],
    ],
    LDR_BLKWHT_COLOUR: [LDR_COLOUR_RGB[0], LDR_COLOUR_RGB[15]],
    LDR_GRAY_COLOUR: [LDR_COLOUR_RGB[71], LDR_COLOUR_RGB[72]],
    LDR_REDYLW_COLOUR: [LDR_COLOUR_RGB[4], LDR_COLOUR_RGB[14]],
    LDR_BLUYLW_COLOUR: [LDR_COLOUR_RGB[1], LDR_COLOUR_RGB[14]],
    LDR_REDBLUYLW_COLOUR: [LDR_COLOUR_RGB[4], LDR_COLOUR_RGB[1], LDR_COLOUR_RGB[14]],
    LDR_GRNBRN_COLOUR: [LDR_COLOUR_RGB[2], LDR_COLOUR_RGB[70]],
    LDR_BLUBRN_COLOUR: [LDR_COLOUR_RGB[1], LDR_COLOUR_RGB[70]],
    LDR_BLUBLU_COLOUR: [LDR_COLOUR_RGB[1], LDR_COLOUR_RGB[73]],
    LDR_DKREDBLU_COLOUR: [LDR_COLOUR_RGB[272], LDR_COLOUR_RGB[320]],
    LDR_BRGREEN_COLOUR: [LDR_COLOUR_RGB[10], LDR_COLOUR_RGB[326]],
    LDR_LAVENDER_COLOUR: [LDR_COLOUR_RGB[30], LDR_COLOUR_RGB[31]],
    LDR_PINK_COLOUR: [LDR_COLOUR_RGB[13], LDR_COLOUR_RGB[29]],
    LDR_LTYLW_COLOUR: [LDR_COLOUR_RGB[226], LDR_COLOUR_RGB[78]],
    LDR_ORGYLW_COLOUR: [LDR_COLOUR_RGB[25], LDR_COLOUR_RGB[14]],
    LDR_ORGBRN_COLOUR: [LDR_COLOUR_RGB[25], LDR_COLOUR_RGB[6]],
    LDR_REDORG_COLOUR: [LDR_COLOUR_RGB[4], LDR_COLOUR_RGB[25]],
    LDR_TANBRN_COLOUR: [LDR_COLOUR_RGB[19], LDR_COLOUR_RGB[6]],
    LDR_REDORGYLW_COLOUR: [LDR_COLOUR_RGB[4], LDR_COLOUR_RGB[25], LDR_COLOUR_RGB[14]],
    LDR_BLUGRN_COLOUR: [LDR_COLOUR_RGB[73], LDR_COLOUR_RGB[10]],
    LDR_TAN_COLOUR: [LDR_COLOUR_RGB[19], LDR_COLOUR_RGB[28]],
}

LDR_FILL_TITLES = {
    LDR_ALL_COLOUR: ["All Colours"],
    LDR_ANY_COLOUR: ["Any Colour"],
    LDR_OTHER_COLOUR: ["Other Colours"],
    LDR_MONO_COLOUR: ["Monochrome"],
    LDR_BLUES_COLOUR: ["Blues"],
    LDR_GREENS_COLOUR: ["Greens"],
    LDR_YELLOWS_COLOUR: ["Yellows"],
    LDR_PINKPURP_COLOUR: ["Pinks/Purples"],
    LDR_BLKWHT_COLOUR: [LDR_COLOUR_NAME[0], LDR_COLOUR_NAME[15]],
    LDR_GRAY_COLOUR: [LDR_COLOUR_NAME[71], LDR_COLOUR_NAME[72]],
    LDR_REDYLW_COLOUR: [LDR_COLOUR_NAME[4], LDR_COLOUR_NAME[14]],
    LDR_BLUYLW_COLOUR: [LDR_COLOUR_NAME[1], LDR_COLOUR_NAME[14]],
    LDR_REDBLUYLW_COLOUR: [LDR_COLOUR_NAME[4], LDR_COLOUR_NAME[1], LDR_COLOUR_NAME[14]],
    LDR_GRNBRN_COLOUR: [LDR_COLOUR_NAME[2], LDR_COLOUR_NAME[70]],
    LDR_BLUBRN_COLOUR: [LDR_COLOUR_NAME[1], LDR_COLOUR_NAME[70]],
    LDR_BLUBLU_COLOUR: [LDR_COLOUR_NAME[1], LDR_COLOUR_NAME[73]],
    LDR_DKREDBLU_COLOUR: [LDR_COLOUR_NAME[272], LDR_COLOUR_NAME[320]],
    LDR_BRGREEN_COLOUR: [LDR_COLOUR_NAME[10], LDR_COLOUR_NAME[326]],
    LDR_LAVENDER_COLOUR: [LDR_COLOUR_NAME[30], LDR_COLOUR_NAME[31]],
    LDR_PINK_COLOUR: [LDR_COLOUR_NAME[13], LDR_COLOUR_NAME[29]],
    LDR_LTYLW_COLOUR: [LDR_COLOUR_NAME[226], LDR_COLOUR_NAME[78]],
    LDR_ORGYLW_COLOUR: [LDR_COLOUR_NAME[25], LDR_COLOUR_NAME[14]],
    LDR_ORGBRN_COLOUR: [LDR_COLOUR_NAME[25], LDR_COLOUR_NAME[6]],
    LDR_REDORG_COLOUR: [LDR_COLOUR_NAME[4], LDR_COLOUR_NAME[25]],
    LDR_TANBRN_COLOUR: [LDR_COLOUR_NAME[19], LDR_COLOUR_NAME[6]],
    LDR_REDORGYLW_COLOUR: [
        LDR_COLOUR_NAME[4],
        LDR_COLOUR_NAME[25],
        LDR_COLOUR_NAME[14],
    ],
    LDR_BLUGRN_COLOUR: [LDR_COLOUR_NAME[1], LDR_COLOUR_NAME[2]],
    LDR_TAN_COLOUR: [LDR_COLOUR_NAME[19], LDR_COLOUR_NAME[28]],
}
