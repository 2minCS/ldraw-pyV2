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
# LDView render class and helper functions

import os
import tempfile
import datetime  # Keep this import
import subprocess
import shlex

# from datetime import datetime # REMOVED THIS LINE - Caused redefinition
from collections import defaultdict
from PIL import Image, ImageOps, ImageChops, ImageFilter, ImageEnhance

# Explicit imports from toolbox
# Note: 'crayons' was mentioned in comments but not explicitly imported or used in the simplified _coord_str.
# If 'crayons' is needed, it should be added as a dependency and imported.
from toolbox import (
    apply_params,
    logmsg,
    split_path,
    full_path,
    colour_path_str,  # This function uses 'crayons' in toolbox, ensure 'crayons' is a dependency if used.
    Vector,
)

# from ldrawpy import * # This will be refactored later with specific imports

LDVIEW_BIN = (
    "/Applications/LDView.app/Contents/MacOS/LDView"  # Needs to be configurable
)
LDVIEW_DICT = {
    "DefaultMatrix": "1,0,0,0,1,0,0,0,1",
    "SnapshotSuffix": ".png",
    "BlackHighlights": 0,
    "ProcessLDConfig": 1,
    "EdgeThickness": 3,
    "EdgesOnly": 0,
    "ShowHighlightLines": 1,
    "ConditionalHighlights": 1,
    "SaveZoomToFit": 0,
    "SubduedLighting": 1,
    "UseSpecular": 1,
    "UseFlatShading": 0,
    "LightVector": "0,1,1",
    "AllowPrimitiveSubstitution": 0,
    "HiResPrimitives": 1,
    "UseQualityLighting": 0,
    "ShowAxes": 0,
    "UseQualityStuds": 1,
    "TextureStuds": 0,
    "SaveActualSize": 0,
    "SaveAlpha": 1,
    "AutoCrop": 0,
    "LineSmoothing": 1,
    "Texmaps": 1,
    "MemoryUsage": 2,
    "MultiThreaded": 1,
}
# 10.0 / tan(0.005 deg)
LDU_DISTANCE = 114591


def camera_distance(
    scale: float = 1.0, dpi: int = 300, page_width: float = 8.5
) -> float:
    one = 20 * 1 / 64 * dpi * scale
    sz = page_width * dpi / one * LDU_DISTANCE * 0.775
    sz *= 1700 / 1000
    return sz


def _coord_str(x, y=None, sep: str = ", ") -> str:
    # Simplified version, does not use crayons.
    # If crayons is intended, it needs to be a dependency and imported.
    # import crayons
    if isinstance(x, (tuple, list)):
        a, b = float(x[0]), float(x[1])
    else:
        a, b = float(x), float(y)  # type: ignore
    sa = ("%f" % (a)).rstrip("0").rstrip(".")
    sb = ("%f" % (b)).rstrip("0").rstrip(".")
    # Original used crayons for color, e.g. str(crayons.yellow(f"{sa}"))
    return f"{sa}{sep}{sb}"


class LDViewRender:
    """LDView render session helper class."""

    # Class attributes with type hints for clarity
    dpi: int
    page_width: float
    page_height: float
    auto_crop: bool
    image_smooth: bool
    no_lines: bool
    wireframe: bool
    quality_lighting: bool
    flat_shading: bool
    specular: bool
    line_thickness: int
    texmaps: bool
    scale: float
    output_path: str | None
    log_output: bool
    log_level: int
    overwrite: bool

    # Instance attributes that will be initialized
    ldr_temp_path: str
    pix_width: int
    pix_height: int
    args_size: str
    cam_dist: int
    args_cam: str
    settings_snapshot: dict | None

    PARAMS = {  # Kept for apply_params compatibility
        "dpi": 300,
        "page_width": 8.5,
        "page_height": 11.0,
        "auto_crop": True,
        "image_smooth": False,
        "no_lines": False,
        "wireframe": False,
        "quality_lighting": True,
        "flat_shading": False,
        "specular": True,
        "line_thickness": 3,
        "texmaps": True,
        "scale": 1.0,
        "output_path": None,
        "log_output": True,
        "log_level": 0,
        "overwrite": False,
    }

    def __init__(self, **kwargs):
        # Initialize attributes with defaults from PARAMS or type defaults
        for param, default_value in self.PARAMS.items():
            setattr(self, param, default_value)

        self.ldr_temp_path = os.path.join(
            tempfile.gettempdir(), "ldrawpy_temp.ldr"
        )  # Unique temp file name
        apply_params(self, kwargs)  # Overrides defaults with kwargs

        self.set_page_size(self.page_width, self.page_height)
        self.set_scale(self.scale)
        self.settings_snapshot = None

    def __str__(self) -> str:
        s = []
        s.append("LDViewRender: ")
        s.append(f" DPI: {self.dpi}  Scale: {self.scale:.2f}")
        s.append(
            f" Page size: {_coord_str(self.page_width, self.page_height, ' x ')} in "
            f"({_coord_str(self.pix_width, self.pix_height, ' x ')}) pixels"
        )
        s.append(f" Auto crop: {self.auto_crop}  Image smooth: {self.image_smooth}")
        s.append(f" Camera distance: {self.cam_dist}")
        return "\n".join(s)

    def snapshot_settings(self):
        self.settings_snapshot = {}
        keys_to_snapshot = [
            "page_width",
            "page_height",
            "auto_crop",
            "image_smooth",
            "no_lines",
            "wireframe",
            "quality_lighting",
            "scale",
            "log_output",
            "log_level",
            "overwrite",
            "texmaps",
            "flat_shading",
            "specular",
            "line_thickness",
            "dpi",  # Added missing ones
        ]
        for key in keys_to_snapshot:
            if hasattr(self, key):
                self.settings_snapshot[key] = getattr(self, key)

    def restore_settings(self):
        if self.settings_snapshot is None:
            return
        keys_to_restore = [
            "page_width",
            "page_height",
            "auto_crop",
            "image_smooth",
            "no_lines",
            "wireframe",
            "quality_lighting",
            "scale",
            "log_output",
            "log_level",
            "overwrite",
            "texmaps",
            "flat_shading",
            "specular",
            "line_thickness",
            "dpi",  # Added missing ones
        ]
        for key in keys_to_restore:
            if key in self.settings_snapshot:
                setattr(self, key, self.settings_snapshot[key])
        # After restoring, re-calculate dependent attributes
        self.set_page_size(self.page_width, self.page_height)
        self.set_scale(self.scale)

    def set_page_size(self, width: float, height: float):
        self.page_width = width
        self.page_height = height
        self.pix_width = int(self.page_width * self.dpi)
        self.pix_height = int(self.page_height * self.dpi)
        self.args_size = f"-SaveWidth={self.pix_width} -SaveHeight={self.pix_height}"

    def set_dpi(self, dpi: int):
        self.dpi = dpi
        self.set_page_size(
            width=self.page_width, height=self.page_height
        )  # Recalculate pixel sizes
        self.set_scale(scale=self.scale)  # Recalculate camera distance

    def set_scale(self, scale: float):
        self.scale = scale
        self.cam_dist = int(camera_distance(self.scale, self.dpi, self.page_width))
        self.args_cam = f"-ca0.01 -cg0.0,0.0,{self.cam_dist}"

    def _logoutput(self, msg: str, tstart: datetime.datetime = None, level: int = 2):
        logmsg(msg, tstart=tstart, level=level, prefix="LDR", log_level=self.log_level)

    def render_from_str(self, ldrstr: str, outfile: str):
        """Render from a LDraw text string."""
        if self.log_output:
            s = ldrstr.splitlines()[0] if ldrstr.splitlines() else ""
            # Assuming colour_path_str and crayons are handled if this colored output is desired
            self._logoutput(f"rendering string ({s[:min(len(s), 80)]})...")
        try:
            with open(self.ldr_temp_path, "w", encoding="utf-8") as f:
                f.write(ldrstr)
            self.render_from_file(self.ldr_temp_path, outfile)
        except IOError as e:
            self._logoutput(f"Error writing temporary LDR file: {e}", level=0)

    def render_from_parts(
        self, parts: list, outfile: str
    ):  # Assuming parts is list of LDRPart-like objects
        """Render using a list of LDRPart objects."""
        if self.log_output:
            self._logoutput(f"rendering parts ({len(parts)})...")
        ldrstr_list = [str(p) for p in parts]
        ldrstr = "".join(ldrstr_list)
        self.render_from_str(ldrstr, outfile)

    def render_from_file(self, ldrfile: str, outfile: str):
        """Render from an LDraw file."""
        t_start_render = datetime.datetime.now()

        filename_to_render = outfile
        if self.output_path is not None:
            # Ensure output_path exists
            os.makedirs(self.output_path, exist_ok=True)

            # Check if outfile is an absolute path or already includes output_path
            if not os.path.isabs(outfile) and not outfile.startswith(self.output_path):
                filename_to_render = os.path.join(
                    self.output_path, os.path.basename(outfile)
                )
            # else: outfile is absolute or already correctly prefixed
        else:
            filename_to_render = full_path(outfile)  # Resolves to absolute path

        # Ensure the directory for the output file exists
        output_dir = os.path.dirname(filename_to_render)
        if output_dir:  # Create if not root directory
            os.makedirs(output_dir, exist_ok=True)

        if not self.overwrite and os.path.isfile(filename_to_render):
            if self.log_output:
                fno_display = colour_path_str(os.path.basename(filename_to_render))
                self._logoutput(f"rendered file {fno_display} already exists, skipping")
            return

        ldv_command_parts = []
        ldv_command_parts.append(LDVIEW_BIN)
        ldv_command_parts.append(f"-SaveSnapShot={filename_to_render}")
        ldv_command_parts.append(self.args_size)
        ldv_command_parts.append(self.args_cam)

        current_ldview_settings = LDVIEW_DICT.copy()
        current_ldview_settings["EdgeThickness"] = self.line_thickness
        current_ldview_settings["UseQualityLighting"] = (
            1 if self.quality_lighting else 0
        )
        current_ldview_settings["Texmaps"] = 1 if self.texmaps else 0
        current_ldview_settings["UseFlatShading"] = 1 if self.flat_shading else 0
        current_ldview_settings["UseSpecular"] = 1 if self.specular else 0

        if self.no_lines:
            current_ldview_settings["EdgeThickness"] = 0
            current_ldview_settings["ShowHighlightLines"] = 0
            current_ldview_settings["ConditionalHighlights"] = 0
            current_ldview_settings["UseQualityStuds"] = (
                0  # Also often disabled with no_lines
            )
        if self.wireframe:
            current_ldview_settings["EdgesOnly"] = 1

        for key, value in current_ldview_settings.items():
            ldv_command_parts.append(f"-{key}={value}")

        ldv_command_parts.append(ldrfile)

        try:
            # Using Popen and wait as before. For more control, subprocess.run is an option.
            process = subprocess.Popen(
                ldv_command_parts, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()  # Wait and get output
            if process.returncode != 0:
                self._logoutput(
                    f"LDView process error for {ldrfile}. Return code: {process.returncode}",
                    level=0,
                )
                if stdout:
                    self._logoutput(
                        f"LDView STDOUT: {stdout.decode(errors='ignore')}", level=0
                    )
                if stderr:
                    self._logoutput(
                        f"LDView STDERR: {stderr.decode(errors='ignore')}", level=0
                    )
                return
        except FileNotFoundError:
            self._logoutput(
                f"Error: LDView executable not found at {LDVIEW_BIN}", level=0
            )
            return
        except Exception as e:
            self._logoutput(f"Error running LDView for {ldrfile}: {e}", level=0)
            return

        if self.log_output:
            fni_display = colour_path_str(os.path.basename(ldrfile))
            fno_display = colour_path_str(os.path.basename(filename_to_render))
            self._logoutput(
                f"rendered file {fni_display} to {fno_display}...",
                t_start_render,
                level=0,
            )

        if os.path.isfile(filename_to_render):
            if self.auto_crop:
                self.crop(filename_to_render)
            if self.image_smooth:
                self.smooth(filename_to_render)
        else:
            self._logoutput(
                f"Warning: Output file {filename_to_render} not created by LDView.",
                level=1,
            )

    def crop(self, filename: str):
        """Crop image file."""
        t_start_crop = datetime.datetime.now()
        try:
            im = Image.open(filename)
            original_mode = im.mode
            if im.mode != "RGBA":
                im = im.convert("RGBA")

            alpha = im.getchannel("A")
            bbox = alpha.getbbox()  # Bounding box of non-zero alpha

            if not bbox:  # If image is fully transparent or alpha channel is unhelpful
                # Fallback to old method: difference from a background color
                # Use a corner pixel that is likely background. (0,0) is common.
                # Ensure image is not paletted for getpixel if it might be.
                temp_im_for_bg = im
                if temp_im_for_bg.mode == "P":  # Paletted
                    temp_im_for_bg = temp_im_for_bg.convert(
                        "RGBA"
                    )  # Convert to get actual pixel

                # Try to pick a background color that's not part of the object
                # For LDraw, white or transparent is common. If alpha exists, transparent is best.
                # If no alpha, and top-left is part of object, this might fail.
                # A more robust method would be flood fill from corners or use a known bg color.
                # For now, using the original logic's fallback path with slight adjustment.
                bg_color_ref = temp_im_for_bg.getpixel(
                    (0, 0)
                )  # Get pixel from potentially converted image

                bg = Image.new(
                    im.mode, im.size, bg_color_ref
                )  # Use determined bg color
                diff = ImageChops.difference(im, bg)

                # To make the difference more pronounced for getbbox, especially if diff is subtle
                # Convert to grayscale and threshold can also work.
                # Original: diff = ImageChops.add(diff, diff, 2.0, 0)
                # Alternative: enhance contrast or convert to L and threshold
                # For now, let's try to make it more robust for transparent images
                if original_mode in (
                    "LA",
                    "RGBA",
                ):  # If original had alpha, use that for diff
                    diff_alpha = diff.getchannel("A")
                    bbox_diff = diff_alpha.getbbox()
                    if not bbox_diff:  # If alpha diff is also empty, try intensity diff
                        bbox_diff = diff.convert(
                            "L"
                        ).getbbox()  # Convert to L for intensity bbox
                    bbox = bbox_diff
                else:  # No alpha, rely on color difference
                    bbox = diff.getbbox()

            if bbox:
                im2 = im.crop(bbox)
            else:
                im2 = im  # No valid bbox found, use original image

            im2.save(filename)
            if self.log_output:
                fn_display = colour_path_str(os.path.basename(filename))
                self._logoutput(
                    f"> cropped {fn_display} from ({_coord_str(im.size)}) to ({_coord_str(im2.size)})",
                    t_start_crop,
                )
        except FileNotFoundError:
            self._logoutput(f"Error cropping: File not found {filename}", level=0)
        except Exception as e:
            self._logoutput(f"Error during image crop for {filename}: {e}", level=0)

    def smooth(self, filename: str):
        """Apply a smoothing filter to image file."""
        t_start_smooth = datetime.datetime.now()
        try:
            im = Image.open(filename)
            im_smooth = im.filter(ImageFilter.SMOOTH)
            im_smooth.save(filename)
            if self.log_output:
                fn_display = colour_path_str(os.path.basename(filename))
                self._logoutput(
                    f"> smoothed {fn_display} ({_coord_str(im.size)})", t_start_smooth
                )
        except FileNotFoundError:
            self._logoutput(f"Error smoothing: File not found {filename}", level=0)
        except Exception as e:
            self._logoutput(f"Error during image smooth for {filename}: {e}", level=0)
