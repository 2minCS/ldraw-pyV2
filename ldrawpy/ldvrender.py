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
import datetime
import subprocess
import shlex
from collections import defaultdict
from PIL import Image, ImageOps, ImageChops, ImageFilter
from typing import Optional, List, Tuple, Any, Dict, Union

# Explicit imports from toolbox
from toolbox import (
    apply_params,
    logmsg,
    split_path,
    full_path,
    colour_path_str,  # This function uses 'crayons' in toolbox
    Vector,
)

LDVIEW_BIN = "/Applications/LDView.app/Contents/MacOS/LDView"
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
LDU_DISTANCE = 114591


def camera_distance(
    scale: float = 1.0, dpi: int = 300, page_width: float = 8.5
) -> float:
    one = 20 * 1 / 64 * dpi * scale
    sz = page_width * dpi / one * LDU_DISTANCE * 0.775
    sz *= 1700 / 1000
    return sz


def _coord_str(
    x: Union[float, Tuple[float, float], List[float]],
    y: Optional[float] = None,
    sep: str = ", ",
) -> str:
    if isinstance(x, (tuple, list)) and len(x) == 2:
        a, b = float(x[0]), float(x[1])
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
        a, b = float(x), float(y)
    else:
        return "coord_error"
    # CONVERTED TO F-STRING (though original was already mostly f-string like)
    sa = f"{a:.6f}".rstrip("0").rstrip(".")  # Ensure enough precision before stripping
    sb = f"{b:.6f}".rstrip("0").rstrip(".")
    # The original crayons call was removed for simplicity earlier.
    # If color is needed: import crayons; return f"{crayons.yellow(sa)}{sep}{crayons.yellow(sb)}"
    return f"{sa}{sep}{sb}"


class LDViewRender:
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
    output_path: Optional[str]
    log_output: bool
    log_level: int
    overwrite: bool
    ldr_temp_path: str
    pix_width: int
    pix_height: int
    args_size: str
    cam_dist: int
    args_cam: str
    settings_snapshot: Optional[Dict[str, Any]]

    PARAMS = {
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

    def __init__(self, **kwargs: Any):
        for param, default_value in self.PARAMS.items():
            setattr(self, param, default_value)
        self.ldr_temp_path = os.path.join(tempfile.gettempdir(), "ldrawpy_temp.ldr")
        apply_params(self, kwargs)
        self.set_page_size(self.page_width, self.page_height)
        self.set_scale(self.scale)
        self.settings_snapshot = None

    def __str__(self) -> str:
        # ALREADY F-STRINGS (from previous update)
        return "\n".join(
            [
                "LDViewRender: ",
                f" DPI: {self.dpi}  Scale: {self.scale:.2f}",
                f" Page size: {_coord_str((self.page_width, self.page_height), sep=' x ')} in "
                f"({_coord_str((self.pix_width, self.pix_height), sep=' x ')}) pixels",
                f" Auto crop: {self.auto_crop}  Image smooth: {self.image_smooth}",
                f" Camera distance: {self.cam_dist}",
            ]
        )

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
            "dpi",
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
            "dpi",
        ]
        for key in keys_to_restore:
            if key in self.settings_snapshot:
                setattr(self, key, self.settings_snapshot[key])
        self.set_page_size(self.page_width, self.page_height)
        self.set_scale(self.scale)

    def set_page_size(self, width: float, height: float):
        self.page_width = width
        self.page_height = height
        self.pix_width = int(self.page_width * self.dpi)
        self.pix_height = int(self.page_height * self.dpi)
        # ALREADY F-STRING
        self.args_size = f"-SaveWidth={self.pix_width} -SaveHeight={self.pix_height}"

    def set_dpi(self, dpi: int):
        self.dpi = dpi
        self.set_page_size(self.page_width, self.page_height)
        self.set_scale(self.scale)

    def set_scale(self, scale: float):
        self.scale = scale
        self.cam_dist = int(camera_distance(self.scale, self.dpi, self.page_width))
        # ALREADY F-STRING
        self.args_cam = f"-ca0.01 -cg0.0,0.0,{self.cam_dist}"

    def _logoutput(
        self, msg: str, tstart: Optional[datetime.datetime] = None, level: int = 2
    ):
        # logmsg is from toolbox, assuming it handles its own formatting.
        # If logmsg itself used %-formatting, it would need update in toolbox.
        logmsg(msg, tstart=tstart, level=level, prefix="LDR", log_level=self.log_level)

    def render_from_str(self, ldrstr: str, outfile: str):
        if self.log_output:
            s = ldrstr.splitlines()[0] if ldrstr.splitlines() else ""
            # CONVERTED TO F-STRING (simplified, original had crayons)
            self._logoutput(f"rendering string ({s[:min(len(s), 80)]})...")
        try:
            with open(self.ldr_temp_path, "w", encoding="utf-8") as f:
                f.write(ldrstr)
            self.render_from_file(self.ldr_temp_path, outfile)
        except IOError as e:
            # CONVERTED TO F-STRING
            self._logoutput(f"Error writing temporary LDR file: {e}", level=0)

    def render_from_parts(self, parts: List[Any], outfile: str):
        if self.log_output:
            # CONVERTED TO F-STRING (simplified, original had crayons)
            self._logoutput(f"rendering parts ({len(parts)})...")
        self.render_from_str("".join([str(p) for p in parts]), outfile)

    def render_from_file(self, ldrfile: str, outfile: str):
        t_start_render = datetime.datetime.now()
        filename_to_render = outfile
        if self.output_path:
            os.makedirs(self.output_path, exist_ok=True)
            if not os.path.isabs(outfile) and not outfile.startswith(self.output_path):
                filename_to_render = os.path.join(
                    self.output_path, os.path.basename(outfile)
                )
        else:
            filename_to_render = full_path(outfile)

        output_dir = os.path.dirname(filename_to_render)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if not self.overwrite and os.path.isfile(filename_to_render):
            if self.log_output:
                fno_display = colour_path_str(os.path.basename(filename_to_render))
                # CONVERTED TO F-STRING
                self._logoutput(f"rendered file {fno_display} already exists, skipping")
            return

        ldv_cmd = [
            LDVIEW_BIN,
            f"-SaveSnapShot={filename_to_render}",
            self.args_size,
            self.args_cam,
        ]
        current_settings = LDVIEW_DICT.copy()
        current_settings.update(
            {
                "EdgeThickness": self.line_thickness,
                "UseQualityLighting": 1 if self.quality_lighting else 0,
                "Texmaps": 1 if self.texmaps else 0,
                "UseFlatShading": 1 if self.flat_shading else 0,
                "UseSpecular": 1 if self.specular else 0,
            }
        )
        if self.no_lines:
            current_settings.update(
                {
                    "EdgeThickness": 0,
                    "ShowHighlightLines": 0,
                    "ConditionalHighlights": 0,
                    "UseQualityStuds": 0,
                }
            )
        if self.wireframe:
            current_settings["EdgesOnly"] = 1
        # CONVERTED TO F-STRING (in loop)
        ldv_cmd.extend(f"-{k}={v}" for k, v in current_settings.items())
        ldv_cmd.append(ldrfile)

        try:
            process = subprocess.Popen(
                ldv_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                # CONVERTED TO F-STRING
                self._logoutput(
                    f"LDView error for {ldrfile}. Code: {process.returncode}", level=0
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
            # CONVERTED TO F-STRING
            self._logoutput(f"Error: LDView not found at {LDVIEW_BIN}", level=0)
            return
        except Exception as e:
            # CONVERTED TO F-STRING
            self._logoutput(f"Error running LDView for {ldrfile}: {e}", level=0)
            return

        if self.log_output:
            fni_disp = colour_path_str(os.path.basename(ldrfile))
            fno_disp = colour_path_str(os.path.basename(filename_to_render))
            # CONVERTED TO F-STRING
            self._logoutput(
                f"rendered file {fni_disp} to {fno_disp}...", t_start_render, level=0
            )

        if os.path.isfile(filename_to_render):
            if self.auto_crop:
                self.crop(filename_to_render)
            if self.image_smooth:
                self.smooth(filename_to_render)
        else:
            # CONVERTED TO F-STRING
            self._logoutput(
                f"Warning: Output {filename_to_render} not created.", level=1
            )

    def crop(self, filename: str):
        t_start_crop = datetime.datetime.now()
        try:
            im: Image.Image = Image.open(filename)
            original_mode = im.mode
            if im.mode != "RGBA":
                im = im.convert("RGBA")
            alpha = im.getchannel("A")
            bbox = alpha.getbbox()
            if not bbox:
                temp_im_for_bg = im if im.mode == "P" else im.convert("RGBA")
                bg_color_ref = temp_im_for_bg.getpixel((0, 0))
                bg = Image.new(im.mode, im.size, bg_color_ref)
                diff = ImageChops.difference(im, bg)
                bbox_diff = (
                    diff.getchannel("A").getbbox()
                    if original_mode in ("LA", "RGBA")
                    else diff.convert("L").getbbox()
                )
                bbox = bbox_diff
            im2 = im.crop(bbox) if bbox else im
            im2.save(filename)
            if self.log_output:
                fn_disp = colour_path_str(os.path.basename(filename))
                # CONVERTED TO F-STRING
                self._logoutput(
                    f"> cropped {fn_disp} from ({_coord_str(im.size)}) to ({_coord_str(im2.size)})",
                    t_start_crop,
                )
        except FileNotFoundError:
            # CONVERTED TO F-STRING
            self._logoutput(f"Error cropping: File not found {filename}", level=0)
        except Exception as e:
            # CONVERTED TO F-STRING
            self._logoutput(f"Error during image crop for {filename}: {e}", level=0)

    def smooth(self, filename: str):
        t_start_smooth = datetime.datetime.now()
        try:
            im: Image.Image = Image.open(filename)
            im_smooth = im.filter(ImageFilter.SMOOTH)
            im_smooth.save(filename)
            if self.log_output:
                fn_disp = colour_path_str(os.path.basename(filename))
                # CONVERTED TO F-STRING
                self._logoutput(
                    f"> smoothed {fn_disp} ({_coord_str(im.size)})", t_start_smooth
                )
        except FileNotFoundError:
            # CONVERTED TO F-STRING
            self._logoutput(f"Error smoothing: File not found {filename}", level=0)
        except Exception as e:
            # CONVERTED TO F-STRING
            self._logoutput(f"Error during image smooth for {filename}: {e}", level=0)
