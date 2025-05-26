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

import tempfile
import datetime
import subprocess

# import shlex # Not used
# from collections import defaultdict # Not used
from pathlib import Path  # ADDED for pathlib
from PIL import Image, ImageOps, ImageChops, ImageFilter  # type: ignore
from typing import Optional, List, Tuple, Any, Dict, Union

# Explicit imports from toolbox
from toolbox import (  # type: ignore
    apply_params,
    logmsg,
    # split_path, # Not used in this file directly
    # full_path, # Replaced by pathlib
    colour_path_str,
    # Vector, # Not directly used in this file
)

# LDVIEW_BIN is the path to the LDView executable.
# This path might need to be configured based on the user's system.
LDVIEW_BIN = "/Applications/LDView.app/Contents/MacOS/LDView"

# LDVIEW_DICT contains default settings for LDView.
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
    """
    Calculates the camera distance for LDView based on scale, DPI, and page width.
    This helps in achieving a consistent view for rendered images.
    """
    one = 20 * 1 / 64 * dpi * scale
    sz = page_width * dpi / one * LDU_DISTANCE * 0.775
    sz *= 1700 / 1000
    return sz


def _coord_str(
    x: Union[float, Tuple[float, float], List[float]],
    y: Optional[float] = None,
    sep: str = ", ",
) -> str:
    """
    Formats a coordinate pair (or a single coordinate if y is provided) into a string.
    Used for logging image dimensions.
    """
    if isinstance(x, (tuple, list)) and len(x) == 2:
        a, b = float(x[0]), float(x[1])
    elif isinstance(x, (int, float)) and isinstance(y, (int, float)):
        a, b = float(x), float(y)
    else:
        # Return an error string or raise an error if coordinates are invalid
        return "coord_error"
    # Format floats to a fixed number of decimal places, then strip trailing zeros and decimal point
    sa = f"{a:.6f}".rstrip("0").rstrip(".")
    sb = f"{b:.6f}".rstrip("0").rstrip(".")
    return f"{sa}{sep}{sb}"


class LDViewRender:
    """
    A class to manage rendering LDraw files or strings using LDView.
    It handles LDView command-line arguments, temporary file creation,
    and optional post-processing like auto-cropping and image smoothing.
    """

    # Type hints for instance variables
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
    output_path: Optional[str]  # This will be converted to Path if used
    log_output: bool
    log_level: int
    overwrite: bool
    ldr_temp_path: Path  # CHANGED to Path
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
        """
        Initializes the LDViewRender instance.
        Sets default parameters and applies any keyword arguments.
        """
        for param, default_value in self.PARAMS.items():
            setattr(self, param, default_value)

        # Use pathlib for temporary path construction
        self.ldr_temp_path = Path(tempfile.gettempdir()) / "ldrawpy_temp.ldr"
        apply_params(self, kwargs)  # Apply user-provided kwargs
        self.set_page_size(self.page_width, self.page_height)
        self.set_scale(self.scale)
        self.settings_snapshot = None

    def __str__(self) -> str:
        """
        Returns a string representation of the LDViewRender settings.
        """
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
        """
        Saves the current rendering settings to allow temporary changes and restoration.
        """
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
        """
        Restores rendering settings from a previously saved snapshot.
        """
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
        # Re-apply dependent settings
        self.set_page_size(self.page_width, self.page_height)
        self.set_scale(self.scale)
        self.settings_snapshot = None  # Clear snapshot after restoring

    def set_page_size(self, width: float, height: float):
        """
        Sets the page size for rendering and calculates pixel dimensions.
        Args:
            width: Page width in inches.
            height: Page height in inches.
        """
        self.page_width = width
        self.page_height = height
        self.pix_width = int(self.page_width * self.dpi)
        self.pix_height = int(self.page_height * self.dpi)
        self.args_size = f"-SaveWidth={self.pix_width} -SaveHeight={self.pix_height}"

    def set_dpi(self, dpi: int):
        """
        Sets the DPI for rendering and updates dependent page and scale settings.
        Args:
            dpi: Dots per inch.
        """
        self.dpi = dpi
        self.set_page_size(self.page_width, self.page_height)
        self.set_scale(self.scale)  # Scale depends on DPI for camera distance

    def set_scale(self, scale: float):
        """
        Sets the rendering scale and calculates the camera distance.
        Args:
            scale: The rendering scale factor.
        """
        self.scale = scale
        self.cam_dist = int(camera_distance(self.scale, self.dpi, self.page_width))
        self.args_cam = f"-ca0.01 -cg0.0,0.0,{self.cam_dist}"

    def _logoutput(
        self, msg: str, tstart: Optional[datetime.datetime] = None, level: int = 2
    ):
        """
        Helper method for logging messages if log_output is enabled.
        """
        if self.log_output:
            logmsg(
                msg, tstart=tstart, level=level, prefix="LDR", log_level=self.log_level
            )

    def render_from_str(self, ldrstr: str, outfile: Union[str, Path]):
        """
        Renders an LDraw model provided as a string.
        Args:
            ldrstr: The LDraw content as a string.
            outfile: The path (string or Path) for the output image.
        """
        if self.log_output:
            s = ldrstr.splitlines()[0] if ldrstr.splitlines() else ""
            self._logoutput(f"rendering string ({s[:min(len(s), 80)]})...")
        try:
            # self.ldr_temp_path is already a Path object
            with open(self.ldr_temp_path, "w", encoding="utf-8") as f:
                f.write(ldrstr)
            self.render_from_file(
                self.ldr_temp_path, Path(outfile)
            )  # Ensure outfile is Path
        except IOError as e:
            self._logoutput(f"Error writing temporary LDR file: {e}", level=0)

    def render_from_parts(self, parts: List[Any], outfile: Union[str, Path]):
        """
        Renders an LDraw model from a list of LDRPart (or stringable) objects.
        Args:
            parts: A list of LDraw parts.
            outfile: The path (string or Path) for the output image.
        """
        if self.log_output:
            self._logoutput(f"rendering parts ({len(parts)})...")
        self.render_from_str(
            "".join([str(p) for p in parts]), Path(outfile)
        )  # Ensure outfile is Path

    def render_from_file(self, ldrfile: Union[str, Path], outfile: Union[str, Path]):
        """
        Renders an LDraw model from a file.
        Args:
            ldrfile: Path (string or Path) to the input LDraw file.
            outfile: Path (string or Path) for the output image.
        """
        t_start_render = datetime.datetime.now()

        ldrfile_path = (
            Path(ldrfile).expanduser().resolve()
        )  # Ensure ldrfile is an absolute Path
        current_outfile_path = Path(outfile)  # Convert input outfile to Path

        filename_to_render: Path
        if self.output_path:
            output_dir_base = Path(self.output_path).expanduser().resolve()
            if current_outfile_path.is_absolute():
                filename_to_render = current_outfile_path
            else:
                filename_to_render = output_dir_base / current_outfile_path
        else:
            filename_to_render = current_outfile_path.resolve()

        # Ensure the parent directory for the final render target exists
        filename_to_render.parent.mkdir(parents=True, exist_ok=True)

        if not self.overwrite and filename_to_render.is_file():
            if self.log_output:
                fno_display = colour_path_str(
                    filename_to_render.name
                )  # .name is a string
                self._logoutput(f"rendered file {fno_display} already exists, skipping")
            return

        # Construct LDView command
        # LDView executable path and output snapshot path must be strings
        ldv_cmd = [
            LDVIEW_BIN,
            f"-SaveSnapShot={str(filename_to_render)}",  # Convert Path to string
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
                    "UseQualityStuds": 0,  # Also often disabled if no lines
                }
            )
        if self.wireframe:
            current_settings["EdgesOnly"] = 1

        ldv_cmd.extend(f"-{k}={v}" for k, v in current_settings.items())
        ldv_cmd.append(str(ldrfile_path))  # Convert Path to string for subprocess

        try:
            process = subprocess.Popen(
                ldv_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = process.communicate()
            if process.returncode != 0:
                self._logoutput(
                    f"LDView error for {ldrfile_path.name}. Code: {process.returncode}",
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
            self._logoutput(f"Error: LDView not found at {LDVIEW_BIN}", level=0)
            return
        except Exception as e:
            self._logoutput(
                f"Error running LDView for {ldrfile_path.name}: {e}", level=0
            )
            return

        if self.log_output:
            fni_disp = colour_path_str(ldrfile_path.name)
            fno_disp = colour_path_str(filename_to_render.name)
            self._logoutput(
                f"rendered file {fni_disp} to {fno_disp}...", t_start_render, level=0
            )

        if filename_to_render.is_file():
            if self.auto_crop:
                self.crop(filename_to_render)
            if self.image_smooth:
                self.smooth(filename_to_render)
        else:
            self._logoutput(
                f"Warning: Output {filename_to_render.name} not created.", level=1
            )

    def crop(self, filename: Union[str, Path]):
        """
        Auto-crops an image to remove empty borders.
        Args:
            filename: Path (string or Path) to the image file.
        """
        t_start_crop = datetime.datetime.now()
        filepath = Path(filename)  # Ensure it's a Path object
        try:
            im: Image.Image = Image.open(filepath)
            original_mode = im.mode  # Store original mode
            if im.mode != "RGBA":  # Ensure RGBA for alpha channel operations
                im = im.convert("RGBA")

            alpha = im.getchannel("A")
            bbox = alpha.getbbox()  # Get bounding box from alpha channel

            # Fallback for images without useful alpha (e.g., fully opaque PNGs from some sources)
            if not bbox:
                # Create a background of the corner pixel color to diff against
                # This helps find the content area if alpha is not informative
                temp_im_for_bg = (
                    im if im.mode == "P" else im.convert("RGBA")
                )  # Handle palette mode
                bg_color_ref = temp_im_for_bg.getpixel((0, 0))
                bg = Image.new(im.mode, im.size, bg_color_ref)
                diff = ImageChops.difference(im, bg)
                # Get bbox from the difference image's alpha or luminance
                bbox_diff = (
                    diff.getchannel("A").getbbox()
                    if original_mode in ("LA", "RGBA")  # If original had alpha
                    else diff.convert(
                        "L"
                    ).getbbox()  # Otherwise, use luminance of difference
                )
                bbox = bbox_diff

            im2 = (
                im.crop(bbox) if bbox else im
            )  # Crop if bbox is valid, else use original
            im2.save(filepath)  # Save the cropped image
            if self.log_output:
                fn_disp = colour_path_str(filepath.name)
                self._logoutput(
                    f"> cropped {fn_disp} from ({_coord_str(im.size)}) to ({_coord_str(im2.size)})",
                    t_start_crop,
                )
        except FileNotFoundError:
            self._logoutput(f"Error cropping: File not found {str(filepath)}", level=0)
        except Exception as e:
            self._logoutput(
                f"Error during image crop for {str(filepath)}: {e}", level=0
            )

    def smooth(self, filename: Union[str, Path]):
        """
        Applies a smoothing filter to an image.
        Args:
            filename: Path (string or Path) to the image file.
        """
        t_start_smooth = datetime.datetime.now()
        filepath = Path(filename)  # Ensure it's a Path object
        try:
            im: Image.Image = Image.open(filepath)
            im_smooth = im.filter(ImageFilter.SMOOTH)  # Apply smooth filter
            im_smooth.save(filepath)  # Save the smoothed image
            if self.log_output:
                fn_disp = colour_path_str(filepath.name)
                self._logoutput(
                    f"> smoothed {fn_disp} ({_coord_str(im.size)})", t_start_smooth
                )
        except FileNotFoundError:
            self._logoutput(f"Error smoothing: File not found {str(filepath)}", level=0)
        except Exception as e:
            self._logoutput(
                f"Error during image smooth for {str(filepath)}: {e}", level=0
            )
