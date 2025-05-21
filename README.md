# ldraw-pyV2

[![Python Version](https://img.shields.io/static/v1?label=python&message=3.8%2B&color=blue&style=flat&logo=python)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
<a href="https://github.com/psf/black"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-000000.svg"></a>
[![GitHub Actions CI](https://github.com/2minCS/ldraw-pyV2/actions/workflows/ci.yml/badge.svg)](https://github.com/2minCS/ldraw-pyV2/actions/workflows/ci.yml)

A modernized and enhanced utility package for creating, modifying, and reading LDraw files and data structures.

LDraw is an open standard for LEGO® CAD software. It is based on a hierarchy of elements describing primitive shapes up to complex LEGO models and scenes.

## About this Version (ldraw-pyV2)

This repository is a fork of the original [ldraw-py by michaelgale](https://github.com/michaelgale/ldraw-py), modernized and adapted for ongoing development. The primary goals of this V2 are:

* To update the codebase to use modern Python practices and tooling (Python 3.8+).
* To improve maintainability, readability, and extensibility.
* To serve as a foundation for potential new features and enhancements.
* To ensure robust testing and CI using GitHub Actions.

## Installation

### From PyPI (Future - if published)

If this version is published to PyPI in the future:
```shell
pip install ldrawpyV2 # Or the chosen package name
```
### From GitHub (Recommended for current use)You can install this package directly from this GitHub repository:
```shell
pip install git+https://github.com/2minCS/ldraw-pyV2.git
```
### For development, clone the repository and install in editable mode:
```shell
git clone https://github.com/2minCS/ldraw-pyV2.git
cd ldraw-pyV2
python -m venv .venv
source .venv/bin/activate  # On Linux/macOS
# .venv\Scripts\activate    # On Windows
pip install -r requirements-dev.txt
pip install -e .
```
### Usage 
After installation, the package can be imported (the import name ldrawpy is retained for compatibility unless the package directory itself is renamed):
```shell
from ldrawpyV2 import LDRColour, LDRPart, Vector


# Create a white colour using LDraw colour code 15 for white
my_colour = LDRColour(15)
print(my_colour)

# Example of creating a part
my_part = LDRPart(colour=my_colour.code, name="3001")
my_part.attrib.loc = Vector(10, 0, 20) # Assuming Vector is from toolbox or ldrawpy
print(str(my_part))
```
Expected Output:
```shell
White
1 15 10 0 20 1 0 0 0 1 0 0 0 1 3001.dat
```
## Requirements
- Python 3.8+
- Pillow
- Rich
- toolbox-py (handled via install_requires from a specified fork/commit)(See setup.py for specific versions and requirements-dev.txt for development dependencies.)
## Key Features (Inherited and Evolving)
- **LDraw Primitives:** Classes for Lines, Triangles, Quads, and Parts.

- **Colour Management:** ```LDRColour``` class for handling LDraw color codes, names, and RGB values.

- **Model Parsing:** ```LDRModel```  class for parsing LDraw files, including submodels and steps.

- **LDView Rendering:** Helper class ```LDViewRender``` to automate rendering snapshots using LDView (requires LDView installation).

- **Geometric Shapes:** Generation of basic shapes like boxes, cylinders, circles.

- **Arrow Generation:** Utilities for creating LDraw arrows for instructions.

- **Command-Line Utility:** ```ldrcat``` script for displaying and cleaning LDraw files.

## Development
This project uses:

- **pytest** for testing.

- **Black** for code formatting (line length 88).

- **Flake8** for linting.

- **MyPy** for static type checking.

Continuous Integration is managed via GitHub Actions (see .github/workflows/ci.yml).

To run tests locally: ```pytest```

To format code: ```black . --line-length=88```

To check for linting issues: ```flake8 .```

## References
[LDraw.org](https://www.ldraw.org/) - Official maintainer of the LDraw file format specification and the LDraw official part library.

[Original ldraw-py Repository](https://github.com/michaelgale/ldraw-py)

[ldraw-vscode](https://github.com/michaelgale/ldraw-vscode) - Visual Studio Code language extension plug-in for LDraw files.

#### Lego CAD Tools
[Bricklink studio](https://www.bricklink.com/v3/studio/download.page)

[LeoCAD](https://www.leocad.org/)

[MLCAD](http://mlcad.lm-software.com/)

[LDView](https://tcobbs.github.io/ldview/)

#### LPub Instructions Tools
[LPub3D](https://trevorsandy.github.io/lpub3d/)

## Contributing
This is primarily a personal project for modernization and learning. Contributions may be considered on a case-by-case basis.

## License
This project is licensed under the MIT License - see the LICENSE file for details. Based on the original ldraw-py by Michael Gale, also under MIT License.