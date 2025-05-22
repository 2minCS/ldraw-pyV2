#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Original ldraw-py repo: https://github.com/michaelgale/ldraw-py
# V2 fork: https://github.com/2minCS/ldraw-pyV2

import os
import sys
import setuptools

PACKAGE_NAME = "ldrawpy"
# Define your minimum Python version as a string
MINIMUM_PYTHON_VERSION_STR = "3.8"


def parse_version_tuple(version_str):
    """Parses a version string (e.g., "3.8") into a tuple (e.g., (3, 8))."""
    return tuple(map(int, version_str.split(".")))


MINIMUM_PYTHON_VERSION_TUPLE = parse_version_tuple(MINIMUM_PYTHON_VERSION_STR)


def check_python_version():
    """Exit when the Python version is too low."""
    if sys.version_info < MINIMUM_PYTHON_VERSION_TUPLE:
        sys.exit(
            f"Python {MINIMUM_PYTHON_VERSION_STR}+ is required. You are running "
            f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )


def read_package_variable(key, filename="__init__.py"):
    """Read the value of a variable from the package without importing."""
    module_path = os.path.join(PACKAGE_NAME, filename)
    with open(module_path, encoding="utf-8") as module:  # Added encoding
        for line in module:
            parts = line.strip().split(" ", 2)
            if parts[:-1] == [key, "="]:
                return parts[-1].strip("'")
    sys.exit(f"'{key}' not found in '{module_path}'")


def build_description():
    """Build a description for the project from documentation files."""
    readme_path = "README.md"
    try:
        with open(readme_path, "r", encoding="utf-8") as f:
            readme = f.read()
    except IOError:
        readme = "A Python utility package for creating, modifying, and reading LDraw files and data structures."  # Fallback

    # If you create a CHANGELOG.md and want to include it:
    # changelog_path = "CHANGELOG.md"
    # try:
    #     with open(changelog_path, "r", encoding="utf-8") as f:
    #         changelog = f.read()
    #     return readme + "\n\n" + changelog
    # except IOError:
    #     pass
    return readme


check_python_version()

setuptools.setup(
    name=read_package_variable("__project__"),
    version=read_package_variable("__version__"),
    description="A Python utility package (V2) for creating, modifying, and reading LDraw files and data structures.",
    long_description=build_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/2minCS/ldraw-pyV2",
    author="Casey Mauldin",
    author_email="dw4pres@gmail.com",
    license="MIT",
    packages=setuptools.find_packages(exclude=["tests", "tests.*"]),
    python_requires=f">={MINIMUM_PYTHON_VERSION_STR}",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",  # Adjusted for a V2 effort
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Graphics :: 3D Modeling",
    ],
    install_requires=[
        "pillow",
        "rich",
        "numpy", # Added to correct CI issues with toolbox
        # Fixed dependency issue with actions expecting toolbox not toolbox-py
        "toolbox @ git+https://github.com/2minCS/toolbox-py.git@2be0b001e83ec15a6f0db137741292d77b57c1be",
    ],
    entry_points={
        "console_scripts": [
            "ldrcat=ldrawpy.scripts.ldrcat:main",  # Or ldrcat-v2 if you change the script name
        ]
    },
)
