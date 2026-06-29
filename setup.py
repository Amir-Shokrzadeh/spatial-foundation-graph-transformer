"""
Editable install so that `src` is importable as a top-level package
throughout the project without path hacks.

Usage:
    pip install -e .
"""
from setuptools import setup, find_packages

setup(
    name="sfgt",
    version="0.1.0",
    description="Spatial Foundation Graph Transformer for Tumor Immune Niche Representation Learning",
    author="Amir Shokrzadeh",
    author_email="",
    url="https://github.com/Amir-Shokrzadeh/spatial-foundation-graph-transformer",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],   # managed by environment.yml / requirements.txt
    extras_require={
        "dev": ["black", "isort", "mypy", "pytest"],
    },
)
