"""
DINOCell Setup Configuration
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
long_description = (Path(__file__).parent / "README.md").read_text()

# Read requirements
requirements = []
with open('requirements.txt') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            requirements.append(line)

setup(
    name="dinocell",
    version="0.1.0",
    author="DINOCell Team",
    description="Cell segmentation using DINOv3 with SSL pretraining",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(include=['dinocell', 'dinocell.*']),
    python_requires=">=3.8",
    install_requires=requirements,
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    entry_points={
        'console_scripts': [
            'dinocell=dinocell.cli:main',
        ],
    },
)
