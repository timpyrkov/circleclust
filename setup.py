from setuptools import setup, find_packages

# Read the contents of README file
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

setup(
    name="circleclust",
    version="0.0.2",
    author="Tim Pyrkov",
    author_email="tim.pyrkov@gmail.com",
    description="Clustering on periodic circular coordinates.",
    long_description=long_description,
    license = "MIT",
    long_description_content_type="text/markdown",
    url="https://github.com/timpyrkov/circleclust",
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "matplotlib>=3.5",
        "pillow>=9.0",
        "kneed>=0.8.2",
        "scipy>=1.8",
    ],
)
