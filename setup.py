from setuptools import setup, find_packages

setup(
    name="circleclust",
    version="0.0.1",
    author="tim.pyrkov@gmail.com",
    description="Clustering on periodic circular coordinates.",
    long_description=read("README.md"),
    license = "MIT License",
    long_description_content_type="text/markdown",
    url="https://github.com/timpyrkov/circleclust",
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Artistic Software",
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
