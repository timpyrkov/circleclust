from setuptools import setup, find_packages

setup(
    name="circleclust",
    version="0.0.1",
    description="Clustering on periodic circular coordinates.",
    author="tim.pyrkov@gmail.com",
    packages=find_packages(exclude=("tests", "tests.*")),
    include_package_data=True,
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20",
        "pandas>=1.3",
        "matplotlib>=3.5",
        "pillow>=9.0",
        "kneed>=0.8.2",
        "scipy>=1.8",
    ],
)
