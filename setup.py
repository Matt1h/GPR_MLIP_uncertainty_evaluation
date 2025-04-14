from setuptools import find_packages, setup

setup(
    name='GPR_MLIP',
    version='0.1.0',
    author="Matthias Holzenkamp",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
