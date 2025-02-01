import sys
from codecs import open
from os import path
from setuptools import setup, find_packages

here = path.abspath(path.dirname(__file__))

pypi_operations = frozenset(["register", "upload"]) & frozenset(
    [x.lower() for x in sys.argv]
)
if pypi_operations:
    raise ValueError(
        "Command(s) {} disabled in this example.".format(", ".join(pypi_operations))
    )

long_description = "YAHPO Gym (Yet Another Hyperparameter Optimization Gym) is a collection of interesting problems to benchmark hyperparameter optimization (HPO) methods described in https://arxiv.org/abs/2109.03670."

__version__ = None
exec(open(path.join(here, "yahpo_gym/about.py")).read())
if __version__ is None:
    raise IOError("about.py in project lacks __version__!")

setup(
    name="yahpo_gym",
    version=__version__,
    author="Florian Pfisterer, Lennart Schneider",
    description="Inference module for YAHPO Gym",
    long_description=long_description,
    license="Apache-2.0",
    python_requires="<=3.11.9",
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    include_package_data=True,
    install_requires=["onnxruntime>=1.10.0", "pyyaml", "configspace<=0.6.1", "numpy<2.0.0", "pandas"],
    extras_require={
        "test": ["pytest>=4.6", "mypy", "pre-commit", "pytest-cov"],
        "docs": [
            "sphinx",
            "sphinx-gallery",
            "sphinx_bootstrap_theme",
            "numpydoc",
            "pandas",
        ],
    },
    keywords=["module", "train", "yahpo"],
    url="https://github.com/slds-lmu/yahpo_gym",
    classifiers=["Development Status :: 3 - Alpha"],
)
