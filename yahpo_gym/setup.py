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
    description="Inference module for the yahpo gym",
    long_description=long_description,
    license="Apache-2.0",
    packages=find_packages(exclude=["contrib", "docs", "tests*"]),
    include_package_data=True,
    install_requires=["onnxruntime>=1.10.0", "pyyaml", "configspace<=0.6.1", "pandas"],
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
    entry_points={
        "console_scripts": ["setup-yahpo = yahpo_gym.scripts.setup_yahpo:main"]
    },
    keywords=["module", "inference", "yahpo"],
    url="https://github.com/slds-lmu/yahpo_gym",
    classifiers=[
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Natural Language :: English",
    ],
    scripts=["scripts/setup_yahpo.py"],
)
