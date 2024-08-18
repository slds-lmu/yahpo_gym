import sys
from codecs import open
from os import path
from setuptools import (setup, find_packages)

here = path.abspath(path.dirname(__file__))

pypi_operations = frozenset(['register', 'upload']) & frozenset([x.lower() for x in sys.argv])
if pypi_operations:
    raise ValueError('Command(s) {} disabled in this example.'.format(', '.join(pypi_operations)))

with open(path.join(here, 'README.md'), encoding='utf-8') as fh:
    long_description = fh.read()

__version__ = None
exec(open(path.join(here, 'yahpo_train/about.py')).read())
if __version__ is None:
    raise IOError('about.py in project lacks __version__!')

setup(name='yahpo_train',
      version=__version__,
      author='Florian Pfisterer, Lennart Schneider',
      description='Train surrogate models for YAHPO Gym',
      long_description=long_description,
      license='Apache-2.0',
      packages=find_packages(exclude=['contrib', 'docs', 'tests*']),
      include_package_data=True,
      install_requires=[
          'torch>=1.0.0',
          'fastai',
          'yahpo_gym @ git+https://github.com/slds-lmu/yahpo_gym#egg=yahpo_gym&subdirectory=yahpo_gym'
      ],
      keywords=['module', 'train', 'yahpo'],
      url="https://github.com/slds-lmu/yahpo_gym",
      classifiers=["Development Status :: 3 - Alpha"])
