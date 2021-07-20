"""Setup script."""

from setuptools import setup

classifiers = """
Development Status :: 4 - Beta
Intended Audience :: Science/Research
Natural Language :: English
Operating System :: OS Independent
Programming Language :: Python :: 3.6
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Topic :: Utilities
Topic :: Scientific/Engineering
Topic :: Software Development :: Libraries :: Python Modules
License :: OSI Approved :: MIT License
"""

setup(
    name='v2e_exps',
    version="0.1.0",

    author="Yuhuang Hu",
    author_email="yuhuang.hu@ini.uzh.ch",

    packages=["v2e_exps"],

    classifiers=list(filter(None, classifiers.split('\n'))),
)
