#!/usr/bin/env python3
import setuptools
import numpy

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="ndarray_ducktypes",
    version="alpha",
    author="Allan Haldane",
    author_email="allan.haldane@temple.edu",
    description=("A set of ndarray ducktypes testing the new "
                 "__array_function__ and __array_ufunc__ functionality in "
                 "numpy. "),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ahaldane/ndarray_ducktypes",
    packages=setuptools.find_packages(),
    include_package_data = True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License ::  BSD-3-Clause License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.7',
    install_requires=[
       'numpy>=1.14',
    ]
)
