#! /usr/bin/env python
from __future__ import absolute_import
import os
from setuptools import find_packages, setup


this_directory = os.path.abspath(os.path.dirname(__file__))

DISTNAME = "Inzynierka"
DESCRIPTION = "Praca inzynierska"
MAINTAINER = "Pawel Polski"
MAINTAINER_EMAIL = "253401@student.pwr.edu.pl"
URL = "https://github.com/pawel150199/Inzynierka"
INSTALL_REQUIRES = ["numpy", "scipy", "scikit-learn", "tabulate", "imblearn"]

setup(
    name = DISTNAME,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES
)