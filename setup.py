#! /usr/bin/env python
from setuptools import find_packages, setup

DISTNAME = "Inzynierka"
DESCRIPTION = "Praca inzynierska"
MAINTAINER = "Pawel Polski"
MAINTAINER_EMAIL = "253401@student.pwr.edu.pl"
URL = "https://github.com/pawel150199/Inzynierka"
INSTALL_REQUIRES = ["numpy", "scipy", "scikit-learn", "tabulate", "imblearn"]

setup(
    name = DISTNAME,
    maintainer=MAINTAINER,
    description=DESCRIPTION,
    maintainer_email=MAINTAINER_EMAIL,
    url=URL,
    install_requires=INSTALL_REQUIRES
)