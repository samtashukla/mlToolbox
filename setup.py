#! /usr/bin/env python
#
# Copyright (C) 2015 Stephanie Gagnon <sgagnon@stanford.edu>

descr = """mlToolbox: Toolboxes for behavioral analyses"""

import os
from setuptools import setup

DISTNAME = 'mlToolbox'
DESCRIPTION = descr
MAINTAINER = 'Steph Gagnon Sorenson'
MAINTAINER_EMAIL = 'stephanie.a.gagnon@gmail.com'
LICENSE = 'BSD (3-clause)'
URL = 'http://stanford.edu/~sgagnon'
DOWNLOAD_URL = 'https://github.com/sgagnon/mlToolbox'
VERSION = '0.0.1.dev'

def check_dependencies():

    # Just make sure dependencies exist, I haven't rigorously
    # tested what the minimal versions that will work are
    needed_deps = ["IPython", "numpy", "scipy", "matplotlib",
                   "pandas", "statsmodels"]
    missing_deps = []
    for dep in needed_deps:
        try:
            __import__(dep)
        except ImportError:
            missing_deps.append(dep)

    if missing_deps:
        missing = (", ".join(missing_deps)
                   .replace("skimage", "scikit-image"))
        raise ImportError("Missing dependencies: %s" % missing)

if __name__ == "__main__":

    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    import sys
    if not (len(sys.argv) >= 2 and ('--help' in sys.argv[1:] or
            sys.argv[1] in ('--help-commands',
                            '--version',
                            'egg_info',
                            'clean'))):
        check_dependencies()

    setup(name=DISTNAME,
        maintainer=MAINTAINER,
        maintainer_email=MAINTAINER_EMAIL,
        description=DESCRIPTION,
        license=LICENSE,
        version=VERSION,
        url=URL,
        download_url=DOWNLOAD_URL,
        packages=['mlToolbox'],
        scripts=[],
        classifiers=[
                     'Intended Audience :: Science/Research',
                     'Programming Language :: Python :: 2.7',
                     'License :: OSI Approved :: BSD License',
                     'Operating System :: POSIX',
                     'Operating System :: Unix',
                     'Operating System :: MacOS'],
    )
