import os


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


DESCRIPTION = "NeuroChaT: Neuron Characterisation Toolbox"
LONG_DESCRIPTION = """NeuroChaT is a neuroscience toolbox written in Python.
"""

DISTNAME = 'neurochat'
MAINTAINER = 'Md Nurul Islam and Sean Martin'
MAINTAINER_EMAIL = 'martins7@tcd.ie'
URL = 'https://github.com/seankmartin/NeuroChaT'
DOWNLOAD_URL = 'https://github.com/seankmartin/NeuroChaT'
VERSION = '1.0'

INSTALL_REQUIRES = [
    'PyPDF2 >= 1.26.0',
    'PyQt5 >= 5.11.3',
    'h5py >= 2.9.0',
    'matplotlib >= 3.0.2',
    'numpy >= 1.15.0',
    'pandas >= 0.24.0',
    'scipy >= 1.2.0',
    'scikit_learn >= 0.20.2',
    'PyYAML >= 4.2b1',
    'xlrd',
    'openpyxl'
]

PACKAGES = [
    'neurochat',
    'neurochat_gui'
]

CLASSIFIERS = [
    'Intended Audience :: Science/Research',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Topic :: Scientific/Engineering :: Mathematics',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: MacOS',
    'Operating System :: Windows'
]

try:
    from setuptools import setup
    _has_setuptools = True
except ImportError:
    from distutils.core import setup

if __name__ == "__main__":

    setup(name=DISTNAME,
          author=MAINTAINER,
          author_email=MAINTAINER_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          license=read('LICENSE'),
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          install_requires=INSTALL_REQUIRES,
          include_package_data=True,
          packages=PACKAGES,
          classifiers=CLASSIFIERS,
          )
