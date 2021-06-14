import os
from setuptools import setup


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


DESCRIPTION = "NeuroChaT: Neuron Characterisation Toolbox"

LONG_DESCRIPTION = read("README.md")
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"

DISTNAME = 'neurochat'
MAINTAINER = 'Md Nurul Islam and Sean Martin'
MAINTAINER_EMAIL = 'martins7@tcd.ie'
URL = 'https://github.com/seankmartin/NeuroChaT'
DOWNLOAD_URL = 'https://github.com/seankmartin/NeuroChaT/archive/v1.2.0.tar.gz'
VERSION = '1.2.0'

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
    'Topic :: Scientific/Engineering :: Mathematics',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
    'Operating System :: POSIX',
    'Operating System :: Unix',
    'Operating System :: Microsoft :: Windows',
    'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',

]

ENTRY_POINTS = {
    "console_scripts": [
        "neurochat = neurochat_gui.neurochat_ui:main"
    ]
}


if __name__ == "__main__":

    setup(name=DISTNAME,
          author=MAINTAINER,
          author_email=MAINTAINER_EMAIL,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          long_description=LONG_DESCRIPTION,
          long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
          url=URL,
          version=VERSION,
          download_url=DOWNLOAD_URL,
          install_requires=INSTALL_REQUIRES,
          include_package_data=True,
          packages=PACKAGES,
          classifiers=CLASSIFIERS,
          entry_points=ENTRY_POINTS,
          )
