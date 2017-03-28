# -*- coding: utf-8 -*-
# from distutils.core import setup
from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()
setup(
    name='nitrotyper',
    # packages=['nitrotyper'],  # this must be the same as the name above
    version='0.4.2',
    description='https://www.nitrotype.com auto typer',
    long_description=long_description,
    author='winxos',
    author_email='winxos@hotmail.com',
    url='https://github.com/winxos/nitrotyper',  # use the URL to the github repo
    download_url='https://github.com/winxos/nitrotyper/archive/0.1.tar.gz',  # I'll explain this in a second
    keywords=['nitrotype', 'winxos', 'auto'],  # arbitrary keywords
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 4 - Beta',

        # Indicate who your project is intended for
        'Intended Audience :: End Users/Desktop',
        'Topic :: Games/Entertainment',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.2',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    install_requires=['pyautogui', 'opencv-python'],
    packages=['nitrotyper'],
    package_dir={"nitrotyper": "nitrotyper"},
    package_data={
        "nitrotyper": ["data/*.json"]
    }
)
