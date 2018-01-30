"""Setup file."""

import os

from distutils.core import setup

NAME = 'takensembed'


install_requires = [	'numpy>=1.13.1',
                     'scipy>=0.19.1',
                     'future>=0.16.0',
                     'scikit-learn>=0.18.2']


def read(fname):
    """Read the readme file."""
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name=NAME,
    install_requires=install_requires,
    version="0.1",
    author="Sander Keemink",
    author_email="swkeemink@scimail.eu",
    description="Uses Taken's embedding theorem to estimate causality" +
                " between variables. ",
    url="https://github.com/swkeemink/takens-embed",
    download_url="",
    package_dir={NAME: "./takensembed"},
    packages=[NAME],
    license="MIT",
    long_description=read('README.md'),
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python",
        "Topic :: Scientific/Engineering"
    ]
)
