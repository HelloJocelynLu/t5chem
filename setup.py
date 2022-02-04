import io
import os
import re

from pathlib import Path
from setuptools import find_packages
from setuptools import setup

def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


def version():
    """ Get the local package version. """
    namespace = {}
    path = Path("t5chem", "__version__.py")
    exec(path.read_text(), namespace)
    return namespace["__version__"]


setup(
    name="t5chem",
    version=version(),
    url="https://github.com/HelloJocelynLu/t5chem",
    license='MIT',

    author="Jocelyn Lu",
    author_email="jl8570@nyu.edu",

    description="A Unified Deep Learning Model for Multi-task Reaction Predictions",
    long_description=read("README.rst"),

    packages=find_packages(exclude=('tests',)),
    package_data={'t5chem': ['vocab/*']},
    entry_points={
        'console_scripts': [
            't5chem = t5chem.__main__:main',
        ],
    },
    install_requires=[
        "transformers==4.10.2",
        "selfies==1.0.4",
        "scikit-learn>=0.24.1",
        "torchtext<=0.8.1",
        "scipy>=1.6.0",
        "tensorboard>=2.6.0",
        "pandas>=1.2.4",
        "rdkit-pypi",
    ],
    
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
)
