import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())


setup(
    name="t5chem",
    version="0.8.0",
    url="https://github.com/HelloJocelynLu/t5chem",
    license='MIT',

    author="Jocelyn Lu",
    author_email="jl8570@nyu.edu",

    description="A Unified Deep Learning Model for Multi-task Reaction Predictions",
    long_description=read("README.rst"),

    packages=find_packages(exclude=('tests',)),

    install_requires=[
        "transformers==4.10.2",
        "selfies==1.0.4",
        "shap==0.39.0",
        "scikit-learn==0.24.1",
        "torchtext==0.8.0",
        "scipy==1.6.0",
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
)
