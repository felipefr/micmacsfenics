import setuptools
import os


setuptools.setup(
    name='micmacsfenics',
    author="Felipe Rocha",
    author_email="felipe.figueredo-rocha@u-pec.fr",
    version="0.1",
    url="https://github.com/felipefr/micmacsfenics",
    packages=setuptools.find_packages(include=['micmacsfenics', 'micmacsfenics.*']),
    license='MIT License',
    description='Easy Fe2 using Fenics'
)
