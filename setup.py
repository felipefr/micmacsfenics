from setuptools import find_packages, setup

setup(name="micmacsfenics",
      description="Easy Fe2 using Fenics",
      long_description="Easy Fe2 using Fenics",
      author="Felipe Rocha",
      author_email="f.rocha.felipe@gmail.com",
      version="0.1",
      license="GNU Library or Lesser General Public License (LGPL)",
      url="https://github.com/felipefr/micmacsFenics",
      classifiers=[
          "Development Status :: 5 - Production/Stable"
          "Intended Audience :: Developers",
          "Intended Audience :: Science/Research",
          "Programming Language :: Python :: 3",
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "Programming Language :: Python :: 3.8",
          "Programming Language :: Python :: 3.9",
          "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
          "Topic :: Scientific/Engineering :: Mathematics",
          "Topic :: Software Development :: Libraries :: Python Modules",
      ],
      packages=find_packages(),
      include_package_data=True,
      zip_safe=False
      )
