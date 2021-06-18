from setuptools import setup, find_packages
import pathlib

with open("README.md", "r") as f:
    long_description = f.read()

setup(
    name='keepingtheBEAT',
    version='0.1.1',
    description='test This package is for the Bayesian Evidence Analysis Tool (BEAT) originally written for '
                'measuring and dissecting multi-component emission lines',
    long_description=long_description,  # Optional
    long_description_content_type='text/markdown',

    url='https://github.com/dzhuliya/keeping-the-BEAT',
    author='Dzhuliya Dashtamirova',
    author_email='dashtamirova@stsci.edu',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3'
    ],
    keywords='astronomy',
    packages=find_packages(),

    python_requires='>=3.6',
    install_requires=["astropy", "numpy>=1.14.2", "pymultinest",
                      "matplotlib>=2.2.2", "pandas"],

    project_urls={
        'GitHub': 'https://github.com/dzhuliya/keeping-the-BEAT',
        'readthedocs': 'https://keeping-the-beat.readthedocs.io/',
    },
)