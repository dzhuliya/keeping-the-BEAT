Welcome to Keeping the BEAT's documentation!
============================================
~**in progress** ~
=================

The Bayesian Evidence Analysis Tool (BEAT) is a Python-based code for measuring
and dissecting multi-component emission lines frequently observed in active
galactic nuclei (AGN) exhibiting outflowing winds. BEAT was designed to analyze
large numbers of spectra (i.e. large surveys or IFU spectroscopy) automatically
and determine the number of physical kinematic components in a given set of
emission lines.

Installations
-------------
For the time being you should clone the BEAT repository from our
`Github page <https://github.com/dzhuliya/keeping-the-BEAT>`_.

.. code::

    git clone https://github.com/dzhuliya/keeping-the-BEAT.git

You will also need to have MultiNest/PyMultiNest installed on your machine.
If you do not have MultiNest installed, follow the instructions found
`here under "on your own computer" <http://johannesbuchner.github.io/pymultinest-tutorial/install.html>`_.
If you are on MacOS, we recommend following the instructions found
`here <https://www.astrobetter.com/wiki/MultiNest+Installation+Notes>`_.

If you run into issues installing PyMultiNest with pip install, try:

.. code::

    conda install -c conda-forge pymultinest

In our experience, the following instructions worked on a Mac:

#. Install both Xcode and the Xcode Command Line Tools

#. Install Macports
    * Set up Macports to `sync via git <https://trac.macports.org/wiki/howto/SyncingWithGit>`_ instead of rsync
    * Use the  `Astrobetter <https://www.astrobetter.com/wiki/MultiNest+Installation+Notes>`_
      website guide to install gcc5, cmake and openmpi. If you have issues with installing
      openmpi via sudo port install openmpi, then try installing with brew.
    * Set correct version of gcc and mpi within Macports.  Refer to this website if you have trouble:
      https://stackoverflow.com/questions/8361002/how-to-use-the-gcc-installed-in-macports
      **Note:**  Make certain the the PATH of the user is set such that the Macports version of gfortran
      is seen as the default gfortran compiler.  Other gfortran compilers will fail.

#. Install anaconda and astroconda. Activate the astroconda environment.

#. Use pip to install mpi4py

#. Download Multinest

.. code::

    git clone https://github.com/JohannesBuchner/MultiNest.git
    cd MultiNest/build/
    cmake ..
    make
    sudo make install

6. Download PyMultiNest

.. code::

    git clone https://github.com/JohannesBuchner/PyMultiNest.git
    cd PyMultiNest
    python setup.py install

7. You can test your installation following the instructions also listed on `Astrobetter <https://www.astrobetter.com/wiki/MultiNest+Installation+Notes>`_




How to get started
------------------
A
`jupyter notebook <https://github.com/dzhuliya/keeping-the-BEAT/blob/master/keeping-the-BEAT/beat_example.ipynb>`_ 
is provided to run through an example fit using
`files <https://github.com/dzhuliya/keeping-the-BEAT/tree/master/keeping-the-BEAT/spectra>`_
included when you clone the repository. You can now open the jupyter notebook
and follow the instructions in the notebook. (additional documentation will be
added to this page later).

.. toctree::
   :maxdepth: 1
   :hidden:

    customizing_your_fits

Indices and tables
==================
* :ref:`genindex`
