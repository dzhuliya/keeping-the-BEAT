Welcome to Keeping the BEAT's documentation!
============================================
~**in progress**~
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

If you run into issues installing PyMultiNest with pip install, try:

.. code::

    conda install -c conda-forge pymultinest

good luck!!!

How to get started
------------------
A jupyter notebook is provided to run through an example fit using files included
when you clone the repository. You can now open the jupyter notebook and follow
the instructions in the notebook. (additional documentation will be added to this
page later).

.. toctree::
   :maxdepth: 2
   :caption: Contents:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
