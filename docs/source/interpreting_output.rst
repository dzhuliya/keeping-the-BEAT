.. _my-reference-label:

Interpreting the output
==========================
*Contributors: Julia Falcone*


When BEAT has finished succesfully running, there will be files in the output folder in the location specified by the ``out_dir`` directory. This section will explain how you can interpret the output to extract the fit results, and use those results to create plots of the kinematics.

Let's look at the output folder for our very first example fit, which should be titled ``ngc7319_out``. There are two further directories: ``out`` contains the statistics of each fit, and ``plots`` show the model fits. However, there is also a text file that is output, and in the example it's called ``ngc7319_halpha.txt``. If we open it in Excel, the first several columns look like this:

.. list-table:: 
   :header-rows: 1
   :class: tight-table

   * - index
     - filename
     - ncomps
     - wave_1
     - width_1
     - flux_1_A
     - flux_1_B
   * - 0
     -  NGC7319.csv
     -  1
     - 6713.86
     - 4.52
     - 1.27E-14
     - 1.76E-14

Here is a description of the values represented in these columns:

* **index**: A counter for the number of files you're fitting. 
* **filename**: The name of the spectrum file corresponding to these fit parameters.
* **ncomps**: The number of Bayesian components determined to result in the optimal fit. The maximum possible number is constrained by the ``maxcomps`` parameter.
* **wave_1**: The wavelength centroid of the first Gaussian component corresponding to whichever emission line is listed as ``line1`` in the ``fit_instructions`` (in the case of this example: Hα at λ6563 Å. Units are in Angstroms.
* **width_1**: The Gaussian width of the first Gaussian component corresponding to whichever emission line is listed as ``line1`` in the ``fit_instructions``. Units are in sigma (Gaussian width).
* **flux_1_A**: The flux of the first Gaussian component of  ``line1``. Units: erg cm-2 s-1. 
* **flux_1_B**: The flux of the first Gaussian component of ``line2``, assuming that ``flux_free`` for this line is True. In our example, this corresponds to [N II] λ6583 Å.
  
Note that although we fit three lines in the ``fit_instructions``, only two fluxes are given in the output. This is because we defined a flux relationship between the [N II] λ6548 Å and [N II] λ6583 Å in the ``fit_instructions``, so you can use the flux output of one emission line to calculate the anticipated flux of the other.

Notice also how the columns for **wave_2**, **width_2**, **flux_2_A**, and **flux_2_B** are all zero. These parameters all represent the output for a second kinematic component, if one existed. However, because BEAT determined that the one-component fit was optimal, these values are zero.

Additionally, there are a number of columns named **wave_1_sigma**, **width_1_sigma**, **flux_1_A_sigma**, etc. The columns with "sigma" in the name are the errors for the corresponding parameters.


