Installation
============

Requirements
------------

GastroPy requires Python 3.10 or later.

Install from PyPI
-----------------

.. code-block:: bash

   pip install gastropy

Optional Dependencies
---------------------

For neuroimaging support (fMRI, EEG, MEG):

.. code-block:: bash

   pip install gastropy[neuro]

Development Installation
------------------------

To install GastroPy for development:

.. code-block:: bash

   git clone https://github.com/embodied-computation-group/gastropy.git
   cd gastropy
   pip install -e ".[dev,docs]"
   pre-commit install
