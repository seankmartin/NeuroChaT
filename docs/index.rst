===========================================
NeuroChaT - Neuron Characterisation Toolbox
===========================================

NeuroChaT (RRID:SCR_018020) is an open-source neuron characterisation toolbox. It is described in our paper on `Wellcome Open Research <https://wellcomeopenresearch.org/articles/4-196>`_.


Installation
============
To install NeuroChaT from PyPi, run the following

.. code-block:: shell

   python -m pip install -u neurochat


Alternatively, to develop with NeuroChaT

.. code-block:: shell

   git clone https://github.com/seankmartin/NeuroChaT
   cd NeuroChaT
   pip install -e .

Authors
=======
Md Nurul Islam, Sean K. Martin, Shane M. O'Mara, and John P. Aggleton.


Contact
=======
Feel free to email us, or pop a message into our `Gitter channel <https://gitter.im/omaraneurolab/NeuroChaT>`_.


Contributing
============
We are open to contributions and would greatly appreciate any input. Please format your code before submitting a pull request. autopep8 has been run on neurochat. You can run this before contributing by the following:

.. code-block:: shell

   pip install autopep8
   python -m autopep8 -r -i neurochat


Getting started
===============
Sample hdf5 datasets and results are stored on OSF, at https://osf.io/kqz8b/files/.
Include the other things from Github README and point out guides here etc.
Also need to point out the User guide and any possible youtube videos.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   api_use_guide
   examples
   methods
   reference/neurochat
   roadmap
   changelog



Index
=====

* :ref:`genindex`
