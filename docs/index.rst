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
The best ways to get started with NeuroChaT are:

For using the UI, download the `executable file for Windows <https://github.com/seankmartin/NeuroChaT/releases>`_ and check out the `user manual <https://github.com/seankmartin/NeuroChaT/blob/master/docs/NeuroChaT%20User%20Guide.pdf>`_.
For using the Python Code, check out the `introductory notebook <https://neurochat.readthedocs.io/en/latest/api_use_guide.html>`_, and `examples on this website <https://neurochat.readthedocs.io/en/latest/examples.html>`_ and `other examples on GitHub <https://github.com/seankmartin/NeuroChaT_API_Scripts>`_.
Additionally, Sample hdf5 datasets and results are stored on `OSF <https://osf.io/kqz8b/files/>`_.
We are open to collaborators, questions, etc. so feel free to get in touch!

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
