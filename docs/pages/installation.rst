Installation
============


From PyPI
---------

To install the latest release version of the SpiceX package, run the following command:

.. code-block:: bash

    pip install spicex

For GPU support, use the following command instead:

.. code-block:: bash

    pip install spicex[cuda12]

.. note::
    For now, to build with GPU support, use the build from source method below. This will be simplified in future releases.


Build from Source
-----------------

Check out the repository:

.. code-block:: bash

    git clone git@github.com:SpiceXProject/spicex.git

Navigate to the project directory:

.. code-block:: bash

    cd spicex

Install the package using pip (CPU version):

.. code-block:: bash

    pip install .

For GPU support, use the following command instead:

.. code-block:: bash

    pip install .[cuda12]

Verify the installation by running the test suite:

.. code-block:: bash

    pytest
