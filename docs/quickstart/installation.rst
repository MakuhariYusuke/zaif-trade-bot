Installation Guide
==================

This guide will help you install Zaif Trade Bot on your system.

Prerequisites
-------------

Zaif Trade Bot requires Python 3.11 or later. Make sure you have the following installed:

* Python 3.11+
* pip (latest version recommended)
* virtualenv or conda (recommended for environment management)

System Requirements
-------------------

* **RAM**: Minimum 8GB, recommended 16GB+
* **Storage**: Minimum 10GB free space
* **OS**: Windows 10+, macOS 10.15+, Ubuntu 18.04+

Installation Methods
--------------------

Method 1: Install from Source (Recommended for Development)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/MakuhariYusuke/zaif-trade-bot.git
   cd zaif-trade-bot

   # Create virtual environment
   python -m venv venv

   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate

   # Install in development mode
   pip install -e .

   # Install additional development dependencies (optional)
   pip install -e .[dev]

Method 2: Install with Docker
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/MakuhariYusuke/zaif-trade-bot.git
   cd zaif-trade-bot

   # Build the Docker image
   docker build -t zaif-trade-bot .

   # Run the container
   docker run -it zaif-trade-bot

Method 3: Install from PyPI (When Available)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

   # Install from PyPI
   pip install zaif-trade-bot

Verification
------------

After installation, verify that everything is working:

.. code-block:: bash

   # Check Python version
   python --version

   # Check if ztb package is available
   python -c "import ztb; print('Zaif Trade Bot installed successfully!')"

   # Check available commands
   ztb --help

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Import Error: No module named 'ztb'**

* Make sure you're in the correct virtual environment
* Try reinstalling: ``pip install -e .``

**Python Version Error**

* Ensure you're using Python 3.11 or later
* Check: ``python --version``

**Permission Errors on Windows**

* Run command prompt as Administrator
* Or use ``python -m venv venv`` instead of ``virtualenv``

**Docker Build Fails**

* Ensure Docker Desktop is running
* Check available disk space
* Try ``docker system prune`` to clean up

Next Steps
----------

Once installed, proceed to :doc:`basic-usage` to learn how to use Zaif Trade Bot.