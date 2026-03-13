Contributing Guide
=================

We welcome contributions to Zaif Trade Bot! This guide will help you get started with contributing to the project.

.. toctree::
   :maxdepth: 2
   :caption: Contributing:

   setup
   development-workflow
   testing
   code-style
   documentation
   requirements.txt
   requirements-dev.txt
   constraints.txt
   mypy_errors.txt
   type_ignore_usage.txt

Development Setup
-----------------

Setting up your development environment:

.. code-block:: bash

   # Fork and clone the repository
   git clone https://github.com/your-username/zaif-trade-bot.git
   cd zaif-trade-bot

   # Install development dependencies
   pip install -e ".[dev]"

   # Install pre-commit hooks
   pre-commit install

Code Style
----------

We use the following tools for code quality:

* **Black** for code formatting
* **isort** for import sorting
* **flake8** for linting
* **mypy** for type checking

Run all checks:

.. code-block:: bash

   # Format code
   black .
   isort .

   # Lint and type check
   flake8 .
   mypy .

Development Workflow
-------------------

1. Create a feature branch from ``main``
2. Make your changes
3. Run tests and checks
4. Submit a pull request

Testing
-------

Run the test suite:

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=ztb --cov-report=html

   # Run specific test
   pytest tests/test_backtest.py

Documentation
-------------

Build the documentation:

.. code-block:: bash

   cd docs
   make html

View the documentation at ``docs/_build/html/index.html``.

Guidelines
----------

* Write clear, concise commit messages
* Add tests for new features
* Update documentation as needed
* Follow the existing code style
* Use type hints for new code

Need Help?
----------

* Check existing issues on GitHub
* Join our Discord community
* Read the full documentation
