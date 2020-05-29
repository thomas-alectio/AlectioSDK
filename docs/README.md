To generate the documentation in html format simply run the following commands:

## Quick Start
```
cd docs
make clean
make html
```
The above commands should generate a bunch of things in the `docs/build` folder. Now, to open the webpage simply run:
```
open build/html/index.html
```

## Dependencies
Install Sphinx based on your system [here](https://www.sphinx-doc.org/en/master/usage/installation.html)

To download [napoleon](https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html) and the theme:
```
pip install sphinxcontrib-napoleon
pip install sphinx-rtd-theme
```
Other than that, you might run into issues but dependencies shouldn't be hard to install. Hard to state all the instructions here 
because some of it is system specific.

## References:
- Based on the famous template provided by [read the docs](https://sphinx-rtd-theme.readthedocs.io/en/stable/) 
