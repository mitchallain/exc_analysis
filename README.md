# Python Packaging Template
Duplicate the parent directory to quickly create a working Python package that can be install in development mode with:

```bash
pip install -e /path/to/mypackage
```
where -e stands for --editable. This creates a symlink rather than placing the package in the standard path.

Below is a condensed version of [this site](https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/quickstart.html#lay-out-your-project)

## Steps to Lay out the project
- Name the project and modify both directories `pytemplate/` and `pytemplate/pytemplate/` to reflect the name.
- Update the setup.py metadata file
    + **name**, you may want this to be unique on the Python Package Index (PyPI), if you plan to share it later on PyPI
    + **version**, see [semantic versioning](http://semver.org)
    + **packages** describes where you've put the source code
- Add an appropriate LICENSE.txt file
- Update the README.md file
- If your project is already fairly well-developed, see ['Arranging your file and directory structure'](https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/creation.html#directory-layout)
- Otherwise, continue on

## Create first release
- Use version *0.1dev* to indicate a dev version.
- When ready to release, run:
```bash
$ python setup.py sdist
```
  to create a tarball of the source code in the dist directory of the project
**continue here later**

## Register your package with PyPI

# References
[1] [The Hitchhiker's Guide to Packaging](https://the-hitchhikers-guide-to-packaging.readthedocs.io/en/latest/index.html)
