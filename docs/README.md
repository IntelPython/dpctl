What?
=====

This README provides the steps to generate the documentation for both the
C and Python API of dpctl. The documentation of dpctl can be generated either
as a set of consolidated HTML pages containing both Python and C API, or
separate set of pages for C API and Python API. It is suggested that the
consolidated documentation is generated using the provided build scripts.

Prerequisite
============

The following tools are needed in order to build the documentation for dpctl

- `Sphinx`
- `Doxygen`
- `Doxyrest` [optional]
- `Lua` [optional]

`Doxyrest` and `Lua` are needed to generate `rst` files from the `Doxygen`
output and add them to a consolidate `Sphinx` generated site. It is preferred
that the latest `Doxyrest` binary is installed from
https://github.com/vovkos/doxyrest/tags.

`Lua` is required if using `Doxyrest`. Please follow your OS specific
instructions to install `liblua`. *E.g.*, on Ubuntu 20.04:

```
sudo apt-get install liblua5.2-dev
```

Generating the docs
===================

The documentation should be generated using the provided `Cmake` build script.
There are a few configurable options that can be used to select the type of
documentation to generate.

Build only Doxygen for C API
----------------------------
```bash
cd dpctl/docs
mkdir -p build
cd build
cmake ..
make Doxygen
```
The above steps will generate the `Doxygen` files at
`dpctl/docs/generated_docs/doxygen/html`. The documentation can also be
generated at a custom location by providing the optional flag

```bash
cd dpctl/docs
mkdir -p build
cd build
cmake .. -DDPCTL_DOCGEN_PREFIX=<WHERE_YOU_WANT_DOCS_TO_BE_GENERATED>
make Doxygen
```

Build only Sphinx for Python API
--------------------------------
```bash
cd dpctl/docs
mkdir -p build
cd build
cmake .. -DDPCTL_DOCGEN_PREFIX=<WHERE_YOU_WANT_DOCS_TO_BE_GENERATED>
make Sphinx
```

The `make Sphinx` command will generate only the Python API docs for dpctl.

Build consolidated docs
-----------------------
It is possible to generate a single site with both Python and C API docs. As
mentioned before, `Doxyrest` and `Lua` are required to generate the consolidated
site.

```bash
cd dpctl/docs
mkdir -p build
cd build
cmake ..                                                     \
  -DDPCTL_ENABLE_DOXYREST=ON                                 \
  -DDoxyrest_DIR=<PATH_TO_DOXYREST_INSTALL_DIR>              \
  -DDPCTL_DOCGEN_PREFIX=<WHERE_YOU_WANT_DOCS_TO_BE_GENERATED>
make Sphinx
```
The `Doxyrest_DIR` flag is optional, but is needed when Doxyrest is installed in
a non-system location.
