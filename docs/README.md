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

The helper script ``scripts/gen_docs.py`` is the preferred way to generate the
documentation. The generated documentation html pages will be installed to the
``CMAKE_INSTALL_PREFIX/docs`` directory.

----------------------------
```bash
python scripts/gen_docs.py --doxyrest-root=<PATH to Doxyrest installation>
```
To skip generating the documentation for ``libsyclinterface``, the
``--doxyrest-root`` option should be omitted.
