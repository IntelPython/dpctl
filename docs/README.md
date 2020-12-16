What?
=====

Generator scripts for dpCtl API documentation. To run these scripts, follow
the following steps:

`mkdir build` <br/>
`cd build` <br/>
`cmake -DDPCTL_DOCGEN_PREFIX=<WHERE_YOU_WANT_DOCS_TO_BE_GENERATED>` <br/>
`make Sphinx` <br/>

The `DPCTL_DOCGEN_PREFIX` flag is optional and can be omitted to generate the
documents in the current source directory in a sub-directory called
`generated_docs`.

The `make Sphinx` command will generate standalone Doxygen documentation and
a consolidated Sphix documentation for both dpCtl Python and C APIs.

Prerequisite
============

Before you generate the documentation make sure you have the following
packages installed:

- sphinx
- doxygen
- breathe
- exhale