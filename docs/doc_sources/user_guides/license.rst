.. _user_guide_dpctl_license:

License
=======

:py:mod:`dpctl` is licensed under Apache License 2.0 that can be found in
`LICENSE <dpctl_license_>`_ file.
All usage and contributions to the project are subject to the terms and
conditions of this license.

Third party components and their licenses
-----------------------------------------

:py:mod:`dpctl` vendors DLPack header file which governed by Apache 2.0 license
that can be found in its `LICENSE <dlpack_license_>`_ vendored file. DLPack header
is used to implement support for data interchanging mechanism in :py:mod:`dpctl.tensor`
as required by Python Array API specification, cf. `data interchange document <array_api_data_interchange_>`_.

:py:mod:`dpctl` vendors `versioneer <versioneer_gh_>`_ to generate it version from git history
of its sources. Versioneer has been placed in public domain per `license file <versioneer_license_>`_
in its original repository.


.. _dpctl_license: https://github.com/IntelPython/dpctl/blob/master/LICENSE
.. _dlpack_license: https://github.com/IntelPython/dpctl/blob/master/dpctl/tensor/include/dlpack/LICENSE.third-party
.. _versioneer_license: https://github.com/python-versioneer/python-versioneer/blob/master/LICENSE
.. _versioneer_gh: https://github.com/python-versioneer/python-versioneer/
.. _array_api_data_interchange: https://data-apis.org/array-api/latest/design_topics/data_interchange.html
