# Example illustrating use of dpctl.sycl in Cython

Dpctl include `dpctl/sycl.pxd` file with incomplete definitions
of SYCL runtime classes and conversion routines from SYCLInterface
library opaque pointers to pointers to these SYCL classes.

This files simplifies usage of SYCL routines from Python extensions
written in Cython.

## Building

```bash
python setup.py develop
```

## Testing

```bash
python -m pytest tests
```
