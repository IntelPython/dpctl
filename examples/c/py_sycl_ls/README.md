# Python module to enumerate SYCL devices

## Building

```bash
python setup.py build_ext --inplace
```

The extension links against `libsyclinterface`, and on Windows,
`py_sycl_ls/__init__.py` must add it to the DLL search path before importing
the extension.

## Testing

```
pytest tests
```

## Running

```
python -m py_sycl_ls
```
