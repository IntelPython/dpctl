# Working with USM Memory

This example demonstrates building of an extension that works with
`dpctl.tensor.usm_ndarray` container.

It implements two Python functions: `blackscholes.populate_params` and
`blackscholes.black_scholes_price`. The first one uses MKL's device RNG
implementation to populate option parameters from uniform distribution
in user-specified ranges, and the other one takes the array with option
parameters and produces array with call and put European vanilla option
prices.

## Building

> **NOTE:** Make sure oneAPI is activated, $ONEAPI_ROOT must be set.

To build the example, run:
```
$ python setup.py build_ext --inplace
```

## Testing

```
$ pytest tests/
```

## Running benchmark

```
$ python scripts/bench.py
```

It gives the example output:

```
(dev_dpctl) opavlyk@opavlyk-mobl:~/repos/dpctl/examples/cython/usm_memory$ python scripts/bench.py
Pricing 30,000,000 vanilla European options using Black-Scholes-Merton formula

Using      : 11th Gen Intel(R) Core(TM) i7-1185G7 @ 3.00GHz
Wall times : [0.07042762002674863, 0.047108696977375075, 0.04325491201598197, 0.045397296984447166, 0.0433025429956615] for dtype=float32
Using      : Intel(R) Graphics [0x9a49]
Wall times : [0.1194021370029077, 0.0720841379952617, 0.0647223969863262, 0.06645121600013226, 0.06911522900918499] for dtype=float32
```
