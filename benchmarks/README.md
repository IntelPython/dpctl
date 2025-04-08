# dpctl benchmarks

Benchmarking dpctl using Airspeed Velocity
Read more about ASV [here](https://asv.readthedocs.io/en/stable/index.html)

## Usage
The benchmarks were made with using an existing environment in-mind before execution. You will see the `asv.conf.json` is minimal without any environmental information supplied.
The expectation is for users to execute `asv run` with an existing environment.

As such, you should have conda or mamba installed, and create an environment [following these instructions](https://intelpython.github.io/dpctl/latest/beginners_guides/installation.html#dpctl-installation)
Additionally, install `asv` and `libmambapy` to the environment.

Then, you may activate the environment and instruct `asv run` to use this existing environment for the benchmarks by pointing it to the environment's python binary, like so:
```
conda activate dpctl_env
asv run --environment existing:/full/mamba/path/envs/dpctl_env/bin/python
```

For `level_zero` devices, you might see `USM Allocation` errors unless you use the `asv run` command with `--launch-method spawn`

## Writing new benchmarks
Read ASV's guidelines for writing benchmarks [here](https://asv.readthedocs.io/en/stable/writing_benchmarks.html)
