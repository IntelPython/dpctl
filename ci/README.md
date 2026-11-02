# How Intel Python CI tests dpctl

Intel Python CI (gh-ci) builds and tests dpctl's conda packages and wheels in GitHub Actions. Intel's Jenkins then tests each package again on Intel machines, including GPU ones, and scans it, and on a release tag publishes the packages it tested as the tag's GitHub Release. Everything it needs from this repository is in this directory, one workflow, the recipe, and `pyproject.toml`.

| File | Role |
|---|---|
| `../conda-recipe/` | The conda recipe, built against conda-forge only. `conda_build_config.yaml` lists the Python versions, which has no 3.15t as conda-forge has no NumPy for it. The recipe skips macOS in `meta.yaml`, and its own tests only import dpctl and list the devices, so they need no device. |
| `../pyproject.toml` | What the wheels are built from. Its Python classifiers list the wheels' Python versions, and its Operating System classifiers their platforms. `[tool.cibuildwheel]` holds the wheel build settings, and `[tool.gh-ci]` holds the run-time requirements the wheels need on top of `[project]`'s. |
| `../.github/workflows/intel-ci.yml` | One job reads both matrices. One job per platform and Python version builds and uploads `dpctl <Platform> Python <version>`, and one builds `dpctl <Platform> Wheels Python <version>`. Then one job per package tests it. On a push to `master` or a tag, two more jobs upload the tested packages to the `dev` label of `dppy` on anaconda.org and remove its old ones. |
| `matrix.py` | Reads the platforms and Python versions from the recipe, or with `--system wheel` from `pyproject.toml`. A pull request gets the oldest and the newest regular versions. |
| `build.py` | Builds one Python version: with conda-build, which runs the recipe's own tests, or with `--system wheel` with cibuildwheel, in a manylinux container on Linux. |
| `hooks.py` | The workflow's test jobs: installs one package into a clean environment with the requirements in `config.yml`, and runs `test.py` the way Jenkins does. A conda package's environment comes from conda-forge; a wheel's has only Python from conda-forge and everything else from PyPI. |
| `install_icx.py` | Installs Intel's icx, which the conda recipe builds with too, for the Windows wheel builds: `[tool.cibuildwheel.windows]` in `../pyproject.toml` runs it first. The Linux wheel builds get icx from Intel's yum repository. Both get the Level Zero headers from a pinned tag of its GitHub repository. |
| `config.yml` | Tells gh-ci which workflow builds the packages, what its artifacts are called, and what the tests and benchmarks need. |
| `test.py` | Runs the installed package's test suite with pytest, in Actions and again in Jenkins. |
| `benchmark.py` | Runs the asv suite in `../benchmarks/` against the installed package, in Jenkins only, on a bare-metal benchmarking machine. Jenkins requires it because `../benchmarks/` exists. |

The scripts need only the Python standard library. `matrix.py`, `hooks.py` and `benchmark.py` are the same in every repository gh-ci tests, `build.py` adds a fix for the CRLF `METADATA` that setuptools writes on Windows, and `install_icx.py` and `test.py` are specific to dpctl. Scanning is not the repository's to define: Jenkins runs its own scans.

## Devices

dpctl's tests need a SYCL device, and GitHub's runners have none but a CPU. `test.py` prints the devices with `python -m dpctl -f` and fails if there is none, so a run that skipped every device test cannot pass.

- **In Actions** the device is the OpenCL CPU runtime of `intel-opencl-rt`, which `config.yml` installs from conda-forge for a conda package and from PyPI for a wheel. `test.py` points the OpenCL loader at it with `OCL_ICD_FILENAMES`, and on Windows also registers it under `HKLM\SOFTWARE\Khronos\OpenCL\Vendors`, as the loader in `System32` reads only that, which needs the administrator rights a GitHub runner gives.
- **In Jenkins** the GPU agents set `GH_CI_DEVICE=gpu`. Then `test.py` leaves the loader alone, apart from letting a Linux environment's loader see the system's ICDs too, and fails unless a GPU device is present.

Two tests compile an extension when they run, `test_c_headers` and `test_cython_api`. `test.py` deselects them, and says so, when setuptools finds no C and C++ compiler. A conda package's environment gets `c-compiler` and `cxx-compiler` from `config.yml`; a wheel's environment uses the machine's.

## Wheels

The wheels link the SYCL runtime that `intel-sycl-rt` and `intel-cmplr-lib-rt` ship, and find it through their RPATH, or the DLL search path on Windows, so `auditwheel` and `delvewheel` leave those libraries out. `[tool.gh-ci] wheel-requires` adds the two as requirements to each wheel, as the wheels on PyPI have. icx is pinned to 2026.2 to match.

## Uploads

A push to `master` or a tag, in `IntelPython/dpctl` only and after every test passed, uploads the conda packages and the wheels the workflow built to `dppy` on anaconda.org with the label `dev`, with the repository secret `ANACONDA_TOKEN`, and then removes the old dev packages. No other event or branch gets the secret.

**A pull request into `master`.** Actions builds the oldest and the newest regular Python versions on Linux and Windows, as a conda package and as a wheel, and tests each package. Jenkins notices the pull request within 3 minutes, waits for that whole run, and then tests and scans every package at once, and benchmarks the newest Linux conda package, sampling each benchmark once. Each cell posts its own status on the pull request, and `intel-ci/pr` sums them up.

**A release tag.** Actions builds every declared version, and Jenkins tests and scans all of them, and measures the benchmarks properly on the newest Linux conda package, before it publishes the release. A benchmark that fails fails the run; a slower one does not, as nothing compares the timings with a baseline yet.

To support another Python, add it to `conda-recipe/conda_build_config.yaml` for the conda packages and as a classifier in `pyproject.toml` for the wheels; nothing else lists versions. Try it locally from a conda base environment with conda-build, or with cibuildwheel for wheels:

```bash
python ci/matrix.py                           # every platform and Python version
python ci/matrix.py --event pull_request      # what a pull request builds
python ci/build.py --python VERSION --output /tmp/out     # VERSION: one matrix.py lists
python ci/hooks.py --python VERSION --artifacts /tmp/out --work /tmp/hooks
python ci/build.py --system wheel --python VERSION --output /tmp/wheels
python ci/hooks.py --kind wheel --python VERSION --artifacts /tmp/wheels --work /tmp/wheel-hooks
```
