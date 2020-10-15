# Python code style

## black

We use [black](https://black.readthedocs.io/en/stable/) code formatter.

- Revision: `20.8b1` or branch `stable`.
- See configuration in `pyproject.toml`.

Run before each commit: `black .`

# C/C++ code style

## clang-format

We use [clang-format](https://clang.llvm.org/docs/ClangFormat.html) code formatter.

Install: `pip install clang`

- Revision: `9.0.1`
- See configuration in `.clang-format`. Created by: `$ clang-format -style=llvm -dump-config > .clang-format`

Run before each commit: `clang-format -style=file -i backends/include/*.h backends/include/Support/*.h backends/source/*.c backends/source/*.cpp backends/tests/*.cpp`
