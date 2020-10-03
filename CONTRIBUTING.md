Diptorup Deb

Todd Anderson

## Check license

```bash
pip install apache-license-check
apache-license-check --copyright "Intel Corporation" LICENSE setup.py dpctl
```

Use `licenser` plugin for VSCode.


## Automate commit checks

```bash
pip install pre-commit
pre-commit install
```

See: [.pre-commit-config.yaml]()
Commit without verification: `git commit --no-verify`

Checkers:
- black
- flake8
- clang-format
- cpplint
- cppcheck

Install tools for C++ checkers: `conda install clang-format cpplink cppcheck -c defaults -c conda-forge`
