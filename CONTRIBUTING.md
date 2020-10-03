Diptorup Deb

Todd Anderson

## Check license

```bash
pip install apache-license-check
apache-license-check --copyright "Intel Corporation" LICENSE setup.py dpctl
```

Use `licenser` plugin for VSCode.


## Automate commit checks

Install:

```bash
pip install pre-commit
pre-commit install
```

Manual run:

```bash
pre-commit run  # diff
pre-commit run -a  # all
pre-commit run HOOK_ID -a  # only hook
```

Update:
```bash
pre-commit autoupdate
```


See: [.pre-commit-config.yaml]()
Commit without verification: `git commit --no-verify`

Checkers:
- black
- flake8
- clang-format
- cpplint
- cppcheck

Install tools via conda: `conda install TOOL -c defaults -c conda-forge`
Install cppcheck from [installer](http://cppcheck.sourceforge.net) for Windows.
