# Rebuilding SPIR-V file from source

```bash
export TOOLS_DIR=$(dirname $(dirname $(which icx)))/bin-llvm
$TOOLS_DIR/clang -cc1 -triple spir double_it.cl -finclude-default-header -flto -emit -llvm-bc -o double_it.bc
$TOOLS_DIR/llvm-spirv double_it.bc -o double_it.spv
rm double_it.bc
```
