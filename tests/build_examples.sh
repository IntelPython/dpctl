#!/bin/bash
set -xv
rm -rf *.o
base="${1%.*}"
gcc -g -DDEBUG -Wall -Wextra -Winit-self -Wuninitialized -Wmissing-declarations -std=c99 -fdiagnostics-color=auto -pedantic-errors -fPIC  -I../include ../source/opencllite.c ${1} -o $base.exe -lOpenCL
