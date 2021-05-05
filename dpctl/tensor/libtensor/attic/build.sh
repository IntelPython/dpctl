g++ -Wall -shared -fPIC $(python3 -m pybind11 --includes) -I../include -L. -ltensor -Wl,-rpath,`pwd` ../attic/strided_utils.cpp -o strided_utils$(python3-config --extension-suffix)

g++ ../attic/a.cpp -L. -ltensor -Wl,-rpath,`pwd`
