##===---------- t_is_in_dppl_ctxt.py - dppl  -------------*- Python -*-----===##
##
##               Python Data Parallel Processing Library (PyDPPL)
##
## Copyright 2020 Intel Corporation
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##    http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##
##===----------------------------------------------------------------------===##
###
### \file
### This file has unit test cases to check the dump functions provided by dppl.
##===----------------------------------------------------------------------===##

from contextlib import contextmanager
import ctypes
import dppl
import io
import os, sys
import tempfile
import unittest


libc = ctypes.CDLL(None)
c_stdout = ctypes.c_void_p.in_dll(libc, 'stdout')

# Sourced from
# https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
@contextmanager
def stdout_redirector (stream):
    # The original fd stdout points to. Usually 1 on POSIX systems.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, 'wb'))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode='w+b')
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        if stream:
            stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)


class TestDPPLDumpFns(unittest.TestCase):

    def test_dppl_dump_runtime(self):
        with stdout_redirector(None):
            self.assertEqual(dppl.dump(), 0)
    
    def test_dppl_dump_queue_info(self):
        with stdout_redirector(None):
            q = dppl.get_current_queue()
            self.assertEqual(dppl.dump_queue_info(q), 0)


suite = unittest.TestLoader().loadTestsFromTestCase(TestDPPLDumpFns)
unittest.TextTestRunner(verbosity=2).run(suite)
