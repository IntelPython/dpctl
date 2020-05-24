from ._version import get_versions
from ._oneapi_interface import *
__version__ = get_versions()['version']
del get_versions
