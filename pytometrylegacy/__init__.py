try:
    from importlib import metadata
except ImportError:  # for Python<3.8
    import importlib_metadata as metadata

__version__ = metadata.version('pytometrylegacy')

from . import converter as io
from . import clustering as cl
from . import preprocessing as pp
from . import tools as tl
