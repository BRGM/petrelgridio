import verstr

try:
    from . import version

    __version__ = verstr.verstr(version.version)
except ImportError:
    __version__ = None
