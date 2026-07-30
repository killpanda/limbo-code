"""Limbo: a minimal terminal AI coding agent."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("limbo-code")
except PackageNotFoundError:  # not installed (e.g. running from a bare checkout)
    __version__ = "0.0.0.dev0"
