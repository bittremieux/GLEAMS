pkg_name = 'gleams'

try:
    from importlib.metadata import version, PackageNotFoundError

    try:
        __version__ = version(pkg_name)
    except PackageNotFoundError:
        __version__ = 'unspecified'
except ImportError:
    from pkg_resources import get_distribution, DistributionNotFound

    try:
        __version__ = get_distribution(pkg_name).version
    except DistributionNotFound:
        __version__ = 'unspecified'
