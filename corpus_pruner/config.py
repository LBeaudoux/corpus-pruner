try:
    from importlib.resources import files
except ImportError:
    # Compatibility for Python <3.9
    from importlib_resources import files


DATA_DIR = files(__package__).joinpath("data")
DATA_DIR.mkdir(parents=True, exist_ok=True)
