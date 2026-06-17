default:
    just --list

repl:
    uv run --all-groups --all-extras --with ipython ipython

fix:
    uv run --group dev ruff check --fix
    uv run --group dev ruff format

data:
    uv run --script data/generate_data.py

test: data
    uv run --group dev pytest --verbose

napari:
    uv run --all-extras napari
