repl:
    uv run --all-groups --all-extras --with ipython ipython

fix:
    uv run --group dev ruff check --fix
    uv run --group dev ruff format

data:
    uv run --script data/generate_base.py

test:
    uv run --group dev pytest --verbose