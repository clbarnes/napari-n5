default:
    just --list

repl:
    uv run --all-groups --all-extras --with ipython ipython

fix:
    uv run --group dev ruff check --fix
    uv run --group dev ruff format

data:
    uv run --script data/generate_data.py

test:
    uv run --group dev pytest --verbose

lint:
    uv run --group dev ruff check src tests
    # uv run --group dev mypy src tests
    uv run --group dev ruff format --check src tests

napari:
    uv run --all-extras napari

# bump level:
#     test -z "$(git status --porcelain)" || ( git status && false )
#     git add .
#     git commit -m "Bump to v$(uv version --short)"
#     git tag -a "v$(uv version --short)" -m "Release v$(uv version --short)"
