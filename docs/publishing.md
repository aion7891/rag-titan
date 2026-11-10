# Publishing to PyPI

This page is for maintainers.

## One-time setup

1. Create accounts on [PyPI](https://pypi.org) and [TestPyPI](https://test.pypi.org).
2. On PyPI, open **Your projects → Publishing → Add a new pending publisher** and enter:
    - PyPI project name: `rag-titan`
    - Owner: `aion7891`, Repository: `rag-titan`
    - Workflow name: `publish.yml`, Environment: `pypi`
3. In the GitHub repo, open **Settings → Environments** and create an environment named `pypi`.

This uses *trusted publishing*, so no API token is stored in GitHub.

## Release steps

1. Update `__version__` in `src/rag_titan/__init__.py` (for example `0.1.1`).
2. Add the changes to `CHANGELOG.md`.
3. Commit and push to `main`.
4. On GitHub, create a **Release** with tag `v0.1.1`.
5. The **Publish to PyPI** workflow builds the package, runs the tests, and uploads it.

## Building and checking locally

```bash
pip install -e ".[dev]"
pytest
python -m build              # creates dist/*.whl and dist/*.tar.gz
twine check dist/*
```

## Manual upload (without GitHub Actions)

```bash
twine upload --repository testpypi dist/*   # try TestPyPI first
pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ rag-titan
twine upload dist/*                          # the real PyPI
```

## Docs website

The **Docs** workflow publishes this site to GitHub Pages on every push to `main`.
Enable it once in **Settings → Pages → Source: GitHub Actions**.
