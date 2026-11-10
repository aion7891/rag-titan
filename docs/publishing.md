# Publishing to PyPI

This page is for maintainers. Releases use *trusted publishing*, so no API
token is stored in GitHub.

## How it works

| You do                                            | The **Publish** workflow does         |
|---------------------------------------------------|---------------------------------------|
| Actions → Publish → **Run workflow**              | test, build, upload to **TestPyPI**   |
| Publish a **GitHub Release** (tag `v0.1.0`)       | test, build, **TestPyPI**, then **PyPI** |

## One-time setup

On **both** [PyPI](https://pypi.org/manage/account/publishing/) and
[TestPyPI](https://test.pypi.org/manage/account/publishing/), add a
*pending publisher* with these values:

| Field             | PyPI           | TestPyPI       |
|-------------------|----------------|----------------|
| PyPI project name | `rag-titan`    | `rag-titan`    |
| Owner             | `aion7891`     | `aion7891`     |
| Repository name   | `rag-titan`    | `rag-titan`    |
| Workflow name     | `publish.yml`  | `publish.yml`  |
| Environment name  | `pypi`         | `testpypi`     |

!!! warning "The values must match exactly"
    If the environment name on PyPI is different from the one in
    `.github/workflows/publish.yml`, the upload fails with
    `invalid-publisher`.

GitHub creates the `pypi` and `testpypi` environments the first time the
workflow runs. To ask for approval before the real upload, open
**Settings → Environments → pypi** and add yourself as a *required reviewer*.

## Release steps

1. Update `__version__` in `src/rag_titan/__init__.py` (for example `0.1.1`).
2. Add the changes to `CHANGELOG.md`.
3. Merge to `main`.
4. (Optional) Run the **Publish** workflow by hand and check the package on TestPyPI:
   ```bash
   pip install -i https://test.pypi.org/simple/ \
       --extra-index-url https://pypi.org/simple/ rag-titan
   ```
5. On GitHub, create a **Release** with tag `v0.1.1`. The package goes to PyPI.

!!! note "Versions are final"
    PyPI and TestPyPI never accept the same version twice, even after deleting it.
    Always increase `__version__` for a new upload.

## Building and checking locally

```bash
pip install -e ".[dev]"
pytest
python -m build              # creates dist/*.whl and dist/*.tar.gz
twine check dist/*
```

## Docs website

The **Docs** workflow publishes this site to GitHub Pages on every push to `main`.
Enable it once in **Settings → Pages → Source: GitHub Actions**.
