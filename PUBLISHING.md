# Publishing the VS Code extension

The Marketplace identity is `SidaLiu.llmdbg`. Publishing is intentionally a
manual GitHub Actions operation, with tests and packaging checks preceding the
public upload.

## One-time Marketplace setup

1. Sign in at <https://marketplace.visualstudio.com/manage/publishers/SidaLiu>
   and confirm access to the publisher whose immutable ID is `SidaLiu`.
2. For that publisher, configure trusted publishing for:
   - GitHub owner: `liusida`
   - Repository: `llmdbg`
   - Workflow: `publish-vscode-extension.yml`
3. No `VSCE_PAT` repository secret is needed. The workflow exchanges GitHub's
   short-lived OIDC identity for a Marketplace publishing credential.

Only someone who controls the `SidaLiu` Marketplace publisher can complete the
one-time setup.

## Publish a pre-release

1. Update `vscode-extension/package.json` and its lockfile to a version that has
   never been uploaded. Marketplace versions use `major.minor.patch`; suffixes
   such as `-beta` are not supported.
2. Commit and push the version.
3. In GitHub, open **Actions → Publish VS Code extension → Run workflow**.
4. Choose **pre-release**.

Users can then find `llmdbg` in VS Code, open its extension page, and choose
**Switch to Pre-Release Version**.

## Version policy

- Keep versions in `major.minor.patch` form. Do not add `-beta` or `-pre`.
- Increment the patch number for every Marketplace upload; never reuse a
  version number.
- Publish development builds through the **pre-release** channel. Normal users
  stay on the latest stable version; only users who opt into pre-releases get
  these builds.
- Publish finished releases through the **release** channel using a newer,
  previously unused version.

Example: `0.1.0` is stable; publish `0.1.1` and `0.1.2` as pre-releases; then
publish `0.1.3` as the next stable release. A version uploaded as a pre-release
cannot later be promoted under the same number.

## Local validation

For the current 0.1.5 pre-release, build with the pre-release flag:

```console
cd vscode-extension
npm run package:pre-release
```

The artifact is `build/releases/vsix/llmdbg-0.1.5-pre-release.vsix`. Publishing is a
separate, explicit action: commit/push the prepared release, then choose **pre-release**
in the workflow. Keep the Python package and extension versions aligned in
`pyproject.toml`, `src/llmdbg/__init__.py`, `uv.lock`, and the extension manifest
and lockfile. Python/PyPI publication remains a separate explicit action.
The Marketplace pre-release status is a VSIX flag, not a version suffix; the
shared source version is `0.1.5`. Do not publish the Python wheel as a PyPI
pre-release without assigning an appropriate Python pre-release version first.

Standard release checks from the repository root:

```console
.venv/bin/python -m unittest discover -s tests -q
node --test vscode-extension/scripts/*.test.js
```

For major-release real-model validation, explicitly opt into the GPU starter suite:

```console
LLMDBG_RELEASE_TESTS=1 PYTHONPATH=tests:src .venv/bin/python -m unittest discover -s tests/integration -p test_starters_gpu.py -v
```

Create a VSIX carrying the Marketplace pre-release flag without publishing it:

```console
cd vscode-extension
npm ci
npm run package:pre-release
```

The artifact is written to
`build/releases/vsix/llmdbg-<version>-pre-release.vsix`.

For an emergency local PAT-based publish, create an Azure DevOps token with
**Marketplace (Manage)** scope across all accessible organizations, then run:

```console
cd vscode-extension
VSCE_PAT=... npx vsce publish --pre-release --no-dependencies
```

Do not commit or print the token.
