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

## Publish

1. Update `vscode-extension/package.json` and its lockfile to a version that has
   never been uploaded. Marketplace versions use `major.minor.patch`; suffixes
   such as `-beta` are not supported.
2. Commit and push the version.
3. In GitHub, open **Actions → Publish VS Code extension → Run workflow**.
4. Choose the channel the version policy below assigns: **release** for an
   `x.y.0` version, **pre-release** for any other patch version. (The workflow's
   default is pre-release, so change it for an `x.y.0` release.)

Pre-release users find `llmdbg` in VS Code, open its extension page, and choose
**Switch to Pre-Release Version**.

## Version policy

- Keep versions in `major.minor.patch` form. Do not add `-beta` or `-pre`.
- Never reuse a version number.
- **Major and minor versions (`x.y.0`) are stable releases** and go through the
  **release** channel.
- **Patch versions (`x.y.1`, `x.y.2`, ...) are pre-releases** and go through the
  **pre-release** channel. Normal users stay on the latest stable version; only
  users who opt into pre-releases get these builds.

Example: `0.2.0` is a stable release; `0.2.1` and `0.2.2` are pre-releases; the
next stable release is `0.3.0`. A version uploaded as a pre-release cannot later
be promoted under the same number.

## Local validation

The current version, 0.2.0, is a stable release. Build it without the
pre-release flag:

```console
cd vscode-extension
npm run package
```

The artifact is `build/releases/vsix/llmdbg-0.2.0.vsix`. For a patch
pre-release, use `npm run package:pre-release` instead; its artifact ends in
`-pre-release.vsix`. Publishing is a separate, explicit action: commit/push the
prepared version, then choose the matching channel in the workflow. Keep the
Python package and extension versions aligned in `pyproject.toml`,
`src/llmdbg/__init__.py`, `uv.lock`, and the extension manifest and lockfile.
Python/PyPI publication remains a separate explicit action. The Marketplace
pre-release status is a VSIX flag, not a version suffix; the shared source
version is `0.2.0`, which PyPI can take as is. Do not publish a patch
pre-release to PyPI without assigning an appropriate Python pre-release version
(e.g. `0.2.1a1`) first.

Standard release checks from the repository root:

```console
.venv/bin/python -m unittest discover -s tests -q
node --test vscode-extension/scripts/*.test.js
```

For major-release real-model validation, explicitly opt into the GPU starter suite:

```console
LLMDBG_RELEASE_TESTS=1 PYTHONPATH=tests:src .venv/bin/python -m unittest discover -s tests/integration -p test_starters_gpu.py -v
```

Create a VSIX without publishing it. For an `x.y.0` release:

```console
cd vscode-extension
npm ci
npm run package
```

The artifact is written to `build/releases/vsix/llmdbg-<version>.vsix`. For a
patch pre-release, run `npm run package:pre-release`, which adds the Marketplace
pre-release flag and writes `llmdbg-<version>-pre-release.vsix`.

For an emergency local PAT-based publish, create an Azure DevOps token with
**Marketplace (Manage)** scope across all accessible organizations, then run:

```console
cd vscode-extension
VSCE_PAT=... npx vsce publish --no-dependencies                 # x.y.0 release
VSCE_PAT=... npx vsce publish --pre-release --no-dependencies   # patch pre-release
```

Do not commit or print the token.
