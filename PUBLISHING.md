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

The same numeric version cannot later be uploaded as a regular release. After
publishing `0.0.117` as a pre-release, use at least `0.0.118` for the regular
release.

## Local validation

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
