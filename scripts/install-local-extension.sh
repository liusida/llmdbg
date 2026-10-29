#!/usr/bin/env bash
# Build the current checkout as a pre-release VSIX and install into local VS Code.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
extension_root="$repo_root/vscode-extension"
vscode_cli="${VSCODE_CLI:-code}"

for executable in node npm "$vscode_cli"; do
    if ! command -v "$executable" >/dev/null 2>&1; then
        printf 'Required command not found: %s\n' "$executable" >&2
        exit 1
    fi
done
if [[ ! -x "$extension_root/node_modules/.bin/vsce" ]]; then
    printf 'Install build dependencies first: npm --prefix "%s" ci\n' "$extension_root" >&2
    exit 1
fi

version="$(node -e 'process.stdout.write(require(process.argv[1]).version)' "$extension_root/package.json")"
vsix="$repo_root/build/releases/vsix/llmdbg-$version-pre-release.vsix"

set -x
npm --prefix "$extension_root" run package:pre-release
"$vscode_cli" --install-extension "$vsix" --force
set +x
printf '\nInstalled %s. Reload the VS Code window and restart any active debug session.\n' "$vsix"
