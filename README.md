# llmdbg

`llmdbg` is an early-stage debugger for stepping through language-model layers
and presenting human-readable interpretations of internal vectors. It speaks
the [Debug Adapter Protocol](https://microsoft.github.io/debug-adapter-protocol/)
so editor integrations can use a standard debugger UI.

The current prototype includes a deterministic model trace and a VS Code debug
extension. It demonstrates the complete editor-to-adapter path before support
for external model runtimes is added.

```console
pip install --pre llmdbg
llmdbg --version
```

## VS Code prototype

Build and install the extension:

```console
cd vscode-extension
npm install
npm run package
code --install-extension llmdbg-0.0.1.vsix
```

Install the Python package in the workspace environment:

```console
python -m venv .venv
.venv/bin/python -m pip install -e .
```

Open the repository in VS Code, select **llmdbg: Demo model** in **Run and
Debug**, and press **F5**. Use **Step Over** to advance one layer at a time.

## License

MIT
