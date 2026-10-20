# llmdbg for VS Code

This extension connects VS Code's debugger UI to the bundled `llmdbg` Python
Debug Adapter Protocol implementation. On first use it creates a private Python
environment in VS Code's per-extension storage. It does not use or modify the
opened workspace's virtual environment.

## Prototype quick start

1. Install this extension's VSIX.
2. Open **Run and Debug**, select **llmdbg: Demo model**, and press **F5**.
3. Wait for the private debugger environment to be created on first use.
4. Expand **Variables** and use **Step Over** to advance through model layers.

The prototype runs a deterministic built-in model trace. At each stop, the
Variables view shows the current operation, a plain-language interpretation,
and named input/output vector dimensions. The Debug Console supports the watch
expressions `layer`, `operation`, `meaning`, `input`, and `output`.
