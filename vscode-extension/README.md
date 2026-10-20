# llmdbg for VS Code

This extension connects VS Code's debugger UI to the `llmdbg` Python Debug
Adapter Protocol implementation.

## Prototype quick start

1. Install `llmdbg` into the workspace's `.venv`.
2. Install this extension's VSIX.
3. Open **Run and Debug**, select **llmdbg: Demo model**, and press **F5**.
4. Expand **Variables** and use **Step Over** to advance through model layers.

The prototype runs a deterministic built-in model trace. At each stop, the
Variables view shows the current operation, a plain-language interpretation,
and named input/output vector dimensions. The Debug Console supports the watch
expressions `layer`, `operation`, `meaning`, `input`, and `output`.
