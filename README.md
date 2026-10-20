# llmdbg

`llmdbg` is an early-stage debugger for stepping through language-model layers
and presenting human-readable interpretations of internal vectors. It speaks
the [Debug Adapter Protocol](https://microsoft.github.io/debug-adapter-protocol/)
so editor integrations can use a standard debugger UI.

The current prototype includes real GPT-2 Small loading, layer-state capture,
logit-lens interpretations, and a VS Code debug extension. A deterministic
built-in trace remains available when no model is configured.

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

Create a `gpt2.llmdbg` file in the workspace:

```toml
[model]
id = "openai-community/gpt2"
revision = "main"
device = "cuda"
dtype = "bfloat16"

[input]
prompt = "The capital of France is"

[debug]
stop_on_entry = true
```

Open that file and press **F5**. If VS Code asks for a debugger, choose
**llmdbg**. No `launch.json` is required. If a workspace launch configuration
is preferred, the complete minimal version is:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "type": "llmdbg",
      "request": "launch",
      "name": "llmdbg: Active file",
      "config": "${file}"
    }
  ]
}
```

On first use, the extension creates a private Python environment in VS Code's
extension storage. It does not install anything into the workspace. The first
GPT-2 run downloads the model into the normal Hugging Face cache. Use **Step
Over** to advance through embedding, attention, and MLP steps.

## License

MIT
