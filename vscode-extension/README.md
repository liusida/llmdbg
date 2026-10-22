# llmdbg: Debug LLMs — Step, Inspect, Intervene, and Plot

Debug an LLM one token and one layer at a time.

Set breakpoints. Step through attention and MLP blocks. Inspect internal
vectors. Remove components and see what changes.

![llmdbg stepping through Qwen3.5 with Residual, MLP, and ICA Lens results](https://raw.githubusercontent.com/liusida/llmdbg/main/vscode-extension/assets/screenshots/lens-overview.png)

## Main features

- Step through an LLM forward pass in VS Code.
- Inspect the residual stream at each stage.
- View attention channels and MLP neurons.
- Use Logit Lens, J-Lens, R-Lens, and ICA Lens.
- See vector magnitude and direction.
- Remove an active component and continue the run.
- Save each intervention as a reproducible experiment log.
- Plot model vectors with Matplotlib.

## Get started

1. Install llmdbg.
2. Open a project folder in VS Code.
3. Press **Ctrl+P**.
4. Type **>llmdbg: Create Starter Files** and press Enter.
5. Open a generated `.llmdbg.py` example.
6. Press **F5**.
7. Set a breakpoint. Use **F10** to step.
8. Open **Lens** from the debug toolbar.

The first run may take a few minutes. llmdbg creates a private Python runtime.
It also downloads the selected model. It does not change your project's Python
environment.

llmdbg selects the compute backend automatically. It supports NVIDIA CUDA on
Windows and Linux. It supports MPS on Apple Silicon. CPU is the fallback.

If setup fails, run **llmdbg: Show Runtime Diagnostics**.

## Lenses and interventions

The Lens page follows the model's execution order. It shows the residual input,
attention update, mid-residual state, MLP update, and residual output.

ICA Lens shows the strongest independent components. Right-click an active ICA
component, attention channel, or MLP neuron. Choose **Intervention: Remove**.
llmdbg sets it to zero and recomputes the current state. You can then continue
the forward pass.

Each intervention creates a note in the project's `.logs` folder. The note
records the model, prompt, location, change, and result.

## Supported models

| Model | Debugging | ICA Lens |
| --- | :---: | :---: |
| GPT-2 Small | ✓ | ✓ |
| Qwen3.5 0.8B | ✓ | — |
| Qwen3.5 2B | ✓ | ✓ |
| Qwen3.5 9B | ✓ | ✓ |
| Qwen3.5 27B | ✓ | — |

J-Lens and R-Lens are available when an artifact exists for the selected model.
The official J++ Lens is available for Qwen3.5 9B.
Lens files are downloaded on first use and cached locally.

## Methods

- [Verbalizable Representations Form a Global Workspace in Language Models](https://transformer-circuits.pub/2026/workspace/index.html)
  introduces J-Lens.
- [R-Lens: Making J-Lens More Faithful on Early Layers](https://www.lesswrong.com/posts/nv8oedrnLXKRzNEL9/r-lens-making-j-lens-more-faithful-on-early-layers)
  introduces R-Lens.
- [J++ Lens: Jacobian Filtering Enables More Faithful Workspace Lenses](https://www.lesswrong.com/posts/nc9dHfcB22JdMzGbr/j-lens-jacobian-filtering-enables-more-faithful-workspace)
  introduces J++ Lens.
- [ICA Lens](https://arxiv.org/abs/2606.11722) introduces ICA Lens.

ICA bases:

- [GPT-2 Small](https://huggingface.co/sida/icalens-gpt2-small-pile10k)
- [Qwen3.5 2B](https://huggingface.co/sida/icalens-qwen3.5-2b-base-pile10k)
- [Qwen3.5 9B](https://huggingface.co/sida/icalens-qwen3.5-9b-base-pile10k)

## Links

- [Source code and issues](https://github.com/liusida/llmdbg)
- [Lens artifacts](https://huggingface.co/sida/llmdbg)
- [Sida Liu's homepage](https://liusida.com)

MIT License.
