# llmdbg agent guide

llmdbg is a debugger that steps through a language model's forward pass token
by token and layer by layer. A person runs it in VS Code; this guide is for an
agent (you) that has been asked to inspect or control their live session.

## 1. Find the session

Each live debug session writes one owner-only JSON file to
`$XDG_CACHE_HOME/llmdbg/sessions/` (default `~/.cache/llmdbg/sessions/`):

| field | meaning |
|---|---|
| `client.command`, `client.env` | the exact command (and `PYTHONPATH`) that runs the llmdbg client on this machine |
| `model`, `source` | model id and the `.llmdbg.py` program being debugged |
| `docs` | this guide |
| `host`, `port`, `token` | connection details; the client reads them for you |

**Never print, log, or send the token anywhere.** The client finds the newest
live session by itself.

## 2. Run commands

```console
PYTHONPATH=<client.env.PYTHONPATH> <client.command...> --client-name <your-name> --json -c where -c 'print x[0,0,5]'
```

- Use `--client-name` (e.g. `claude`, `codex`): the person sees it in their
  Debug Console and in the llmdbg Control panel next to everything you do.
- `--json` prints one JSON object per command on stdout (`{"command", "ok", "body" | "error"}`);
  progress goes to stderr.
- Each `-c` runs in order, then the client detaches. Saved vectors and edits
  stay in the session after you detach.
- Steps that run the model can take seconds to minutes; use a long timeout.

## 3. Commands

| command | effect |
|---|---|
| `where` | current token, layer/step, source line |
| `tokens` | prompt and generated tokens with positions |
| `vars [REF]` | debugger scopes and variables at this stop |
| `next`, `step`, `out`, `continue` | step over / into / out; run to the next breakpoint |
| `break LINE\|mlp\|attention [token=N,layer=N]`, `clear` | **replace** the file's breakpoints (the person sees them in VS Code) |
| `print EXPR` | evaluate, e.g. `print x[0,0,46]`, `print x.norm()`; slices return a tensor handle |
| `vector` | full residual stream at this stop as numbers: `body.vectors.residual_stream` (also `input`, `delta`) |
| `lens EXPR` | decode a model-width vector through Logit/J/R/J++ lenses |
| `vec NAME = EXPR`, `vec NAME from FILE`, `vecs` | save a named vector (from this stop, or .npy/.json/text) |
| `let x = EXPR`, `let x[...] = EXPR`, `let x *= 2` | rewrite the residual stream at this stop and replay the token |
| `let weight["<param>"][...] = EXPR`, `let W_o[...] *= 0` | edit weights for this debug session; replays from token 0 |
| `params FILTER`, `weights`, `undo weights [NAME]` | list parameters; list or undo weight edits |
| `set EXPR = VALUE` | set one value of any inspectable tensor, e.g. `set mlp_expansion[1843] = 0` |
| `request COMMAND JSON` | any DAP or `llmdbg/*` request (does not wait for a stop) |
| `help` | the full command list |

Expressions support `+ - * / @`, integer slices, the paused line's local
variables (`x`/`residual_stream`, `normalized`, `hidden`, `update`, ...),
`W_o`, `weight["name"]`, saved vectors, and `.norm() .sum() .mean() .abs()`.
They are tensor math only, not Python.

## 4. Rules of the road

- The person can see and undo everything you do. Say what you changed.
- Residual-stream writes apply to the **current token**, at a stop **before**
  that site executes. Weight edits apply to every token and are undone when the
  debug session ends.
- Ask before long runs (`continue` with no breakpoint can generate many tokens)
  or broad edits.
- Launching, stopping, and configuring the session belong to the person.
- The person can switch you to read-only or disconnect you from the Control panel.
  `Agent access is read-only` errors mean exactly that.

## 5. Example

```console
llmdbg() { PYTHONPATH=... python -m llmdbg "$@"; }   # from the session file's client fields
llmdbg debug --attach --client-name claude --json \
  -c tokens -c 'break 51 token=4,layer=5' -c continue \
  -c vector -c 'vec base = x' -c 'let x[0, 0, 46] = 0' -c 'lens x' -c 'let x = base'
```
