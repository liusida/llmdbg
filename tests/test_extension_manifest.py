from __future__ import annotations

import json
import unittest
from pathlib import Path


class ExtensionManifestTest(unittest.TestCase):
    def test_debugger_activation_and_contribution_match(self) -> None:
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "vscode-extension" / "package.json").read_text())

        self.assertIn("onDebug", manifest["activationEvents"])
        debugger_types = {debugger["type"] for debugger in manifest["contributes"]["debuggers"]}
        self.assertIn("llmdbg", debugger_types)
        self.assertEqual(manifest["main"], "./extension.js")
        self.assertIn("bundled/**", manifest["files"])
        self.assertIn("stage-adapter", manifest["scripts"])
        self.assertEqual(manifest["contributes"]["languages"][0]["extensions"], [".llmdbg"])
        self.assertEqual(manifest["contributes"]["languages"][1]["extensions"], [".llmdbg.py"])
        self.assertEqual(manifest["contributes"]["breakpoints"][0]["language"], "llmdbg")
        self.assertEqual(manifest["contributes"]["breakpoints"][1]["language"], "llmdbg-python")
        command_ids = {command["command"] for command in manifest["contributes"]["commands"]}
        self.assertIn("llmdbg.plotCurrentVector", command_ids)
        self.assertIn("llmdbg.showNeuronLens", command_ids)
        self.assertIn("llmdbg.unloadModel", command_ids)
        self.assertIn("llmdbg.showModelStatus", command_ids)

    def test_extension_uses_a_private_runtime(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "vscode-extension" / "extension.js").read_text()

        self.assertIn("context.globalStorageUri.fsPath", source)
        self.assertIn('["-m", "llmdbg.host"', source)
        self.assertIn('new vscode.DebugAdapterServer(port, "127.0.0.1")', source)
        self.assertIn('"gb10-load-llm==0.1.2"', source)
        self.assertIn('["venv", "--python", bootstrapPython(), runtimeRoot]', source)
        self.assertIn('"pip", "install"', source)
        self.assertIn('get("uvPath", "uv")', source)
        self.assertIn("session.configuration.adapterSource", source)
        self.assertIn("registerDebugConfigurationProvider", source)
        self.assertIn("activeLlmdbgFile", source)
        self.assertIn('customRequest("llmdbg/plot"', source)
        self.assertIn('customRequest("llmdbg/neuronLens"', source)
        self.assertIn("context.globalState.get(LEGACY_PLOT_SNIPPETS_KEY", source)
        self.assertIn('file.endsWith(".plot.py")', source)
        self.assertIn("function plotFunctionCode(code)", source)
        self.assertIn("def plot(vector, vector_name, plt, np):", source)
        self.assertIn("vscode.workspace.openTextDocument", source)
        self.assertIn("viewColumn: vectorPanel?.viewColumn || vscode.ViewColumn.Beside", source)
        self.assertIn('"matplotlib>=3.10,<4"', source)
        self.assertIn('registerCommand("llmdbg.plotCurrentVector"', source)
        self.assertIn('registerCommand("llmdbg.showNeuronLens"', source)
        self.assertIn('"llmdbg.neuronLens", "llmdbg Lens"', source)
        self.assertIn("void showNeuronLens(session)", source)
        self.assertIn("preserveFocus: true", source)
        self.assertNotIn("llmdbg Neuron Lens", source)
        self.assertIn('registerCommand("llmdbg.unloadModel"', source)
        self.assertIn('registerCommand("llmdbg.showModelStatus"', source)
        self.assertIn("showInformationMessage(message, \"Unload Model\")", source)
        self.assertIn("createStatusBarItem", source)
        self.assertIn('get("autoReloadAfterUpdate", true)', source)
        self.assertIn('executeCommand("workbench.action.reloadWindow")', source)
        self.assertIn("vscode.extensions.onDidChange", source)
        self.assertIn("newestInstalledVersion(context)", source)
        self.assertIn("fs.promises.readdir(extensionParent", source)
        self.assertIn("compareVersions(installedVersion, runningVersion) > 0", source)
        self.assertIn('event.event === "llmdbg/modelLoaded"', source)
        self.assertIn("modelBytes", source)
        self.assertIn("cudaAllocatedBytes", source)
        self.assertNotIn('path.join(".venv"', source)

    def test_neuron_lens_keeps_selected_mode_across_debug_stops(self) -> None:
        root = Path(__file__).resolve().parents[1]
        source = (root / "vscode-extension" / "extension.js").read_text()

        self.assertIn('let neuronLensMode = "j-lens"', source)
        self.assertIn('let neuronLensView = "all"', source)
        self.assertIn(
            'async function refreshNeuronLens(mode = neuronLensMode, view = neuronLensView)',
            source,
        )
        self.assertIn("neuronLensMode = message.mode", source)
        self.assertIn("neuronLensView = message.view", source)
        self.assertIn('<option value="mlp-output">MLP Output Lens</option>', source)
        self.assertIn('<option value="attention-output">Attention Output Lens</option>', source)
        self.assertIn('<option value="attention-neurons">Attention Output Channel Lens</option>', source)
        self.assertIn('<option value="neurons">MLP Neuron Lens</option>', source)
        expected_lenses = (
            '<option value="residual-pre">Residual Lens · Pre</option>',
            '<option value="attention-neurons">Attention Output Channel Lens</option>',
            '<option value="attention-output">Attention Output Lens</option>',
            '<option value="residual-mid">Residual Lens · Mid</option>',
            '<option value="neurons">MLP Neuron Lens</option>',
            '<option value="mlp-output">MLP Output Lens</option>',
            '<option value="residual-post">Residual Lens · Post</option>',
        )
        positions = [source.index(option) for option in expected_lenses]
        self.assertEqual(positions, sorted(positions))
        self.assertNotIn("saved from layer ", source)
        self.assertIn("' [layer '+item.transformerLayer+']'", source)
        self.assertIn("item.view.startsWith('residual-')?'|x| ':'|Δx| '", source)
        self.assertIn("item.vectorNorm.toFixed(2)", source)
        self.assertIn("item.transformerLayer!==activeTransformerLayer", source)
        self.assertIn("#context:empty{display:none}", source)
        self.assertIn("context.textContent=''", source)
        self.assertIn("opacity:.6", source)
        self.assertIn('<option value="all" selected>All</option>', source)
        self.assertIn('if (view === "all")', source)
        self.assertIn('<option value="j-lens" selected>J-Lens</option>', source)
        self.assertIn('<option value="all">All lenses</option>', source)
        self.assertIn("chip.title='rank '+item.rank+' · token ID '+item.id+' · logit '", source)
        self.assertIn("title.className='neuron-badge'", source)
        self.assertIn("title.textContent=String(neuron.index)", source)
        self.assertIn("norm.textContent='|d| '+neuron.directionNorm.toFixed(2)", source)
        self.assertIn("impact.textContent='|Δx| '+neuron.contributionMagnitude.toFixed(2)+relative", source)
        self.assertIn("activation.textContent='a '+neuron.activation.toFixed(2)", source)
        self.assertIn("activation coefficient a = ", source)
        self.assertIn("output direction norm |d| = ", source)
        self.assertIn("head.append(title,impact,activation,norm)", source)
        self.assertIn(".card.full-width{grid-column:1/-1}", source)
        self.assertIn("card.className='card full-width'", source)
        self.assertIn(".residual-triangle{", source)
        self.assertIn(".triangle-input{stroke:#5aa7e8;stroke-dasharray:3 2}", source)
        self.assertIn("item.view.startsWith('residual-')", source)
        self.assertIn("residualTriangle(item.view,item.residualGeometry)", source)
        self.assertIn("'Dashed blue: residual input x'", source)
        self.assertIn("unitLength=inputTip[0]-origin[0]", source)
        self.assertNotIn("scale=Math.min(28/", source)
        self.assertIn("cos(x, Δx)", source)
        self.assertIn("Dashed blue: residual input x", source)
        self.assertIn("Orange: update Δx", source)
        self.assertIn("Teal: residual output x + Δx", source)
        self.assertIn("styleCard(card,item);addReadouts(card,item);neurons.append(card);return", source)
        self.assertNotIn("title.textContent=item.vectorLabel", source)
        self.assertIn("neuron.relativeContribution??neuron.relativeActivation", source)
        self.assertIn("residual-contribution rank", source)
        self.assertIn("heading.textContent=title+':'", source)
        self.assertIn(".readout{display:flex;align-items:center;flex-wrap:wrap", source)
        self.assertNotIn("className='token-list'", source)
        self.assertIn("white-space:pre-wrap;overflow-wrap:anywhere", source)
        self.assertIn("padding:3px 7px 2px", source)
        self.assertNotIn("text-overflow:ellipsis", source)
        self.assertNotIn("table.innerHTML='<thead><tr><th", source)
        self.assertIn('event.event === "llmdbg/vectorChanged"', source)
        self.assertIn("card.classList.add('saved')", source)
        self.assertIn("repeating-linear-gradient", source)
        self.assertIn("rgba(0,0,0,.24)", source)
        self.assertNotIn("Saved result from an earlier execution position", source)
        self.assertIn('type: "lens-error"', source)
        self.assertIn('type: "reset"', source)


if __name__ == "__main__":
    unittest.main()
