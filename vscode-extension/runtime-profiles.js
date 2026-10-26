"use strict";

const crypto = require("crypto");

function selectProfile(model = "", modelKind = "", override = "auto") {
  if (!["auto", "stable", "gemma4-preview"].includes(override)) throw new Error(`Unknown runtime profile: ${override}`);
  if (override !== "auto") return override;
  return modelKind === "gemma4" || /(?:^|\/)gemma-4-/i.test(model) ? "gemma4-preview" : "stable";
}

function profileRecipe(profile, backend, platform, arch) {
  if (!["stable", "gemma4-preview"].includes(profile)) throw new Error(`Unknown runtime profile: ${profile}`);
  const versions = backend === "cu130"
    ? (profile === "stable" ? ["2.10.0", "0.25.0"] : ["2.14.1", "0.29.1"])
    : backend === "cu118" ? ["2.7.1", "0.22.1"] : ["2.10.0", "0.25.0"];
  if (!["cu130", "cu128", "cu126", "cu118", "cpu", "native"].includes(backend)) throw new Error(`Unsupported torch backend: ${backend}`);
  const recipe = {
    profile, id: backend, platform, arch, python: "3.12", torch: versions[0], torchvision: versions[1],
    transformers: "5.20.2", hub: "1.33.0", safetensors: "0.8.0", matplotlib: "3.10.8", loader: "0.1.2",
    index: backend === "native" ? undefined : `https://download.pytorch.org/whl/${backend}`,
    schema: 1,
  };
  recipe.key = `${profile}-${platform}-${arch}-${backend}-${crypto.createHash("sha256").update(JSON.stringify(recipe)).digest("hex").slice(0, 12)}`;
  return recipe;
}

module.exports = { selectProfile, profileRecipe };
