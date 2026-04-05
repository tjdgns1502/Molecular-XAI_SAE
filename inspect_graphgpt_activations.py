#!/usr/bin/env python3
"""Step-1 GraphGPT activation inspection on a small PCQM4Mv2 subset.

This script is intentionally strict about activation provenance:
- It discovers the GraphGPT model class from official repository layout.
- It uses an adapter interface for graph->model inputs + token_to_node mapping.
- It inspects the final two transformer blocks and hidden_states outputs.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import random
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch


@dataclass
class AdapterBatch:
    model_inputs: Dict[str, torch.Tensor]
    token_to_node: Optional[torch.Tensor]


AdapterFn = Callable[[Sequence[Dict[str, Any]], torch.device, Any], AdapterBatch]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_pcqm4mv2_dataset(dataset_root: str):
    try:
        from ogb.lsc import PCQM4Mv2

        return PCQM4Mv2(root=dataset_root)
    except Exception:
        from ogb.lsc import PCQM4Mv2Dataset

        return PCQM4Mv2Dataset(root=dataset_root)


def import_graphgpt_modules(graphgpt_repo: Optional[str]) -> Tuple[Any, Any]:
    if graphgpt_repo:
        sys.path.insert(0, graphgpt_repo)

    cfg_mod = importlib.import_module("src.models.graphgpt.configuration_graphgpt")
    model_mod = importlib.import_module("src.models.graphgpt.modeling_finetune")
    GraphGPTConfig = getattr(cfg_mod, "GraphGPTConfig")

    for name in ("GraphGPTTaskModel", "GraphGPTDoubleHeadsModel"):
        if hasattr(model_mod, name):
            return GraphGPTConfig, getattr(model_mod, name)

    raise RuntimeError("Could not find GraphGPT finetune model class in modeling_finetune.py")


def load_model(graphgpt_repo: Optional[str], checkpoint: Optional[str], device: torch.device):
    GraphGPTConfig, model_cls = import_graphgpt_modules(graphgpt_repo)
    if checkpoint:
        config = GraphGPTConfig.from_pretrained(checkpoint)
        model = model_cls.from_pretrained(checkpoint, config=config)
    else:
        config = GraphGPTConfig()
        model = model_cls(config)

    model = model.to(device)
    model.eval()
    return model


def builtin_node_token_adapter(graphs: Sequence[Dict[str, Any]], device: torch.device, model: Any) -> AdapterBatch:
    """Builtin fallback adapter.

    WARNING: This is not GraphGPT's official Eulerian tokenization.
    It is provided only so the activation pipeline can be exercised.
    """
    _ = model
    pad_id = 0
    seqs: List[List[int]] = []
    maps: List[List[int]] = []
    for g in graphs:
        node_feat = g["node_feat"]
        ids = [int(x) for x in node_feat[:, 0].tolist()]
        seqs.append(ids)
        maps.append(list(range(len(ids))))

    max_len = max(len(x) for x in seqs)
    input_ids, attention_mask, token_to_node = [], [], []
    for ids, idxmap in zip(seqs, maps):
        pad = max_len - len(ids)
        input_ids.append(ids + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
        token_to_node.append(idxmap + [-1] * pad)

    return AdapterBatch(
        model_inputs={
            "input_ids": torch.tensor(input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device=device),
        },
        token_to_node=torch.tensor(token_to_node, dtype=torch.long, device=device),
    )


def load_adapter(adapter_path: Optional[str], function_name: str) -> AdapterFn:
    if adapter_path is None:
        return builtin_node_token_adapter

    p = Path(adapter_path)
    if not p.exists():
        raise FileNotFoundError(f"Adapter file not found: {p}")

    spec = importlib.util.spec_from_file_location("graphgpt_user_adapter", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to import adapter from {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    fn = getattr(mod, function_name, None)
    if fn is None:
        raise RuntimeError(f"Adapter function '{function_name}' not found in {p}")
    return fn


def pick_last_two_transformer_blocks(model: torch.nn.Module) -> List[Tuple[str, torch.nn.Module]]:
    blocks: List[Tuple[str, torch.nn.Module]] = []
    for name, module in model.named_modules():
        if re.search(r"(?:^|\.)layers\.\d+$", name):
            blocks.append((name, module))
    return blocks[-2:]


def tensor_sample(t: torch.Tensor, n: int = 8) -> List[float]:
    flat = t.detach().float().reshape(-1)
    n = min(n, flat.numel())
    return flat[:n].cpu().tolist()


def print_stats(tag: str, t: torch.Tensor) -> None:
    ft = t.detach().float()
    print(
        f"{tag}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"mean={ft.mean().item():.6f} std={ft.std().item():.6f} sample={tensor_sample(ft)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphgpt-repo", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--dataset-root", type=str, default="./data")
    ap.add_argument("--subset", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adapter", type=str, default=None, help="Python adapter path")
    ap.add_argument("--adapter-fn", type=str, default="build_batch")
    ap.add_argument("--output", type=str, default="artifacts/graphgpt_smoke_batch.pt")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_pcqm4mv2_dataset(args.dataset_root)
    idxs = list(range(len(dataset)))
    random.Random(args.seed).shuffle(idxs)
    idxs = idxs[: args.subset]

    model = load_model(args.graphgpt_repo, args.checkpoint, device)
    adapter_fn = load_adapter(args.adapter, args.adapter_fn)

    print("=== MODULE NAMES (first 120) ===")
    module_names = [name for name, _ in model.named_modules()]
    for name in module_names[:120]:
        print(name)
    if len(module_names) > 120:
        print(f"... ({len(module_names) - 120} more)")

    hook_cache: Dict[str, torch.Tensor] = {}
    hooks = []
    selected_blocks = pick_last_two_transformer_blocks(model)
    print("=== SELECTED LAST TWO TRANSFORMER BLOCKS ===")
    if not selected_blocks:
        print("No modules matching '*.layers.N' found; falling back to hidden_states only.")
    for name, module in selected_blocks:
        print(name)

        def _mk_hook(module_name: str):
            def _hook(_m, _inputs, out):
                t = out[0] if isinstance(out, tuple) else out
                if torch.is_tensor(t):
                    hook_cache[module_name] = t.detach()

            return _hook

        hooks.append(module.register_forward_hook(_mk_hook(name)))

    graphs = [dataset[i][0] for i in idxs[: args.batch_size]]
    batch = adapter_fn(graphs, device, model)

    fwd_inputs = dict(batch.model_inputs)
    fwd_inputs["output_hidden_states"] = True
    fwd_inputs["return_dict"] = True

    with torch.no_grad():
        outputs = model(**fwd_inputs)

    for h in hooks:
        h.remove()

    if not hasattr(outputs, "hidden_states") or outputs.hidden_states is None:
        raise RuntimeError("Model output has no hidden_states. Cannot inspect last two layers.")

    hs = outputs.hidden_states
    if len(hs) < 3:
        raise RuntimeError(f"Expected embedding + >=2 layers; got hidden_states len={len(hs)}")

    print("=== LAST 2 HIDDEN STATES ===")
    layer_m2 = hs[-2]
    layer_m1 = hs[-1]
    print_stats("hidden_states[-2]", layer_m2)
    print_stats("hidden_states[-1]", layer_m1)

    print("=== HOOK CAPTURED BLOCK OUTPUTS ===")
    for k, v in hook_cache.items():
        print_stats(f"hook:{k}", v)

    token_to_node = batch.token_to_node
    if token_to_node is not None:
        node_mask = token_to_node >= 0
        node_acts = layer_m1[node_mask]
        print("=== NODE-LEVEL ACTIVATION DEFINITION ===")
        print("Node-level activation := hidden_states[-1] at positions where token_to_node >= 0")
        print_stats("node_acts", node_acts)
    else:
        node_acts = None
        print("token_to_node is None; cannot produce node-level activations.")

    print("edge_acts: not explicitly exposed by standard GraphGPT forward outputs (skipped).")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "inputs": {k: v.detach().cpu() for k, v in batch.model_inputs.items()},
            "token_to_node": None if token_to_node is None else token_to_node.detach().cpu(),
            "hidden_states_minus_2": layer_m2.detach().cpu(),
            "hidden_states_minus_1": layer_m1.detach().cpu(),
            "node_acts": None if node_acts is None else node_acts.detach().cpu(),
            "hook_outputs": {k: v.detach().cpu() for k, v in hook_cache.items()},
            "meta": {
                "subset": args.subset,
                "batch_size": args.batch_size,
                "adapter": args.adapter or "builtin_node_token_adapter",
                "node_activation_definition": "hidden_states[-1] filtered by token_to_node >= 0",
                "edge_acts": "not available from default forward outputs",
            },
        },
        out_path,
    )

    with open(out_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "output": str(out_path),
                "selected_blocks": [name for name, _ in selected_blocks],
                "adapter": args.adapter or "builtin_node_token_adapter",
            },
            f,
            indent=2,
        )

    print(f"Saved smoke activations to: {out_path}")


if __name__ == "__main__":
    main()
