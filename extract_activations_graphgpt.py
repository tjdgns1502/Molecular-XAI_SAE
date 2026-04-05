#!/usr/bin/env python3
"""Step-2 GraphGPT activation extraction on PCQM4Mv2.

Uses the same validated tensor definition as Step-1:
- primary tensor: hidden_states[-1]
- node-level rows: positions where token_to_node >= 0
- edge-level activations: omitted unless adapter/model surfaces explicit mapping
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import torch


class AdapterBatch:
    def __init__(self, model_inputs: Dict[str, torch.Tensor], token_to_node: Optional[torch.Tensor]):
        self.model_inputs = model_inputs
        self.token_to_node = token_to_node


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


def import_graphgpt_modules(graphgpt_repo: Optional[str]):
    if graphgpt_repo:
        sys.path.insert(0, graphgpt_repo)
    cfg_mod = importlib.import_module("src.models.graphgpt.configuration_graphgpt")
    model_mod = importlib.import_module("src.models.graphgpt.modeling_finetune")
    cfg_cls = getattr(cfg_mod, "GraphGPTConfig")
    for name in ("GraphGPTTaskModel", "GraphGPTDoubleHeadsModel"):
        if hasattr(model_mod, name):
            return cfg_cls, getattr(model_mod, name)
    raise RuntimeError("Could not find GraphGPT model class in modeling_finetune.py")


def load_model(graphgpt_repo: Optional[str], checkpoint: Optional[str], device: torch.device):
    cfg_cls, model_cls = import_graphgpt_modules(graphgpt_repo)
    if checkpoint:
        cfg = cfg_cls.from_pretrained(checkpoint)
        model = model_cls.from_pretrained(checkpoint, config=cfg)
    else:
        cfg = cfg_cls()
        model = model_cls(cfg)
    return model.to(device).eval()


def builtin_node_token_adapter(graphs: Sequence[Dict[str, Any]], device: torch.device, model: Any) -> AdapterBatch:
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


def load_adapter(adapter_path: Optional[str], fn_name: str) -> AdapterFn:
    if adapter_path is None:
        return builtin_node_token_adapter

    p = Path(adapter_path)
    if not p.exists():
        raise FileNotFoundError(f"Adapter file not found: {p}")
    spec = importlib.util.spec_from_file_location("graphgpt_user_adapter", str(p))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load adapter module from {p}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    fn = getattr(mod, fn_name, None)
    if fn is None:
        raise RuntimeError(f"Adapter function '{fn_name}' missing in {p}")
    return fn


def print_stats(name: str, t: torch.Tensor) -> None:
    ft = t.float().reshape(-1)
    sample = ft[: min(8, ft.numel())].cpu().tolist()
    print(
        f"{name}: shape={tuple(t.shape)} dtype={t.dtype} "
        f"mean={ft.mean().item():.6f} std={ft.std().item():.6f} sample={sample}"
    )


def save_chunk(out_dir: Path, chunk_idx: int, node_chunks: List[torch.Tensor], metadata: List[Dict[str, Any]]) -> None:
    node_acts = torch.cat(node_chunks, dim=0) if node_chunks else torch.empty((0,))
    chunk_file = out_dir / f"chunk_{chunk_idx:05d}.pt"
    meta_file = out_dir / f"chunk_{chunk_idx:05d}.json"

    torch.save(
        {
            "node_acts": node_acts,
            "edge_acts": None,
            "format_version": 1,
            "activation_source": "hidden_states[-1] filtered by token_to_node>=0",
        },
        chunk_file,
    )
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"saved chunk tensor: {chunk_file}")
    print(f"saved chunk metadata: {meta_file}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graphgpt-repo", type=str, default=None)
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--dataset-root", type=str, default="./data")
    ap.add_argument("--subset", type=int, default=10000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--chunk-size", type=int, default=200000, help="target node rows per chunk")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adapter", type=str, default=None)
    ap.add_argument("--adapter-fn", type=str, default="build_batch")
    ap.add_argument("--out-dir", type=str, default="artifacts/graphgpt_acts")
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_pcqm4mv2_dataset(args.dataset_root)
    all_idxs = list(range(len(dataset)))
    random.Random(args.seed).shuffle(all_idxs)
    idxs = all_idxs[: min(args.subset, len(all_idxs))]

    model = load_model(args.graphgpt_repo, args.checkpoint, device)
    adapter_fn = load_adapter(args.adapter, args.adapter_fn)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    chunk_idx = 0
    node_chunks: List[torch.Tensor] = []
    chunk_meta: List[Dict[str, Any]] = []
    rows_in_chunk = 0

    for start in range(0, len(idxs), args.batch_size):
        batch_idxs = idxs[start : start + args.batch_size]
        graphs = [dataset[i][0] for i in batch_idxs]
        adapter_batch = adapter_fn(graphs, device, model)

        forward_inputs = dict(adapter_batch.model_inputs)
        forward_inputs["output_hidden_states"] = True
        forward_inputs["return_dict"] = True

        with torch.no_grad():
            out = model(**forward_inputs)

        last = out.hidden_states[-1]  # [B, S, D]
        if adapter_batch.token_to_node is None:
            raise RuntimeError("Adapter returned token_to_node=None; cannot derive node-level activations")

        token_to_node = adapter_batch.token_to_node
        node_mask = token_to_node >= 0
        node_acts = last[node_mask].detach().cpu()
        node_chunks.append(node_acts)
        rows_in_chunk += int(node_acts.shape[0])

        print_stats("batch.hidden_states[-1]", last.detach().cpu())
        print_stats("batch.node_acts", node_acts)
        print("edge_acts: skipped (not explicitly exposed in standard GraphGPT forward outputs).")

        for row_i, mol_idx in enumerate(batch_idxs):
            num_atoms = int((token_to_node[row_i] >= 0).sum().item())
            edge_index = graphs[row_i].get("edge_index", None)
            num_bonds = int(edge_index.shape[1] // 2) if edge_index is not None else -1
            chunk_meta.append(
                {
                    "molecule_idx": int(mol_idx),
                    "num_atoms": num_atoms,
                    "num_bonds": num_bonds,
                    "chunk_file": f"chunk_{chunk_idx:05d}.pt",
                }
            )

        if rows_in_chunk >= args.chunk_size:
            save_chunk(out_dir, chunk_idx, node_chunks, chunk_meta)
            chunk_idx += 1
            node_chunks = []
            chunk_meta = []
            rows_in_chunk = 0

    if node_chunks:
        save_chunk(out_dir, chunk_idx, node_chunks, chunk_meta)


if __name__ == "__main__":
    main()
