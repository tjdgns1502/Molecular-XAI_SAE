"""Template adapter for GraphGPT PCQM4Mv2 integration.

Implement `build_batch(graphs, device, model)` with GraphGPT's official Eulerian
preprocessing/tokenization from your checked-out GraphGPT repository.

Contract:
- Input `graphs`: list of OGB PCQM4Mv2 graph dicts.
- Return object with:
    model_inputs: dict ready for model(**model_inputs)
    token_to_node: LongTensor[B, S], -1 for non-node tokens
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch


@dataclass
class AdapterBatch:
    model_inputs: Dict[str, torch.Tensor]
    token_to_node: torch.Tensor


def build_batch(graphs: Sequence[Dict[str, Any]], device: torch.device, model: Any) -> AdapterBatch:
    raise NotImplementedError(
        "Implement GraphGPT Eulerian tokenizer mapping here. "
        "token_to_node must map sequence positions back to node indices (or -1)."
    )
