from rfa_toolbox.graphs import EnrichedNetworkNode, LayerDefinition

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

import torch


class LayerInfoHandler(Protocol):
    def can_handle(self, name: str) -> bool:
        ...

    def __call__(
        self, model: torch.nn.Module, resolvable_string: str, name: str
    ) -> LayerDefinition:
        ...


class NodeSubstitutor(Protocol):
    def can_handle(self, name: str) -> bool:
        ...

    def __call__(self, node: EnrichedNetworkNode) -> EnrichedNetworkNode:
        ...
