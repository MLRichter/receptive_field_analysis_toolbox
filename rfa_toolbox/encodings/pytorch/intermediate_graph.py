from typing import Dict, List, Optional, Tuple

import torch
from attr import attrib, attrs
from graphviz import Digraph as GraphVizDigraph

from rfa_toolbox.encodings.pytorch.domain import LayerInfoHandler, NodeSubstitutor
from rfa_toolbox.encodings.pytorch.layer_handlers import (
    AnyAdaptivePool,
    AnyConv,
    AnyHandler,
    AnyPool,
    LinearHandler,
)
from rfa_toolbox.encodings.pytorch.substitutors import (
    input_substitutor,
    numeric_substitutor,
    output_substitutor,
)
from rfa_toolbox.graphs import EnrichedNetworkNode, LayerDefinition


def standard_resolving_strategy():
    return [AnyConv(), AnyPool(), AnyAdaptivePool(), LinearHandler(), AnyHandler()]


def standard_substitutions_strategy():
    return [input_substitutor(), output_substitutor(), numeric_substitutor()]


@attrs(auto_attribs=True, slots=True)
class Digraph:
    ref_mod: torch.nn.Module
    format: str
    graph_attr: Dict[str, str]
    inner_graph: GraphVizDigraph = attrib(init=False)
    edge_collection: List[Tuple[str, str]] = attrib(factory=list)
    raw_nodes: Dict[str, Tuple[str, str]] = attrib(factory=dict)
    layer_definitions: Dict[str, LayerDefinition] = attrib(factory=dict)
    layer_info_handlers: List[LayerInfoHandler] = attrib(
        factory=standard_resolving_strategy
    )
    layer_substitutors: List[NodeSubstitutor] = attrib(
        factory=standard_substitutions_strategy
    )

    def __attrs_post_init__(self):
        self.inner_graph = GraphVizDigraph(
            format=self.format, graph_attr=self.graph_attr
        )

    def _find_predecessors(self, name: str) -> List[str]:
        return [e[0] for e in self.edge_collection if e[1] == name]

    def _get_layer_definition(self, label: str) -> LayerDefinition:
        resolvable = self._get_resolvable(label)
        name = self._get_name(label)
        for handler in self.layer_info_handlers:
            if handler.can_handle(label):
                return handler(
                    model=self.ref_mod, resolvable_string=resolvable, name=name
                )
        raise ValueError(f"Did not find a way to handle the following layer: {name}")

    def attr(self, label: str) -> None:
        self.inner_graph.attr(label=label)

    def edge(self, node_id1: str, node_id2: str) -> None:
        self.edge_collection.append((node_id1, node_id2))
        self.inner_graph.edge(node_id1, node_id2)

    def node(
        self,
        name: str,
        label: Optional[str] = None,
        shape: str = "box",
        style: Optional[str] = None,
    ) -> None:
        # print(name, label)
        self.inner_graph.node(name, label=label, shape=shape, style=style)
        label = name if label is None else label
        layer_definition = self._get_layer_definition(label)
        self.layer_definitions[name] = layer_definition

    def subgraph(self, name: str) -> GraphVizDigraph:
        return self.inner_graph.subgraph(name=name)

    def _is_resolvable(
        self, predecessors: List[str], resolved_nodes: Dict[str, EnrichedNetworkNode]
    ) -> bool:
        if not predecessors:
            return True
        else:
            return all([pred in resolved_nodes for pred in predecessors])

    def _find_resolvable_node(
        self,
        node_to_pred_map: Dict[str, List[str]],
        resolved_nodes: Dict[str, EnrichedNetworkNode],
    ) -> Optional[str]:
        for name, preds in node_to_pred_map.items():
            if name not in resolved_nodes and self._is_resolvable(
                preds, resolved_nodes
            ):
                return name
        return None

    def _substitute(self, node: EnrichedNetworkNode):
        all_Layers = node.all_layers[:]
        for substitutor in self.layer_substitutors:
            for nd in all_Layers:
                if substitutor.can_handle(nd.layer_info.name):
                    substitutor(nd)
                    continue
        return

    def to_graph(self) -> EnrichedNetworkNode:
        node_to_pred_map: Dict[str, List[str]] = {}
        for name in self.layer_definitions.keys():
            preds = self._find_predecessors(name)
            node_to_pred_map[name] = preds
        resolved_nodes: Dict[str, EnrichedNetworkNode] = {}
        resolved_node = None
        while len(resolved_nodes) != len(node_to_pred_map):
            resolvable_node_name = self._find_resolvable_node(
                node_to_pred_map, resolved_nodes
            )
            if resolvable_node_name is None:
                break
            resolved_node = self.create_enriched_node(
                resolved_nodes,
                node_to_pred_map[resolvable_node_name],
                self.layer_definitions[resolvable_node_name],
                resolvable_node_name,
            )
            resolved_nodes[resolvable_node_name] = resolved_node
        self._substitute(resolved_node)
        return resolved_node

    def _get_resolvable(self, name: str) -> str:
        return name.split(" ")[0]

    def _get_name(self, label: str) -> str:
        if "(" in label:
            return label.split("(")[1].replace(")", "")
        else:
            return label

    def create_enriched_node(
        self,
        resolved_nodes: Dict[str, EnrichedNetworkNode],
        preds: List[str],
        layer_def: LayerDefinition,
        name: str,
    ) -> EnrichedNetworkNode:
        pred_nodes: List[EnrichedNetworkNode] = [resolved_nodes[p] for p in preds]
        node = EnrichedNetworkNode(
            name=name,
            layer_info=layer_def,
            predecessors=pred_nodes,
        )
        return node
