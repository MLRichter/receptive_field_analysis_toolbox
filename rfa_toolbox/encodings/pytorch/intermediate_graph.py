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
    FunctionalKernelHandler,
    LinearHandler,
)
from rfa_toolbox.encodings.pytorch.substitutors import (
    input_substitutor,
    numeric_substitutor,
    output_substitutor,
)
from rfa_toolbox.graphs import EnrichedNetworkNode, LayerDefinition

RESOLVING_STRATEGY = [
    AnyConv(),
    AnyPool(),
    AnyAdaptivePool(),
    LinearHandler(),
    FunctionalKernelHandler(),
    AnyHandler(),
]
SUBSTITUTION_STRATEGY = [
    numeric_substitutor(),
    input_substitutor(),
    output_substitutor(),
]


@attrs(auto_attribs=True, slots=True)
class Digraph:
    """This digraph object is used to transform the j
    it-compiled digraph into the graph-representation
    of this library.

    Args:
        ref_mod:    the neural network model in a non-jit-compiled
                    variant
    """

    ref_mod: torch.nn.Module
    format: str = ""
    graph_attr: Dict[str, str] = attrib(factory=dict)
    edge_collection: List[Tuple[str, str]] = attrib(factory=list)
    raw_nodes: Dict[str, Tuple[str, str]] = attrib(factory=dict)
    layer_definitions: Dict[str, LayerDefinition] = attrib(factory=dict)
    layer_info_handlers: List[LayerInfoHandler] = attrib(
        factory=lambda: RESOLVING_STRATEGY
    )
    layer_substitutors: List[NodeSubstitutor] = attrib(
        factory=lambda: SUBSTITUTION_STRATEGY
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
        """This is a dummy function to mimic the behavior
        of a digraph-object from Graphviz with no functionality."""
        pass

    def edge(self, node_id1: str, node_id2: str) -> None:
        """Creates an directed edge in the compute graph
        from one node to the other in the current Digraph-Instance

        Args:
            node_id1:   the id of the start node
            node_id2:   the id of the target node

        Returns:
            Nothing.

        """
        self.edge_collection.append((node_id1, node_id2))

    def node(
        self,
        name: str,
        label: Optional[str] = None,
        shape: str = "box",
        style: Optional[str] = None,
    ) -> None:
        """Creates a node in the digraph-instance.

        Args:
            name:   the name of the node, the name must be unique
                    to properly identify the node.
            label:  the label is descriptive for the functionality
                    of the node
            shape:  unused variable for compatibility with GraphViz
            style:  unused variable for compatibility with GraphViz

        Returns:
            Nothing.
        """
        label = name if label is None else label
        layer_definition = self._get_layer_definition(label)
        self.layer_definitions[name] = layer_definition

    def subgraph(self, name: str) -> GraphVizDigraph:
        """This is a dummy function to mimic the behavior
        of a digraph-object from Graphviz with no functionality."""
        pass

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
        """Transforms the graph stored in the Digraph-Instance into
        a graph consisting of EnrichedNetworkNode-objects.
        Allowing the computation of border layers and the visualization of the
        graph using the visualize-Module.

        Returns:
            The output-node of the EnrichedNetworkNode-based graph

        """
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
        """Creates an enriched node from the current graph node.

        Args:
            resolved_nodes: a dicationary, mapping node-ids to the nodes
                            to their corresponding EnrichedNetworkNode instances
            preds:          a list the direct predecessor (ids)
            layer_def:      the layer definition instance for this node.
            name:           thr name of the node, used as id

        Returns:
            The EnrichedNetworkNode instance of the same node

        """
        pred_nodes: List[EnrichedNetworkNode] = [resolved_nodes[p] for p in preds]
        node = EnrichedNetworkNode(
            name=name,
            layer_info=layer_def,
            predecessors=pred_nodes,
        )
        return node
