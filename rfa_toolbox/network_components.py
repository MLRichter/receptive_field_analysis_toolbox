from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from attr import attrib, attrs

from rfa_toolbox.domain import Layer, Node


@attrs(auto_attribs=True, frozen=True, slots=True)
class LayerDefinition(Layer):
    name: str
    kernel_size: Optional[int] = attrib(factory=lambda x: np.inf if x is None else x)
    stride_size: int = 1

    @staticmethod
    def from_dict(**config) -> "LayerDefinition":
        return LayerDefinition(**config)

    def to_dict(self) -> Dict[str, Union[int, str]]:
        return {
            "name": self.name,
            "kernel_size": self.kernel_size,
            "stride_size": self.stride_size,
        }


@attrs(auto_attribs=True, frozen=True, slots=True)
class InputLayer(Layer):
    name: str = "input"
    kernel_size = 1
    stride_size = 1

    @staticmethod
    def from_dict(**config) -> "InputLayer":
        return InputLayer(**config)

    def to_dict(self) -> Dict[str, Union[int, str]]:
        return {
            "name": self.name,
            "kernel_size": self.kernel_size,
            "stride_size": self.stride_size,
        }


@attrs(auto_attribs=True, frozen=True, slots=True)
class OutputLayer(Layer):
    name: str = "output"
    kernel_size = 1
    stride_size = 1

    @staticmethod
    def from_dict(**config) -> "OutputLayer":
        return OutputLayer(**config)

    def to_dict(self) -> Dict[str, Union[int, str]]:
        return {
            "name": self.name,
            "kernel_size": self.kernel_size,
            "stride_size": self.stride_size,
        }


@attrs(auto_attribs=True, frozen=True, slots=True)
class NetworkNode(Node):
    name: str
    layer_type: Layer
    predecessor_list: List["NetworkNode"] = attrib(factory=list)

    @property
    def predecessors(self) -> Dict[str, Layer]:
        return {pred.name: pred for pred in self.predecessor_list}

    @staticmethod
    def from_dict(**config) -> "NetworkNode":
        if "id" in config:
            config.pop("id")
        config["layer_type"] = LayerDefinition.from_dict(**config["layer_type"])
        return NetworkNode(**config)

    def to_dict(self) -> Dict[str, Union[int, str]]:
        return {
            "id": id(self),
            "name": self.name,
            "layer_type": self.layer_type.to_dict(),
            "predecessor_list": [id(pred) for pred in self.predecessor_list],
        }


@attrs(auto_attribs=True, frozen=True, slots=True)
class EnrichedNetworkNode(Node):
    name: str
    layer_type: LayerDefinition
    receptive_field_sizes: List[int]
    predecessor_list: List["EnrichedNetworkNode"] = attrib(factory=list)
    succecessor_list: List["EnrichedNetworkNode"] = attrib(init=False, factory=list)

    @property
    def receptive_field_min(self):
        return min(self.receptive_field_sizes)

    @property
    def receptive_field_max(self):
        return max(self.receptive_field_sizes)

    def __attrs_post_init__(self):
        for pred in self.predecessor_list:
            pred.succecessor_list.append(self)

    def add_successor(self, successor: "EnrichedNetworkNode") -> None:
        if successor not in self.succecessor_list:
            self.succecessor_list.append(successor)

    def _apply_function_to_all_successors(
        self, func: Callable[["EnrichedNetworkNode"], Any]
    ) -> List[Any]:
        direct_successors = [func(succ) for succ in self.succecessor_list]
        indirect_successors = []

        for succ in self.succecessor_list:
            indirect_successors.extend(succ._apply_function_to_all_successors(func))

        return direct_successors + indirect_successors

    def is_border(
        self,
        input_resolution: int,
        receptive_field_provider: Callable[
            ["EnrichedNetworkNode"], int
        ] = lambda x: x.receptive_field_min,
    ) -> bool:
        # the border layer is defined as the layer that receives
        # all inputs with a receptive field size
        # SMALLER than the input resolution
        direct_predecessors = [
            input_resolution <= receptive_field_provider(pred)
            for pred in self.predecessor_list
        ]
        # of course, this means that this layer also needs to fullfill this property
        own = input_resolution <= receptive_field_provider(self)
        # additionally (only relevant for multipath architectures)
        # all following layer are border layers as well
        successors = [
            input_resolution <= result
            for result in self._apply_function_to_all_successors(
                receptive_field_provider
            )
        ]
        # in short all direct predecessors,
        # the layer itself and all following layers have a receptive field size
        # GREATER than the input resolution
        return all(direct_predecessors) and own and all(successors)

    @staticmethod
    def from_dict(**config) -> "EnrichedNetworkNode":
        if "id" in config:
            config.pop("id")
        config["layer_type"] = LayerDefinition.from_dict(**config["layer_type"])
        return EnrichedNetworkNode(**config)

    @staticmethod
    def from_node(
        node: Node, predecessor_mapping: Dict[Node, "EnrichedNetworkNode"]
    ) -> "EnrichedNetworkNode":
        node_dict: Dict[
            str, Union[str, int, List["EnrichedNetworkNode"]]
        ] = node.to_dict()
        node_dict["predecessor_list"] = [
            predecessor_mapping[pred] for pred in node.predecessor_list
        ]
        return EnrichedNetworkNode.from_dict(**node_dict)

    def to_dict(self) -> Dict[str, Union[int, str]]:
        return {
            "id": id(self),
            "name": self.name,
            "layer_type": self.layer_type.to_dict(),
            "receptive_field_sizes": self.receptive_field_sizes,
            "predecessor_list": [id(pred) for pred in self.predecessor_list],
        }


@attrs(auto_attribs=True, slots=True)
class ModelGraph:
    name: str
    output_node: Node
    _node_list: List[EnrichedNetworkNode] = attrib(init=False)

    @staticmethod
    def obtain_all_nodes_from_root(
        output_node: EnrichedNetworkNode,
    ) -> List[EnrichedNetworkNode]:
        node_list = [output_node]
        for node in output_node.predecessor_list:
            node_list.extend(ModelGraph.obtain_all_nodes_from_root(node))
        return list(set(node_list))

    @staticmethod
    def obtain_paths(
        start: EnrichedNetworkNode, end: EnrichedNetworkNode
    ) -> List[List[EnrichedNetworkNode]]:
        paths: List[List[Node]] = []
        for node in start.predecessor_list:
            if node == start:
                # found the shortest possible path
                paths.append([start, end])
            elif isinstance(node.layer_type, InputLayer):
                # no need to go further
                continue
            else:
                # recursivly search for paths
                paths = ModelGraph.obtain_paths(start=start, end=node)
                for path in paths:
                    path.append(end)
        return paths

    @staticmethod
    def _find_input_node(
        all_node_list: List[EnrichedNetworkNode],
    ) -> EnrichedNetworkNode:
        for node in all_node_list:
            if isinstance(node.layer_type, InputLayer):
                return node

    @staticmethod
    def _compute_receptive_field_for_node(
        node: Node,
        prev_receptive_field_size: int,
        prev_kernel_size: int,
        growth_rate: int,
    ) -> Tuple[int, int]:
        receptive_field_size = prev_receptive_field_size + (
            (prev_kernel_size - 1) * growth_rate
        )
        current_growth_rate = growth_rate * node.layer_type.stride_size
        return receptive_field_size, current_growth_rate

    @staticmethod
    def compute_receptive_field_for_node_sequence(
        sequence: List[EnrichedNetworkNode],
    ) -> List[EnrichedNetworkNode]:
        multiplicator = sequence[0].layer_type.stride_size
        receptive_field_size = sequence[0].layer_type.kernel_size
        for i, node in enumerate(sequence):
            if i != 0:
                # update receptive field size and growth multiplicator
                (
                    receptive_field_size,
                    multiplicator,
                ) = ModelGraph._compute_receptive_field_for_node(
                    node,
                    receptive_field_size,
                    node.layer_type.kernel_size,
                    multiplicator,
                )[
                    0
                ]
            if receptive_field_size not in node.receptive_field_sizes:
                node.receptive_field_sizes.append(receptive_field_size)
        return sequence

    @staticmethod
    def enrich_graph(
        current_node: Node, enriched_nodes: Dict[Node, EnrichedNetworkNode]
    ) -> EnrichedNetworkNode:
        if not current_node.predecessor_list:
            node = EnrichedNetworkNode.from_node(current_node)
            enriched_nodes[current_node] = node
            return node
        else:
            nodes_known = [
                node in enriched_nodes for node in current_node.predecessor_list
            ]
            if not all(nodes_known):
                for predecessor in current_node.predecessor_list:
                    if predecessor in enriched_nodes:
                        continue
                    else:
                        # enrich the predecessor
                        ModelGraph.enrich_graph(predecessor, enriched_nodes)
            node = EnrichedNetworkNode.from_node(current_node, enriched_nodes)
            enriched_nodes[current_node] = node
            return node

    @staticmethod
    def obtain_all_paths(
        output_node: EnrichedNetworkNode, all_nodes: List[EnrichedNetworkNode]
    ) -> List[List[EnrichedNetworkNode]]:
        input_node = ModelGraph._find_input_node(all_nodes)
        paths = ModelGraph.obtain_paths(input_node, output_node)
        return paths

    def __attrs_post_init__(self):
        node = ModelGraph.enrich_graph(self.output_node, {})
        self.output_node = node
        self.node_list = self.obtain_all_nodes_from_root(node)
        # obtain all paths from input to output
        paths = ModelGraph.obtain_all_paths(node, self.node_list)
        # enrich all nodes along the way with receptive field size information
        for path in paths:
            ModelGraph.compute_receptive_field_for_node_sequence(path)

    @staticmethod
    def from_dict(**config) -> "ModelGraph":
        node_mapping = {
            identity: EnrichedNetworkNode.from_dict(**node)
            if "receptive_field_sizes" in node
            else NetworkNode.from_dict(**node)
            for identity, node in config["node_mapping"].items()
        }
        graph = ModelGraph(config["name"], node_mapping[config["input_node"]])
        return graph

    def unproductive_layers(self, input_resolution: int) -> List[EnrichedNetworkNode]:
        return [node for node in self.node_list if node.is_border(input_resolution)]

    def earliest_border_layers(
        self, input_resolution: int
    ) -> List[EnrichedNetworkNode]:
        candidates = self.unproductive_layers(input_resolution=input_resolution)
        candidates_with_no_predecessor_border_layer = []
        for candidate in candidates:
            predecessors_are_border_layers = [
                pred.is_border(input_resolution) for pred in candidate.predecessor_list
            ]
            if not any(predecessors_are_border_layers):
                candidates_with_no_predecessor_border_layer.append(candidate)
        return candidates_with_no_predecessor_border_layer

    def to_dict(self) -> Dict[str, Union[int, str]]:
        return {
            "name": self.name,
            "output_node": id(self.output_node),
            "node_mapping": {id(node): node.to_dict() for node in self.node_list},
        }
