from typing import Any, Callable, Dict, List, Optional, Union

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
    receptive_field_max: int
    receptive_field_min: int
    receptive_field_sizes: List[int]
    predecessor_list: List["EnrichedNetworkNode"] = attrib(factory=list)
    succecessor_list: List["EnrichedNetworkNode"] = attrib(init=False, factory=list)

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
        return direct_successors + [
            succ._apply_function_to_all_successors(func)
            for succ in self.succecessor_list
        ]

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

    def to_dict(self) -> Dict[str, Union[int, str]]:
        return {
            "id": id(self),
            "name": self.name,
            "layer_type": self.layer_type.to_dict(),
            "receptive_field_max": self.receptive_field_max,
            "receptive_field_min": self.receptive_field_min,
            "receptive_field_sizes": self.receptive_field_sizes,
            "predecessor_list": [id(pred) for pred in self.predecessor_list],
        }


@attrs(auto_attribs=True, slots=True)
class ModelGraph:
    name: str
    output_node: Node
    _node_list: List[EnrichedNetworkNode] = attrib(init=False)

    @staticmethod
    def obtain_all_nodes_from_root(output_node: Node) -> List[Node]:
        node_list = [output_node]
        for node in output_node.predecessor_list:
            node_list.extend(ModelGraph.obtain_all_nodes_from_root(node))
        return list(set(node_list))

    @staticmethod
    def obtain_paths(start: Node, end: Node) -> List[List[Node]]:
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
    def compute_receptive_field_for_node_sequence(sequence: List[Node]) -> int:
        ...

    @staticmethod
    def enrich_node_sequence(
        sequence: Node, start_receptive_field: int = 0
    ) -> EnrichedNetworkNode:
        ...

    def __attrs_post_init__(self):
        nodes = ModelGraph.obtain_all_nodes_from_root(input_node=self.input_node)
        self.node_list = [self.input_node] + nodes

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

    def to_dict(self) -> Dict[str, Union[int, str]]:
        return {
            "name": self.name,
            "input_node": id(self.input_node),
            "node_mapping": {id(node): node.to_dict() for node in self.node_list},
        }
