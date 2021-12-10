from typing import Optional, List, Dict, Callable, Union
from attr import attrs, attrib
import numpy as np

from rfa_toolbox.domain import Layer, Node


@attrs(auto_attribs=True, frozen=True, slots=True)
class LayerDefinition(Layer):

    name: str
    kernel_size: Optional[int] = attrib(factory=lambda x: np.inf if x is None else x)
    stride_size: int = 1

    @staticmethod
    def from_dict(**config):
        return LayerDefinition(**config)


@attrs(auto_attribs=True, frozen=True, slots=True)
class InputLayer(Layer):
    name: str = "input"
    kernel_size = 1
    stride_size = 1


@attrs(auto_attribs=True, frozen=True, slots=True)
class OutputLayer(Layer):
    name: str = "output"
    kernel_size = 1
    stride_size = 1


@attrs(auto_attribs=True, frozen=True, slots=True)
class NetworkNode(Node):

    name: str
    layer_type: Layer
    predecessor_list: List["NetworkNode"] = attrib(factory=list)
    successors_list: List["NetworkNode"] = attrib(factory=list)

    @property
    def predecessors(self) -> Dict[str, Layer]:
        return {predec.name: predec for predec in self.predecessor_list}

    @property
    def successors(self) -> Dict[str, Layer]:
        return {succ.name: succ for succ in self.successors_list}


@attrs(auto_attribs=True, frozen=True, slots=True)
class EnrichedNetworkNode(Node):

    name: str
    layer_type: LayerDefinition
    receptive_field_max: int
    receptive_field_min: int
    receptive_field_sizes: List[int]
    predecessor_list: List["EnrichedNetworkNode"] = attrib(factory=list)
    successors_list: List["EnrichedNetworkNode"] = attrib(factory=list)

    def is_border(self, input_resolution: int, receptive_field_provider: Callable[["EnrichedNetworkNode"], int]= lambda x: x.receptive_field_min) -> bool:
        ...


@attrs(auto_attribs=True, slots=True)
class ModelGraph:
    name: str
    input_node: Node
    output_node: Node
    _node_list: List[EnrichedNetworkNode] = attrib(init=False)

    @staticmethod
    def obtain_all_nodes_from_root(input_node: Node) -> List[Node]:
        node_list = [input_node]
        for node in input_node.successors_list:
            node_list.extend(ModelGraph.obtain_all_nodes_from_root(node))
        return list(set(node_list))

    @staticmethod
    def obtain_paths(start: Node, end: Node) -> List[List[Node]]:
        ...

    @staticmethod
    def compute_receptive_field_for_node_sequence(sequence: List[Node]) -> int:
        ...

    @staticmethod
    def enrich_node_sequence(sequence: Node) -> EnrichedNetworkNode:
        ...

    def __attrs_post_init__(self):
        nodes = ModelGraph.obtain_all_nodes_from_root(input_node=self.input_node)
        self.node_list = [self.input_node] + nodes + [self.output_node]


