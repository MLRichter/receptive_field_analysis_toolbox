from typing import Protocol, Optional, List, Dict, Union


class Layer(Protocol):
    name: str
    kernel_size: Optional[int]
    stride_size: int

    @staticmethod
    def from_dict(**config) -> "Layer":
        ...

    def to_dict(self) -> Dict[str, Union[int, str]]:
        ...


class Node(Protocol):
    name: str
    layer_type: Layer
    predecessor_list: List["Node"]
    successors_list: List["Node"]

    def predecessors(self) -> Dict[str, Layer]:
        ...

    def successors(self) -> Dict[str, Layer]:
        ...

    @staticmethod
    def from_dict(**config) -> "Node":
        ...

    def to_dict(self) -> Dict[str, Union[int, str]]:
        ...


class Graph:
    name: str
    input_node: Node
    output_node: Node
    node_list: List[Node]

    @staticmethod
    def from_dict(**config) -> "Graph":
        ...

    def to_dict(self) -> Dict[str, Union[int, str]]:
        ...
