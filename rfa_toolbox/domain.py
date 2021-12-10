from typing import Protocol, Optional, List, Dict


class Layer(Protocol):
    name: str
    kernel_size: Optional[int]
    stride_size: int


class Node(Protocol):
    name: str
    layer_type: Layer
    predecessor_list: List["Node"]
    successors_list: List["Node"]

    def predecessors(self) -> Dict[str, Layer]:
        ...

    def successors(self) -> Dict[str, Layer]:
        ...


class Graph:
    name: str
    input_node: Node
    output_node: Node
    node_list: List[Node]
