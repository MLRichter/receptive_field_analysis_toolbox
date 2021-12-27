try:
    from typing import Any, Dict, List, Optional, Protocol, Union
except ImportError:
    from typing import Dict, List, Optional, Union

    from typing_extensions import Protocol


class Layer(Protocol):
    name: str
    kernel_size: Optional[int]
    stride_size: int
    filters: Optional[int]

    @staticmethod
    def from_dict(**config) -> "Layer":
        ...

    def to_dict(self) -> Dict[str, Union[int, str]]:
        ...


class Node(Protocol):
    name: str
    layer_type: Layer
    predecessor_list: List["Node"]

    def successors(self) -> Dict[str, Layer]:
        ...

    @staticmethod
    def from_dict(**config) -> "Node":
        ...

    def to_dict(self) -> Dict[str, Union[int, str]]:
        ...

    def is_in(self, container: Union[List["Node"], Dict["Node", Any]]) -> bool:
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
