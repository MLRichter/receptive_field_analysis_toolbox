try:
    from typing import Any, Dict, List, Optional, Protocol, Union
except ImportError:
    from typing import Dict, List, Optional, Union

    from typing_extensions import Protocol


class Layer(Protocol):
    """This Protocol describes a simple information container for nerwork layers.

    Args:
        name:           name of the layer
        kernel_size:    the kernel size of convolution operation.
                        Non convolutional layers are treated as having
                        an infinite kernel size.
        stride_size:    The stride size of the convolution operation.
                        Fully connected layers are treated as having
                        a stride-size of 1.
        filters:        The number of filters in the layer,
                        only set if convolutional
        units:          The units of a fully connected layers, only
                        set if kernel size is infinite.
    """

    name: str
    kernel_size: Optional[int]
    stride_size: int
    filters: Optional[int]
    units: Optional[int]

    @staticmethod
    def from_dict(**config) -> "Layer":
        """Create a layer object from the dictionary.

        Args:
            **config:   keyword arguments for the constructor.

        Returns:
            A layer instance.
        """
        ...

    def to_dict(self) -> Dict[str, Union[int, str]]:
        """Create a json-serializable dictionary from which
        the object can be reconstructed.

        Returns:
            A dictionary from which the layer can be reconstructed.
        """
        ...


class Node(Protocol):
    """This instances is the core component the a graph is
    constructed from.

    Args:
        name:               the name of the node
        layer_type:         the layer information container
        predecessor_list:   the list of predecessor nodes
    """

    name: str
    layer_type: Layer
    predecessor_list: List["Node"]

    def is_in(self, container: Union[List["Node"], Dict["Node", Any]]) -> bool:
        """Checks if this particular node is in a container based on the object-id.

        Args:
            container:  the container to search through.
                        The search is non-recursive and thus
                        does not look into nested containers.

        Returns:
            True this object is in the container, else False.

        """
        ...
