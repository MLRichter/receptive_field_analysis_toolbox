from operator import attrgetter
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from attr import attrib, attrs

from rfa_toolbox.domain import Layer, Node


def receptive_field_provider_with_1x1_handling(
    node: "EnrichedNetworkNode",
) -> Optional[int]:
    """Provides the MINIMUM receptive field size of a layer
    with an exception handling for 1x1-convolutions, which are treated
    as having an infinite receptive field size. This provider
    is based on the hypothesis that 1x1 convolutions are
    always unproductive. It is worth noting that this
    hypothesis is still under investigation.

    Args:
        node:   the node to receive the receptive field size from.

    Returns:
        the receptive field size, infinite if the kernel size is equal to 1.
    """
    return node.receptive_field_min if node.layer_type.kernel_size > 1 else np.inf


def receptive_field_provider(node: "EnrichedNetworkNode") -> Optional[int]:
    """Provides the MINIMUM receptive field size from a node.
    Based on the result of https://arxiv.org/abs/2106.12307 this
    is currently the most reliable way of predicting unproductive layers.


    Args:
        node: the node to return the receptive field size.

    Returns:
        the minimum receptive field size.
    """
    return node.receptive_field_min


# FIXME: Make this function work for scenarios,
# where the infos contain tuples of receptive field sizes describing an area
def naive_minmax_filter(
    info: Tuple["ReceptiveFieldInfo"],
) -> Tuple["ReceptiveFieldInfo", "ReceptiveFieldInfo"]:
    """Filters all receptive field infos, except for the one
    with the mininum and maximum receptive field size.
    Currently only works if all receptive field sizes, kernel and
    stride sizes are scalar

    Args:
        info:   Tuple of receptive field info containers to filters

    Returns:
        A two-tuple containing the minimum and maximum receptive field size info.

    """
    maximum_receptive_field: ReceptiveFieldInfo = max(
        info, key=attrgetter("receptive_field")
    )
    minimum_receptive_field: ReceptiveFieldInfo = min(
        info, key=attrgetter("receptive_field")
    )
    return minimum_receptive_field, maximum_receptive_field


def noop_filter(
    info: Tuple["ReceptiveFieldInfo"],
) -> Tuple["ReceptiveFieldInfo"]:
    return info


@attrs(auto_attribs=True, frozen=True, slots=True)
class ReceptiveFieldInfo:
    """The container holding information for the successive receptive
    field size computation.

    Args:
        receptive_field:    the receptive field size
        multiplicator:      the current growth multiplicator,
                            increased by stride sizes > 1
    """

    receptive_field: Union[int, Sequence[int]] = attrib(
        converter=lambda x: tuple(x)
        if isinstance(x, Sequence) or isinstance(x, np.ndarray)
        else x
    )
    multiplicator: Union[int, Sequence[int]] = attrib(
        converter=lambda x: tuple(x)
        if isinstance(x, Sequence) or isinstance(x, np.ndarray)
        else x
    )


@attrs(auto_attribs=True, frozen=True, slots=True)
class LayerDefinition(Layer):
    """The standard representation of a neural network layer.
    Contains information needed for receptive field computation.

    Args:
        name:           name of the layer
        kernel_size:    size of the kernel, None if this is a dense-layer
        stride_size:    the stride size the kernel is convolved. None for dense-layers.
        filters:        number of filter produced by the convolution operation
        units:          number of units of a fully connected layer

    """

    name: str
    kernel_size: Optional[Union[int, Sequence[int]]] = attrib(
        converter=lambda x: np.inf if x is None else x, default=None
    )
    stride_size: Optional[Union[int, Sequence[int]]] = attrib(
        converter=lambda x: 1 if x is None else x, default=None
    )
    filters: Optional[int] = None
    units: Optional[int] = None

    @kernel_size.validator
    def validate_kernel_size(
        self, attribute: str, value: Union[int, Sequence[int]]
    ) -> None:
        if isinstance(value, Sequence):
            for v in value:
                self.validate_kernel_size(attribute, v)
        elif value is not None and value < 1:
            raise ValueError(
                f"{attribute} values must be greater than 0 or "
                f"infinite (which indicates a dense layer)"
            )

    @stride_size.validator
    def validate_stride_size(
        self, attribute: str, value: Union[int, Sequence[int]]
    ) -> None:
        if isinstance(value, Sequence):
            for v in value:
                self.validate_stride_size(attribute, v)
        elif value is not None and value < 1:
            raise ValueError(
                f"{attribute} values must be greater than 0 or "
                f"infinite (which indicates a dense layer)"
            )

    def _check_consistency_for_kernel_and_stride_sequences(self) -> None:
        if isinstance(self.kernel_size, Sequence) and isinstance(
            self.stride_size, Sequence
        ):
            if len(self.kernel_size) != len(self.stride_size):
                raise ValueError(
                    "kernel_size and stride_size must have the same length"
                )
            for i in range(len(self.kernel_size)):
                if len(self.kernel_size) != len(self.stride_size):
                    raise ValueError(
                        "When kernel_size and stride_size are both sequences, "
                        "they must have the same length, kernel_size: "
                        f"{self.kernel_size}, stride_size: {self.stride_size}"
                    )

    def __attrs_post_init__(self):
        self._check_consistency_for_kernel_and_stride_sequences()

    @classmethod
    def from_dict(cls, config) -> "LayerDefinition":
        """Create a LayerDefinition from the dictionary.

        Args:
            config: create layer definiton from the dictionary.

        Returns:
            A LayerDefinition instance

        """
        return LayerDefinition(**config)

    def to_dict(self) -> Dict[str, Union[int, str]]:
        """Create a json-serializable dictionary from this object instance.

        Returns:
            A diction from which this object can be reconstructed.

        """
        return {
            "name": self.name,
            "kernel_size": self.kernel_size,
            "stride_size": self.stride_size,
        }


def compute_receptive_field_sizes(
    receptive_field_info: Set[ReceptiveFieldInfo], layer_info: Layer
) -> Tuple[ReceptiveFieldInfo]:
    """Compute the receptive field sizes for a node given
    receptive field-infos from predecessor-nodes and the
    current layer information.

    Args:
        receptive_field_info:   A iterable collection of receptive field informations
                                collected from predecessor layers.
        layer_info:             The layer information container for the current layer.

    Returns:
        A tuple of ReceptiveFieldInfo-instances for this particular layer.

    """
    result: List[ReceptiveFieldInfo] = list()
    for rf_info in receptive_field_info:
        receptive_field = np.asarray(rf_info.receptive_field) + (
            (np.asarray(layer_info.kernel_size) - 1) * np.asarray(rf_info.multiplicator)
        )
        multiplicator = np.asarray(layer_info.stride_size) * np.asarray(
            rf_info.multiplicator
        )
        new_info = ReceptiveFieldInfo(
            receptive_field=receptive_field, multiplicator=multiplicator
        )
        result.append(new_info)

    return tuple(result)


@attrs(auto_attribs=True, frozen=True, slots=True, hash=False, repr=False)
class EnrichedNetworkNode(Node):
    """The EnrichedNetworkNode is the core component of a network graph in this framework.
    Any node af a network can be used as a handle for the entire graph.
    A neural network is exspected to have exactly one input and arbitrary
    many outputs. Networks with multiple inputs may cause inconsistencies.

    Args:
        name:                       the name of the current node
        layer_info:                 the layer information container
        predecessors:               A list of predecessor nodes, empty-list by default.
        receptie_field_info_filter: Function, which filters the ReceptiveFieldInfo
                                    to reduce the number of computations in networks
                                    with many pathways or skip connections.
                                    By default only the highest and lowest
                                    receptive field size container are kept.

    Params:
        receptive_field_info:   a n-tuple holding the ReceptiveFieldInfo-instances,
                                used for receptive field size computation
        receptive_field_min:    minimum receptive field size
        receptive_field_max:    maximum receptive field size
        receptive_field_sizes:  all receptive field sizes, please note
                                that a filter is applied
        all_laxers:             a list of all nodes contained in the graph
        kernel_size:            the size of the kernel, passthrough from
                                the layer_info container
        stride_size:            the stride size, passthrough from
                                the layer_info container

    """

    name: str
    layer_info: LayerDefinition
    predecessors: List["EnrichedNetworkNode"] = attrib(converter=list)
    succecessors: List["EnrichedNetworkNode"] = attrib(
        init=False, factory=list, eq=False
    )
    receptive_field_info: Tuple[ReceptiveFieldInfo] = attrib(init=False)
    receptive_field_min: int = attrib(init=False)
    receptive_field_max: int = attrib(init=False)

    receptive_field_info_filter: Callable[
        [Tuple[ReceptiveFieldInfo]], Tuple[ReceptiveFieldInfo]
    ] = noop_filter
    all_layers: List["EnrichedNetworkNode"] = attrib(init=False)

    @property
    def receptive_field_sizes(self) -> List[int]:
        return [elem.receptive_field for elem in self.receptive_field_info]

    def _group_by_dim(
        self, rf_sizes: List[Union[Sequence[int], int]]
    ) -> Dict[Union[int, str], List[int]]:
        """Find the minimum receptive field size.

        Args:
            rf_sizes:   A list of receptive field sizes.

        Returns:
            The minimum size.

        """
        if all(
            [
                isinstance(elem, int) and not isinstance(elem, Sequence)
                for elem in rf_sizes
            ]
        ):
            return {"all": rf_sizes}
        else:
            result: Dict[Union[int, str], List[int]] = {"all": []}
            for rf_size in rf_sizes:
                if isinstance(rf_size, Sequence):
                    for i, size in enumerate(rf_size):
                        if i not in result:
                            result[i] = []
                        result[i].append(size)
                else:
                    result["all"].append(rf_size)
            return result

    @staticmethod
    def _apply_function_on_receptive_field_groups(
        groups: Dict[Union[int, str], List[int]], func: Callable[[List[int]], int]
    ) -> Union[Sequence[int], int]:
        """Apply a function on a list of receptive field sizes.

        Args:
            groups:    A dictionary of receptive field sizes.
            func:      The function to apply.

        Returns:
            The result of the function.

        """
        if "all" in groups:
            scalars: List[int] = groups.pop("all")
            if len(groups) == 0:
                return func(scalars)
        else:
            raise ValueError(
                "'all'-key not in sequence for receptive field computation"
            )

        result: List[int] = []
        max_dim: int = max(groups.keys())
        for i in range(max_dim + 1):
            if i not in groups:
                raise ValueError(f"Missing dimension {i}")
            dim: List[int] = groups[i] + scalars
            result.append(func(dim))
        return tuple(result)

    def _apply_function_on_receptive_field_sizes(
        self, func: Callable[[List[int]], int]
    ) -> Union[Sequence[int], int]:
        """Apply a function on the receptive field sizes.

        Args:
            func:  The function to apply.

        Returns:
            The result of the function.

        """
        return self._apply_function_on_receptive_field_groups(
            self._group_by_dim(self.receptive_field_sizes), func
        )

    def _receptive_field_min(self):
        return self._apply_function_on_receptive_field_sizes(
            lambda x: min(x, default=0)
        )
        # return min(self.receptive_field_sizes, default=0)

    def _receptive_field_max(self):
        return self._apply_function_on_receptive_field_sizes(
            lambda x: max(x, default=0)
        )

        # return max(self.receptive_field_sizes, default=0)

    @property
    def kernel_size(self):
        return self.layer_info.kernel_size

    @property
    def stride_size(self):
        return self.layer_info.stride_size

    @predecessors.validator
    def verify_predecessor_list(
        self, attribute: str, value: List["EnrichedNetworkNode"]
    ) -> None:
        if len(value) != 0:
            if not all([isinstance(node, EnrichedNetworkNode) for node in value]):
                raise ValueError(f"{attribute} must be a list of EnrichedNetworkNodes")

    def __attrs_post_init__(self):
        infos: Set[ReceptiveFieldInfo] = set()
        if self.layer_info.kernel_size == np.inf:
            infos.update([ReceptiveFieldInfo(receptive_field=np.inf, multiplicator=1)])
        elif len(self.predecessors):
            for pred in self.predecessors:
                infos.update(pred.receptive_field_info)
        else:
            infos.update([ReceptiveFieldInfo(receptive_field=1, multiplicator=1)])
        rf_infos = compute_receptive_field_sizes(infos, self.layer_info)
        rf_infos_filtered = self.receptive_field_info_filter(rf_infos)
        object.__setattr__(
            self,
            "receptive_field_info",
            rf_infos_filtered,
        )
        object.__setattr__(self, "receptive_field_min", self._receptive_field_min())
        object.__setattr__(self, "receptive_field_max", self._receptive_field_max())
        object.__setattr__(
            self,
            "all_layers",
            [] if not self.predecessors else self.predecessors[0].all_layers,
        )
        self.all_layers.append(self)
        for pred in self.predecessors:
            pred.succecessors.append(self)

    def is_border(
        self,
        input_resolution: Union[int, Sequence[int]],
        receptive_field_provider: Callable[
            ["EnrichedNetworkNode"], Union[float, int]
        ] = receptive_field_provider,
    ) -> bool:
        """Checks if this layer is a border layer.
        A border layer is predicted not advance the
        intermediate solution
        quality and can thus be considered "dead weight".

        Args:
            input_resolution:           the input resolution to check for
            receptive_field_provider:   a provider function that produces a
                                        receptive field value, from which the
                                        border-layer decision can be derived.
                                        By default the minimum receptive field size
                                        will yielded from the set of
                                        receptive field sizes, which is currently
                                        the most reliable  way of predicting
                                        unproductive layers.

        Returns:
            True if this layer is predicted to be unproductive
            for the given input resolution, else False.

        """
        # the border layer is defined as the layer that receives
        # all inputs with a receptive field size
        # SMALLER than the input resolution
        direct_predecessors = [
            np.all(
                np.asarray(input_resolution)
                <= np.asarray(receptive_field_provider(pred))
            )
            for pred in self.predecessors
        ]
        # of course, this means that this layer also needs to fullfill this property
        own = np.all(
            np.asarray(input_resolution) <= np.asarray(self.receptive_field_min)
        )
        # additionally (only relevant for multipath architectures)
        # all following layer are border layers as well
        # successors = [
        #    input_resolution <= result
        #    for result in self._apply_function_to_all_successors(
        #        receptive_field_provider
        #    )
        # ]
        # in short all direct predecessors,
        # the layer itself and all following layers have a receptive field size
        # GREATER than the input resolution
        # return all(direct_predecessors) and own and all(successors)
        return all(direct_predecessors) and own  # and all(successors)

    def is_in(self, container: Union[List[Node], Dict[Node, Any]]) -> bool:
        """Checks if this node is inside a an iterable collection.
        Args:
            container: dictionary with node as key or list of EnrichedNetworkNodes.

        Returns:
            True if the node is contained in the collection, else False.

        """
        if isinstance(container, list):
            return any(id(self) == id(node) for node in container)
        else:
            return any(id(self) == id(node) for node in container.keys())

    def __repr__(self):
        pred_names = [pred.name for pred in self.predecessors]
        succ_names = [succ.name for succ in self.succecessors]
        return (
            f"EnrichedNetworkNode(\n"
            f"\tname={self.name},\n"
            f"\tpredecessors={pred_names},\n"
            f"\tsuccessors={succ_names},\n"
            f"\tlayer_info={self.layer_info},\n"
            f"\treceptive_field_min={self.receptive_field_min},\n"
            f"\treceptive_field_max={self.receptive_field_max},\n"
            f"\treceptive_field_sizes={self.receptive_field_sizes},\n"
            f")\n"
        )
