from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
from attr import attrib, attrs

from rfa_toolbox.domain import Layer, Node


def receptive_field_provider_with_1x1_handling(
    node: "EnrichedNetworkNode",
) -> Optional[int]:
    return node.receptive_field_min if node.layer_type.kernel_size > 1 else np.inf


def receptive_field_provider(node: "EnrichedNetworkNode") -> Optional[int]:
    return node.receptive_field_min


@attrs(auto_attribs=True, frozen=True, slots=True)
class ReceptiveFieldInfo:
    receptive_field: int
    multiplicator: int


@attrs(auto_attribs=True, frozen=True, slots=True)
class LayerDefinition(Layer):
    """The standard representation of a neural network layer.
    Contains information needed for receptive field computation.

    Args:
        name:           name of the layer
        kernel_size:    size of the kernel, None if this is a dense-layer
        stride_size:    the stride size the kernel is convolved. None for dense-layers.

    """

    name: str
    kernel_size: Optional[int] = attrib(converter=lambda x: np.inf if x is None else x)
    stride_size: Optional[int] = attrib(converter=lambda x: 1 if x is None else x)

    @kernel_size.validator
    def validate_kernel_size(self, attribute: str, value: int) -> None:
        if value is not None and value < 1:
            raise ValueError(
                f"{attribute} must be greater than 0 or "
                f"infinite (which indicates a dense layer)"
            )

    @stride_size.validator
    def validate_stride_size(self, attribute: str, value: int) -> None:
        if value is not None and value < 1:
            raise ValueError(
                f"{attribute} must be greater than 0 "
                f"(choose 1, if this is a dense layer)"
            )

    @classmethod
    def from_dict(cls, config) -> "LayerDefinition":
        return LayerDefinition(**config)

    def to_dict(self) -> Dict[str, Union[int, str]]:
        return {
            "name": self.name,
            "kernel_size": self.kernel_size,
            "stride_size": self.stride_size,
        }


def compute_receptive_field_sizes(
    receptive_field_info: Set[ReceptiveFieldInfo], layer_info: Layer
) -> Tuple[ReceptiveFieldInfo]:
    result: List[ReceptiveFieldInfo] = list()
    for rf_info in receptive_field_info:
        receptive_field = rf_info.receptive_field + (
            (layer_info.kernel_size - 1) * rf_info.multiplicator
        )
        multiplicator = layer_info.stride_size * rf_info.multiplicator
        new_info = ReceptiveFieldInfo(
            receptive_field=receptive_field, multiplicator=multiplicator
        )
        result.append(new_info)

    return tuple(result)


@attrs(auto_attribs=True, frozen=True, slots=True, hash=False, repr=False)
class EnrichedNetworkNode(Node):
    name: str
    layer_info: LayerDefinition
    predecessors: List["EnrichedNetworkNode"] = attrib(converter=tuple)
    succecessors: List["EnrichedNetworkNode"] = attrib(
        init=False, factory=list, eq=False
    )
    receptive_field_info: Tuple[ReceptiveFieldInfo] = attrib(init=False)
    receptive_field_min: int = attrib(init=False)
    receptive_field_max: int = attrib(init=False)

    @property
    def receptive_field_sizes(self) -> List[int]:
        return [elem.receptive_field for elem in self.receptive_field_info]

    def _receptive_field_min(self):
        return min(self.receptive_field_sizes, default=0)

    def _receptive_field_max(self):
        return max(self.receptive_field_sizes, default=0)

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
        object.__setattr__(
            self,
            "receptive_field_info",
            compute_receptive_field_sizes(infos, self.layer_info),
        )
        object.__setattr__(self, "receptive_field_min", self._receptive_field_min())
        object.__setattr__(self, "receptive_field_max", self._receptive_field_max())
        for pred in self.predecessors:
            pred.succecessors.append(self)

    def is_border(
        self,
        input_resolution: int,
        receptive_field_provider: Callable[
            ["EnrichedNetworkNode"], Union[float, int]
        ] = receptive_field_provider,
    ) -> bool:
        # the border layer is defined as the layer that receives
        # all inputs with a receptive field size
        # SMALLER than the input resolution
        direct_predecessors = [
            input_resolution <= receptive_field_provider(pred)
            for pred in self.predecessor_list
        ]
        # of course, this means that this layer also needs to fullfill this property
        own = input_resolution <= self.receptive_field_min
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
