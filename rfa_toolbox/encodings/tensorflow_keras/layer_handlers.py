from rfa_toolbox.graphs import LayerDefinition

try:
    from typing import Any, Dict, Protocol
except ImportError:
    from typing_extensions import Protocol

from attr import attrs


class LayerInfoHandler(Protocol):
    """Creates a LayerDefinition from the model and a resolvable string."""

    def can_handle(self, node: Dict[str, Any]) -> bool:
        """Checks if this handler can process the
        node in the compute graph of the model.

        Args:
            node: the node in question

        Returns:
            True if the node can be processed into a
            valid LayerDefinition by this handler.
        """
        ...

    def __call__(self, node: Dict[str, Any]) -> LayerDefinition:
        """Transform the json-representation of a compute node in the tensorflow-graph
        into a LayerDefinition.

        Args:
            node: the node in question

        Returns:
            A LayerDefinition that reflects the properties of the layer.
        """
        ...


@attrs(frozen=True, slots=True, auto_attribs=True)
class KernelBasedHandler(LayerInfoHandler):
    def can_handle(self, node: Dict[str, Any]) -> bool:
        """Handles only layers featuring a kernel_size and filters"""
        return "kernel_size" in node["config"]

    def __call__(self, node: Dict[str, Any]) -> LayerDefinition:
        """Transform the json-representation
        of a compute node in the tensorflow-graph"""
        name = (
            f"{node['class_name']} "
            f"{'x'.join(str(x) for x in node['config']['kernel_size'])} "
            f"/ {node['config']['strides']}"
        )
        filters = None if "filters" not in node["config"] else node["config"]["filters"]
        return LayerDefinition(
            name=name,
            kernel_size=node["config"]["kernel_size"],
            stride_size=node["config"]["strides"],
            filters=filters,
        )


@attrs(frozen=True, slots=True, auto_attribs=True)
class PoolingBasedHandler(LayerInfoHandler):
    def can_handle(self, node: Dict[str, Any]) -> bool:
        """Handles only layers featuring a pool_size"""
        return "pool_size" in node["config"]

    def __call__(self, node: Dict[str, Any]) -> LayerDefinition:
        """Transform the json-representation of a
        compute node in the tensorflow-graph"""
        name = (
            f"{node['class_name']} "
            f"{'x'.join(str(x) for x in node['config']['pool_size'])} "
            f"/ {node['config']['strides']}"
        )
        return LayerDefinition(
            name=name,
            kernel_size=node["config"]["pool_size"],
            stride_size=node["config"]["strides"],
        )


@attrs(frozen=True, slots=True, auto_attribs=True)
class DenseHandler(LayerInfoHandler):
    def can_handle(self, node: Dict[str, Any]) -> bool:
        """Handles only layers feature units as attribute"""
        return "units" in node["config"]

    def __call__(self, node: Dict[str, Any]) -> LayerDefinition:
        """Transform the json-representation
        of a compute node in the tensorflow-graph"""
        name = node["class_name"]
        return LayerDefinition(
            name=name, kernel_size=1, stride_size=1, units=node["config"]["units"]
        )


@attrs(frozen=True, slots=True, auto_attribs=True)
class GlobalPoolingHandler(LayerInfoHandler):
    def can_handle(self, node: Dict[str, Any]) -> bool:
        """Handles only layers feature units as attribute"""
        return "Global" in node["class_name"] and "Pooling" in node["class_name"]

    def __call__(self, node: Dict[str, Any]) -> LayerDefinition:
        """Transform the json-representation
        of a compute node in the tensorflow-graph"""
        name = node["class_name"]
        return LayerDefinition(name=name)


@attrs(frozen=True, slots=True, auto_attribs=True)
class FlattenHandler(LayerInfoHandler):
    def can_handle(self, node: Dict[str, Any]) -> bool:
        """Handles only layers feature units as attribute"""
        return "Flatten" in node["class_name"]

    def __call__(self, node: Dict[str, Any]) -> LayerDefinition:
        """Transform the json-representation
        of a compute node in the tensorflow-graph"""
        name = node["class_name"]
        return LayerDefinition(name=name, kernel_size=None)


@attrs(frozen=True, slots=True, auto_attribs=True)
class InputHandler(LayerInfoHandler):
    def can_handle(self, node: Dict[str, Any]) -> bool:
        """This is strictly meant for handling input nodes"""
        return node["class_name"] == "InputLayer"

    def __call__(self, node: Dict[str, Any]) -> LayerDefinition:
        """Transform the json-representation of
        a compute node in the tensorflow-graph"""
        return LayerDefinition(name=node["class_name"], kernel_size=1, stride_size=1)


@attrs(frozen=True, slots=True, auto_attribs=True)
class AnyHandler(LayerInfoHandler):
    def can_handle(self, node: Dict[str, Any]) -> bool:
        """This is a catch-all handler"""
        return True

    def __call__(self, node: Dict[str, Any]) -> LayerDefinition:
        """Transform the json-representation of a
        compute node in the tensorflow-graph"""
        return LayerDefinition(name=node["class_name"], kernel_size=1, stride_size=1)
