from collections import Sequence
from typing import Callable

import numpy as np
import torch
from attr import attrs

from rfa_toolbox.encodings.pytorch.domain import LayerInfoHandler
from rfa_toolbox.graphs import LayerDefinition


def obtain_module_with_resolvable_string(
    resolvable: str, model: torch.nn.Module
) -> torch.nn.Module:
    """Attempts to find the module inside a PyTorch-model based on a
    resolvable-string extracted from
    a JIT-compiled version of the same model.

    Args:
        resolvable: the resolvable string
        model:      the PyTorch-model instance in which the layer can be found.

    Returns:
        PyTorch-Module extracted from the model.

    Raises:
        ValueError if the module cannot be extracted from the module.

    """
    current = model
    for elem in resolvable.split("."):
        if elem.isnumeric():
            current = current[int(elem)]
        else:
            current = getattr(current, elem, None)
            if current is None:
                raise ValueError(f"Cannot resolve '{current}.{elem}' from {resolvable}")
    return current


@attrs(auto_attribs=True, frozen=True, slots=True)
class Conv2d(LayerInfoHandler):
    """This handler explicitly operated on the torch.nn.Conv2d-Layer."""

    def can_handle(self, name: str) -> bool:
        if "Conv2d" in name.split(".")[-1]:
            return True
        else:
            return False

    def __call__(
        self, model: torch.nn.Module, resolvable_string: str, name: str, **kwargs
    ) -> LayerDefinition:
        conv_layer = obtain_module_with_resolvable_string(resolvable_string, model)
        kernel_size = (
            conv_layer.kernel_size
            # if isinstance(conv_layer.kernel_size, int)
            # else conv_layer.kernel_size[0]
        )
        stride_size = (
            conv_layer.stride
            # if isinstance(conv_layer.stride, int)
            # else conv_layer.stride[0]
        )
        filters = conv_layer.out_channels
        if not isinstance(kernel_size, Sequence) and not isinstance(
            kernel_size, np.ndarray
        ):
            kernel_size_name = f"{kernel_size}x{kernel_size}"
        else:
            kernel_size_name = "x".join([str(k) for k in kernel_size])
        final_name = f"{name} {kernel_size_name} / {stride_size}"
        return LayerDefinition(
            name=final_name,  # f"{name} {kernel_size}x{kernel_size}",
            kernel_size=kernel_size,
            stride_size=stride_size,
            filters=filters,
        )


@attrs(auto_attribs=True, frozen=True, slots=True)
class ConvNormActivationHandler(LayerInfoHandler):
    """This handler explicitly operated on the torch.nn.Conv2d-Layer."""

    def can_handle(self, name: str) -> bool:
        if "ConvNormActivation" in name.split(".")[-1]:
            return True
        else:
            return False

    def __call__(
        self, model: torch.nn.Module, resolvable_string: str, name: str, **kwargs
    ) -> LayerDefinition:
        module = obtain_module_with_resolvable_string(resolvable_string, model)
        submodules = list(module.modules())
        conv_layer = submodules[1]
        activation = "" if len(submodules) < 4 else f"-{type(submodules[3]).__name__}"
        kernel_size = (
            conv_layer.kernel_size
            # if isinstance(conv_layer.kernel_size, int)
            # else conv_layer.kernel_size[0]
        )
        stride_size = (
            conv_layer.stride
            # if isinstance(conv_layer.stride, int)
            # else conv_layer.stride[0]
        )
        filters = conv_layer.out_channels
        kernel_size_name = (
            f"{kernel_size}x{kernel_size}"
            if not isinstance(kernel_size, Sequence)
            and not isinstance(kernel_size, np.ndarray)
            else "x".join([str(k) for k in kernel_size])
        )
        final_name = f"Conv-Norm{activation} {kernel_size_name} / {stride_size}"
        return LayerDefinition(
            name=final_name,  # f"{name} {kernel_size}x{kernel_size}",
            kernel_size=kernel_size,
            stride_size=stride_size,
            filters=filters,
        )


@attrs(auto_attribs=True, frozen=True, slots=True)
class AnyConv(Conv2d):
    """Extract layer information in convolutional layers."""

    def can_handle(self, name: str) -> bool:
        if "Conv" in name.split(".")[-1]:
            return True
        else:
            return False


@attrs(auto_attribs=True, frozen=True, slots=True)
class AnyPool(Conv2d):
    """Extract layer information in any pooling layer that is not adaptive."""

    def can_handle(self, name: str) -> bool:
        working_name = name.split(".")[-1]
        return "Pool" in working_name and "Adaptive" not in working_name

    def __call__(
        self, model: torch.nn.Module, resolvable_string: str, name: str, **kwargs
    ) -> LayerDefinition:
        conv_layer = obtain_module_with_resolvable_string(resolvable_string, model)
        kernel_size = conv_layer.kernel_size

        stride_size = conv_layer.stride
        if not isinstance(kernel_size, Sequence) and not isinstance(
            kernel_size, np.ndarray
        ):
            kernel_size_name = f"{kernel_size}x{kernel_size}"
        else:
            kernel_size_name = "x".join([str(k) for k in kernel_size])
        final_name = f"{name} {kernel_size_name} / {stride_size}"
        return LayerDefinition(
            name=final_name,  # f"{name} {kernel_size}x{kernel_size}",
            kernel_size=kernel_size,
            stride_size=stride_size,
        )


@attrs(auto_attribs=True, frozen=True, slots=True)
class AnyAdaptivePool(Conv2d):
    """Extract information from adaptive pooling layers."""

    def can_handle(self, name: str) -> bool:
        return "pool" in name.lower() and "adaptive" in name.lower()

    def __call__(
        self, model: torch.nn.Module, resolvable_string: str, name: str, **kwargs
    ) -> LayerDefinition:
        kernel_size = None
        stride_size = 1
        return LayerDefinition(
            name=name, kernel_size=kernel_size, stride_size=stride_size
        )


@attrs(auto_attribs=True, frozen=True, slots=True)
class SqueezeExcitationHandler(Conv2d):
    """Extract information from adaptive pooling layers."""

    def can_handle(self, name: str) -> bool:
        return "SqueezeExcitation" in name

    def __call__(
        self, model: torch.nn.Module, resolvable_string: str, name: str, **kwargs
    ) -> LayerDefinition:
        kernel_size = 1
        stride_size = 1
        return LayerDefinition(
            name=name, kernel_size=kernel_size, stride_size=stride_size
        )


@attrs(auto_attribs=True, frozen=True, slots=True)
class GenericLayerTypeHandler(LayerInfoHandler):
    """Extracts information from linear (fully connected) layers."""

    class_name: str
    name_provider: Callable[[torch.nn.Module, str], str]
    kernel_size_provider: Callable[[torch.nn.Module], int]
    stride_size_provider: Callable[[torch.nn.Module], int]
    filters_provider: Callable[[torch.nn.Module], int]
    units_provider: Callable[[torch.nn.Module], int]

    def can_handle(self, name: str) -> bool:
        return self.class_name in name

    def __call__(
        self, model: torch.nn.Module, resolvable_string: str, name: str, **kwargs
    ) -> LayerDefinition:
        layer = obtain_module_with_resolvable_string(resolvable_string, model)
        return LayerDefinition(
            name=self.name_provider(layer, self.class_name),
            kernel_size=self.kernel_size_provider(layer),
            stride_size=self.stride_size_provider(layer),
            units=self.units_provider(layer),
            filters=self.filters_provider(layer),
        )


@attrs(auto_attribs=True, frozen=True, slots=True)
class LinearHandler(LayerInfoHandler):
    """Extracts information from linear (fully connected) layers."""

    def can_handle(self, name: str) -> bool:
        return "Linear" in name

    def __call__(
        self, model: torch.nn.Module, resolvable_string: str, name: str, **kwargs
    ) -> LayerDefinition:
        kernel_size = 1
        stride_size = 1
        features = obtain_module_with_resolvable_string(
            resolvable_string, model
        ).out_features
        return LayerDefinition(
            name="Fully Connected",
            kernel_size=kernel_size,
            stride_size=stride_size,
            units=features,
        )


@attrs(auto_attribs=True, frozen=True, slots=True)
class FlattenHandler(LayerInfoHandler):
    """Extracts information from linear (fully connected) layers."""

    def can_handle(self, name: str) -> bool:
        return "flatten" in name.lower()

    def __call__(
        self, model: torch.nn.Module, resolvable_string: str, name: str, **kwargs
    ) -> LayerDefinition:
        kernel_size = None
        stride_size = 1
        return LayerDefinition(
            name="Flatten", kernel_size=kernel_size, stride_size=stride_size
        )


@attrs(auto_attribs=True, frozen=False, slots=True)
class FunctionalKernelHandler(LayerInfoHandler):
    """This handler is explicitly build for functional pooling layers.
    It will default to some often used kernel and stride size but will also produce
    a warning, since the correct sizes of kernel and strides cannot be extracted.
    Therefore this handler is more for aiding in the developer in not falling for
    faulty visualizations.
    """

    coerce: bool = False

    def can_handle(self, name: str) -> bool:
        return "pool" in name.split(".")[-1] or "conv" in name.split(".")[-1]

    def __call__(
        self, model: torch.nn.Module, resolvable_string: str, name: str, **kwargs
    ) -> LayerDefinition:

        if "(" in resolvable_string and ")" in name:
            result = name.split("(")[-1].replace(")", "")
        else:
            result = f"{name.split('.')[-1]}"

        kernel_size = kwargs["kernel_size"]
        stride_size = kwargs["stride_size"]

        if not isinstance(kernel_size, Sequence) and not isinstance(
            kernel_size, np.ndarray
        ):
            kernel_size_name = f"{kernel_size}x{kernel_size}"
        else:
            kernel_size_name = "x".join([str(k) for k in kernel_size])
        final_name = (
            f"{result} {kernel_size_name} / "
            f"{stride_size if isinstance(stride_size, int) else tuple(stride_size)}"
        )

        return LayerDefinition(
            name=final_name,
            kernel_size=kernel_size,
            stride_size=stride_size,
        )


@attrs(auto_attribs=True, frozen=True, slots=True)
class AnyHandler(LayerInfoHandler):
    """This handler is a catch-all handler, which transform
    any layer into an EnrichedNetworkNode.
    However, this Handler will assume a kernel-size of 1 and a
    stride-size of 1 and will not attempt to extract
    any information on the number of filters or units.
    Therefore, it is mostly meant as a "last-resort"-handler
    for layers that are not handleable by any other handler.
    """

    def can_handle(self, name: str) -> bool:
        return True

    def __call__(
        self, model: torch.nn.Module, resolvable_string: str, name: str, **kwargs
    ) -> LayerDefinition:
        kernel_size = 1
        stride_size = 1
        if "(" in resolvable_string and ")" in name:
            result = name.split("(")[-1].replace(")", "")
        else:
            result = f"{name.split('.')[-1]}"

        return LayerDefinition(
            name=result, kernel_size=kernel_size, stride_size=stride_size
        )
