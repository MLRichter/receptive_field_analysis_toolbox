import torch
from attr import attrs

from rfa_toolbox.encodings.pytorch.domain import LayerInfoHandler
from rfa_toolbox.graphs import LayerDefinition


def obtain_module_with_resolvable_string(
    resolvable: str, model: torch.nn.Module
) -> torch.nn.Module:
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
    def can_handle(self, name: str) -> bool:
        return "Conv2d" in name

    def __call__(
        self, model: torch.nn.Module, resolvable_string: str, name: str
    ) -> LayerDefinition:
        conv_layer = obtain_module_with_resolvable_string(resolvable_string, model)
        kernel_size = (
            conv_layer.kernel_size
            if isinstance(conv_layer.kernel_size, int)
            else conv_layer.kernel_size[0]
        )
        stride_size = (
            conv_layer.stride
            if isinstance(conv_layer.stride, int)
            else conv_layer.stride[0]
        )
        return LayerDefinition(
            name=f"Conv{kernel_size}x{kernel_size}",
            kernel_size=kernel_size,
            stride_size=stride_size,
        )


@attrs(auto_attribs=True, frozen=True, slots=True)
class AnyConv(Conv2d):
    def can_handle(self, name: str) -> bool:
        return "conv" in name.lower()


@attrs(auto_attribs=True, frozen=True, slots=True)
class AnyPool(Conv2d):
    def can_handle(self, name: str) -> bool:
        return "pool" in name.lower() and "adaptive" not in name.lower()

    def __call__(
        self, model: torch.nn.Module, resolvable_string: str, name: str
    ) -> LayerDefinition:
        conv_layer = obtain_module_with_resolvable_string(resolvable_string, model)
        kernel_size = (
            conv_layer.kernel_size
            if isinstance(conv_layer.kernel_size, int)
            else conv_layer.kernel_size[0]
        )
        stride_size = (
            conv_layer.stride
            if isinstance(conv_layer.stride, int)
            else conv_layer.stride[0]
        )
        return LayerDefinition(
            name=f"{name}{kernel_size}x{kernel_size}",
            kernel_size=kernel_size,
            stride_size=stride_size,
        )


@attrs(auto_attribs=True, frozen=True, slots=True)
class AnyAdaptivePool(Conv2d):
    def can_handle(self, name: str) -> bool:
        return "pool" in name.lower() and "adaptive" in name.lower()

    def __call__(
        self, model: torch.nn.Module, resolvable_string: str, name: str
    ) -> LayerDefinition:
        kernel_size = None
        stride_size = 1
        return LayerDefinition(
            name=name, kernel_size=kernel_size, stride_size=stride_size
        )


@attrs(auto_attribs=True, frozen=True, slots=True)
class LinearHandler(LayerInfoHandler):
    def can_handle(self, name: str) -> bool:
        return "Linear" in name

    def __call__(
        self, model: torch.nn.Module, resolvable_string: str, name: str
    ) -> LayerDefinition:
        kernel_size = None
        stride_size = 1
        return LayerDefinition(
            name="DenseLayer", kernel_size=kernel_size, stride_size=stride_size
        )


@attrs(auto_attribs=True, frozen=True, slots=True)
class AnyHandler(LayerInfoHandler):
    def can_handle(self, name: str) -> bool:
        return True

    def __call__(
        self, model: torch.nn.Module, resolvable_string: str, name: str
    ) -> LayerDefinition:
        kernel_size = 1
        stride_size = 1
        if "(" in resolvable_string and ")" in name:
            # print(result)
            result = name.split("(")[-1].replace(")", "")
        else:
            result = f"{name.split('.')[-1]}"

        return LayerDefinition(
            name=result, kernel_size=kernel_size, stride_size=stride_size
        )
