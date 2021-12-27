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
        if "Conv2d" in name.split(".")[-1]:
            print(name)
            return True
        else:
            return False

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
        filters = conv_layer.out_channels
        return LayerDefinition(
            name=f"{name} {kernel_size}x{kernel_size}",
            kernel_size=kernel_size,
            stride_size=stride_size,
            filters=filters,
        )


@attrs(auto_attribs=True, frozen=True, slots=True)
class AnyConv(Conv2d):
    def can_handle(self, name: str) -> bool:
        if "Conv2d" in name.split(".")[-1]:
            print(name)
            return True
        else:
            return False


@attrs(auto_attribs=True, frozen=True, slots=True)
class AnyPool(Conv2d):
    def can_handle(self, name: str) -> bool:
        working_name = name.split(".")[-1]
        return "Pool" in working_name and "Adaptive" not in working_name

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
            name=f"{name} {kernel_size}x{kernel_size}",
            kernel_size=kernel_size,
            stride_size=stride_size,
        )


@attrs(auto_attribs=True, frozen=True, slots=True)
class AnyAdaptivePool(Conv2d):
    def can_handle(self, name: str) -> bool:
        return "Pool" in name and "adaptive" in name

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
