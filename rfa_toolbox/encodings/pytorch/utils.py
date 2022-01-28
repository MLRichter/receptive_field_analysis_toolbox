import warnings
from typing import Callable

import torch

from rfa_toolbox.encodings.pytorch.intermediate_graph import RESOLVING_STRATEGY
from rfa_toolbox.encodings.pytorch.layer_handlers import GenericLayerTypeHandler


def toggle_coerce_torch_functional(
    coerce: bool = True,
    kernel_size: int = 1,
    stride_size: int = 1,
    handler_idx: int = -2,
):
    """Toogle the raise condition of the torch.nn.functional-calls.
    Disabling this condition is strongly discouraged, since there is
    some unknown behavior associated with torch.nn.functional.

    Args:
        coerce:         True by default, enabled or disables a raise-condition
                        whenever a call from torch.nn.functional is encountered
        kernel_size:    kernel size of the convolution, this is a default value
                        to be assumed for all_functional
                        layers. Default is a kernel_size of 1,
                        since it does not change the receptive field expansion.
        stride_size:    stride size of the convolution, this is a default value
                        to be assumed for all_functional layers.
                        Default is a stride_size of 1, since it does not change
                        the receptive field expansion.
        handler_idx:    index of the handler in the list of handlers
    """
    warnings.warn(
        "This function is deprecated and is no longer needed, "
        "you may remove it from your code",
        DeprecationWarning,
    )


def add_custom_layer_handler(
    class_name: str,
    name_handler: Callable[[torch.nn.Module, str], str] = lambda x, y: y,
    kernel_size_provider: Callable[[torch.nn.Module], int] = lambda x: 1,
    stride_size_provider: Callable[[torch.nn.Module], int] = lambda x: 1,
    filters_provider: Callable[[torch.nn.Module], int] = lambda x: None,
    units_provider: Callable[[torch.nn.Module], int] = lambda x: None,
) -> None:
    """Add a custom layer handler to the list of handlers.

    Args:
        name_handler:           function that returns the name of the layer
        class_name:             name of the class that is handled by the handler
        kernel_size_provider:   function that returns the kernel size of the layer
        stride_size_provider:   function that returns the stride size of the layer
        filters_provider:       function that returns the number of filters of the layer
        units_provider:         function that returns the number of units of the layer
    """
    handler = GenericLayerTypeHandler(
        class_name=class_name,
        name_provider=name_handler,
        kernel_size_provider=kernel_size_provider,
        stride_size_provider=stride_size_provider,
        filters_provider=filters_provider,
        units_provider=units_provider,
    )
    RESOLVING_STRATEGY.insert(0, handler)
