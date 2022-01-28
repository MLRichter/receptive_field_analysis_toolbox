import warnings


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
        kenrel_size:    kernel size of the convolution, this is a default value
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
