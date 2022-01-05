import re
from typing import Callable, Optional

from rfa_toolbox.graphs import LayerDefinition


def convolutional_config_from_string(
    config_string: str,
    pattern=re.compile(r"Conv\dx\d"),
    kernel_size_extractor: Callable[[str], int] = lambda x: int(
        x.replace("Conv", "").split("x")[0]
    ),
    stride_size_extractor: Callable[[str], int] = lambda x: 1,
) -> Optional[LayerDefinition]:
    if pattern.match(config_string):
        name = config_string
        kernel_size = kernel_size_extractor(config_string)
        stride_size = stride_size_extractor(config_string)
        return LayerDefinition(name, kernel_size, stride_size)
    else:
        return None


def obtain_paths_from_graph():
    ...
