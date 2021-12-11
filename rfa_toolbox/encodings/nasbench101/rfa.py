from typing import List, Tuple

from rfa_toolbox.network_components import LayerDefinition

NASBenchNetwork = Tuple[List[List[bool]], List[str]]


def compute_receptive_field(
    network_config: NASBenchNetwork, layer_definitions: List[LayerDefinition]
):
    ...
