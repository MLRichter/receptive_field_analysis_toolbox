from typing import Tuple, List, Union

NASBenchNetwork = Tuple[List[List[bool]], List[str]]


def compute_receptive_field(network_config: NASBenchNetwork, layer_definitions: List[LayerDefinition]):
    ...
