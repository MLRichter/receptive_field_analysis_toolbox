from typing import List

import numpy as np

from rfa_toolbox.graphs import EnrichedNetworkNode


def _remove_duplicates(nodes: List[EnrichedNetworkNode]) -> List[EnrichedNetworkNode]:
    result = []
    for node in nodes:
        if node.is_in(result):
            continue
        else:
            result.append(node)
    return result


def obtain_all_nodes(
    output_node: EnrichedNetworkNode, search_from_output: bool = False
) -> List[EnrichedNetworkNode]:
    if search_from_output:
        all_nodes = [output_node]
        for pred in output_node.predecessors:
            all_nodes.extend(obtain_all_nodes(pred, False))
        return _remove_duplicates(all_nodes)
    else:
        return output_node.all_layers


def obtain_border_layers(
    output_node: EnrichedNetworkNode, input_resolution: int, filter_dense: bool = True
) -> List[EnrichedNetworkNode]:
    all_nodes = obtain_all_nodes(output_node)
    # print(all_nodes)
    result = [node for node in all_nodes if node.is_border(input_resolution)]
    return filters_non_convolutional_node(result) if filter_dense else result


def obtain_all_critical_layers(
    output_node: EnrichedNetworkNode, input_resolution: int, filter_dense: bool = True
) -> List[EnrichedNetworkNode]:
    all_nodes = obtain_all_nodes(output_node)
    result = [node for node in all_nodes if node.receptive_field_min > input_resolution]
    return filters_non_convolutional_node(result) if filter_dense else result


def filters_non_convolutional_node(
    nodes: List[EnrichedNetworkNode],
) -> List[EnrichedNetworkNode]:
    return [node for node in nodes if node.layer_info.kernel_size != np.inf]
