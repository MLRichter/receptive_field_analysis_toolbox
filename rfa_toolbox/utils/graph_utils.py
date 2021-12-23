from typing import Dict, List

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


def obtain_all_nodes(output_node: EnrichedNetworkNode) -> List[EnrichedNetworkNode]:
    all_nodes = [output_node]
    for pred in output_node.predecessors:
        all_nodes.extend(obtain_all_nodes(pred))
    return _remove_duplicates(all_nodes)


def obtain_output_nodes(node: EnrichedNetworkNode) -> List[EnrichedNetworkNode]:
    if not node.succecessors:
        return [node]
    result = []
    for successor in node.succecessors:
        output_nodes = obtain_output_nodes(successor)
        result.extend(
            [out_node for out_node in output_nodes if not out_node.is_in(result)]
        )
    return result


def obtain_object_id_to_receptive_field_mapping(
    output_node: EnrichedNetworkNode,
) -> Dict[int, int]:
    all_nodes = obtain_all_nodes(output_node)
    return {id(node): node.receptive_field_min for node in all_nodes}


def obtain_border_layers(
    output_node: EnrichedNetworkNode, input_resolution: int, filter_dense: bool = True
) -> List[EnrichedNetworkNode]:
    all_nodes = obtain_all_nodes(output_node)
    result = [node for node in all_nodes if node.is_border(input_resolution)]
    return filters_non_convolutional_node(result) if filter_dense else result


def obtain_all_critical_layer(
    output_node: EnrichedNetworkNode, input_resolution: int, filter_dense: bool = True
) -> List[EnrichedNetworkNode]:
    all_nodes = obtain_all_nodes(output_node)
    result = [node for node in all_nodes if node.receptive_field_min > input_resolution]
    return filters_non_convolutional_node(result) if filter_dense else result


def filters_non_convolutional_node(
    nodes: List[EnrichedNetworkNode],
) -> List[EnrichedNetworkNode]:
    return [node for node in nodes if node.layer_info.validate_kernel_size != np.inf]
