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
    """Fetch all nodes from a single node of the compute graph.

    Args:
        output_node:            output node of the graph
        search_from_output:     False by default. If True,
                                the nodes will be searched
                                using the BFS-Algorithm. If False,
                                the internal registry of the node will be used,
                                which may be dangerous if more than one
                                input-node exists.

    Returns:
        A List containing all EnrichedNetworkNodes.

    """
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
    """Obtain all border layers.

    Args:
        output_node:        a node of the compute graph
        input_resolution:   the input resolution for which the
                            border layer should be computed
        filter_dense:       exclude all layers with infinite receptive field size
                            (essentially all layers that are fully connected
                            or successors of fully connected layers)
                            This is True by default.
    Returns:
        All layers predicted to be unproductive.

    """
    all_nodes = obtain_all_nodes(output_node)
    result = [node for node in all_nodes if node.is_border(input_resolution)]
    return filters_non_convolutional_node(result) if filter_dense else result


def obtain_all_critical_layers(
    output_node: EnrichedNetworkNode, input_resolution: int, filter_dense: bool = True
) -> List[EnrichedNetworkNode]:
    """Obtain all critical layers.
    A layer is defined as critical if it has a receptive field size LARGER
    than the input resolution. Critical layers have substantial
    probability of being unproductive.

    Args:
        output_node:        a node of the compute graph
        input_resolution:   the input resolution for which the critical
                            layers shall be computed
        filter_dense:       exclude all layers with infinite receptive field size
                            (essentially all layers that are
                            fully connected or successors of fully connected layers)
                            This is True by default.

    Returns:
        All layers predicted to be critical.
    """
    all_nodes = obtain_all_nodes(output_node)
    result = [node for node in all_nodes if node.receptive_field_min > input_resolution]
    return filters_non_convolutional_node(result) if filter_dense else result


def filters_non_convolutional_node(
    nodes: List[EnrichedNetworkNode],
) -> List[EnrichedNetworkNode]:
    """Filter all components that are not part of the feature extractor.

    Args:
        nodes: the list of noodes that shall be filtered.

    Returns:
        A list of all layers that are part of the feature extractor.
        This is decided by the kernel size, which is non-infinite
        for layers that are part of the feature extractor.
        Please note that layers like Dropout, BatchNormalization,
        which are agnostic towards the input shape,
        are treated like a convolutional layer with a kernel
        and stride size of 1.
    """
    return [node for node in nodes if node.layer_info.kernel_size != np.inf]
