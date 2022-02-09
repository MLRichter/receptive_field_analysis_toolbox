from typing import List, Sequence, Tuple, Union

import numpy as np

from rfa_toolbox.graphs import EnrichedNetworkNode


def obtain_all_nodes(output_node: EnrichedNetworkNode) -> List[EnrichedNetworkNode]:
    """Fetch all nodes from a single node of the compute graph.

    Args:
        output_node:            output node of the graph

    Returns:
        A List containing all EnrichedNetworkNodes.

    """
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
    result = []
    for node in nodes:
        if isinstance(node.receptive_field_min, Sequence) or isinstance(
            node.receptive_field_min, tuple
        ):
            if not np.any(np.isinf(node.receptive_field_min)):
                result.append(node)
        elif node.receptive_field_min != np.inf:
            result.append(node)
    return result


def filters_non_infinite_rf_sizes(
    rf_sizes: List[Union[Tuple[int, ...], int]],
) -> List[Union[Tuple[int, ...], int]]:
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
    result = []
    for rf_size in rf_sizes:
        if isinstance(rf_size, Sequence) or isinstance(rf_size, tuple):
            if not np.any(np.isinf(rf_size)):
                result.append(rf_size)
        elif rf_size != np.inf:
            result.append(rf_size)
    return result


def input_resolution_range(
    graph: EnrichedNetworkNode,
    filter_all_inf_rf: bool = True,
    filter_kernel_size_1: bool = False,
    cardinality: int = 2,
    lower_bound: bool = False,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Obtain the smallest and largest feasible input resolution.
    The smallest feasible input resolution is defined as the input smallest input
    resolution with no unproductive convolutional layers.
    The largest feasible input resolution is defined as the input
    resolution with at least one convolutional layer with a maximum
    receptive field large enough to grasp the entire image.
    These can be considered upper and lower bound for potential input resolutions.
    Everything smaller than the provided input resolution will result
    in unproductive layers, any resolution larger than the large feasible
    input resolution will result in potential patterns being undetectable due to
    a to small receptive field size.

    Args:
        graph:              The neural network
        filter_all_inf_rf:  filters ALL infinite receptive field sizes before
                            computing the result, this may come in handy
                            if you want to ignore the influence
                            of attention mechanisms like SE-modules,
                            which technically adds a global context to the image
                            (increasing the maximum receptive field
                            size to infinity in the process).
                            However this can be somewhat
                            misleading, since these types of modules are not realy
                            build to extract features from the image.
                            This functionality is disabled by default.
        cardinality:        The tensor shape, which is 2D by default.
        lower_bound:        Disabled by default. If disabled, returns the lowest
                            resolution which utilizes the entire
                            network receptive field expansion.
                            If enabled it returns the lowest resolution exspected
                            to yield no unproductive, weighted layers.

    Returns:
        Smallest and largest feasible input resolution.
    """
    all_nodes = obtain_all_nodes(graph)
    all_nodes = filters_non_convolutional_node(all_nodes)
    if filter_kernel_size_1:
        all_nodes = [node for node in all_nodes if node.kernel_size > 1]
    if not filter_all_inf_rf:
        rf_min = [xi.receptive_field_min for x in all_nodes for xi in x.predecessors]
        rf_max = [x.receptive_field_max for x in all_nodes]
    else:
        rf_min = [
            x._apply_function_on_receptive_field_sizes(
                lambda x: min(filters_non_infinite_rf_sizes(x), default=0)
            )
            for x in all_nodes
        ]
        rf_max = [
            x._apply_function_on_receptive_field_sizes(
                lambda x: max(filters_non_infinite_rf_sizes(x), default=0)
            )
            for x in all_nodes
        ]

    def find_max(
        rf: List[Union[Tuple[int, ...], int]],
        axis: int = 0,
        second_largest: bool = False,
    ) -> int:
        """Find the maximum value of a list of tuples or integers.

        Args:
            rf:    a list of tuples or integers
            axis:  the axis along which the maximum shall be found

        Returns:
            The maximum value of the list.
        """
        rf_no_tuples = {
            x[axis] if isinstance(x, Sequence) or isinstance(x, np.ndarray) else x
            for x in rf
        }
        if not second_largest:
            return max(rf_no_tuples)
        else:
            rf_no_tuples = set(rf_no_tuples)
            rf_no_tuples.remove(max(rf_no_tuples))
            return max(rf_no_tuples)

    r_max = tuple(find_max(rf_max, i) for i in range(cardinality))
    r_min = tuple(find_max(rf_min, i, lower_bound) for i in range(cardinality))
    return r_min, r_max
