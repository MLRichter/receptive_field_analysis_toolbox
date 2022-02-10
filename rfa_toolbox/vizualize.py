from typing import Sequence, Union

import graphviz
import numpy as np

from rfa_toolbox.graphs import EnrichedNetworkNode


def node_id(node: EnrichedNetworkNode) -> str:
    """Provide a unique string for each node based on its name and object id.
    This makes the node-id human readable while also easy to process since it contains
    human interpretable elements while also being unique.

    Args:
        node: the EnrichedNetworkNode-instance the unique id shall be obtained

    Returns:
        A unique node id as a string of the following format ${node.name}-${id(node}
    """
    return f"{node.name}-{id(node)}"


def _feature_map_size_label(feature_map_size: Union[int, Sequence[int]]) -> str:
    if not isinstance(feature_map_size, Sequence) and not isinstance(
        feature_map_size, np.ndarray
    ):
        return (
            f"\\nFeature Map Res.: {max(feature_map_size, 1)} "
            f"x {max(feature_map_size, 1)}"
        )
    else:
        fm = np.asarray(feature_map_size)
        fm[fm < 1] = 1
        return "\\nFeature Map Res.: " f"{' x '.join(fm.astype(int).astype(str))}"


def visualize_node(
    node: EnrichedNetworkNode,
    dot: graphviz.Digraph,
    input_res: int,
    color_border: bool,
    color_critical: bool,
    include_rf_info: bool = True,
    filter_kernel_size_1: bool = False,
    include_fm_info: bool = True,
) -> None:
    """Create a node in a graphviz-graph based on an EnrichedNetworkNode instance.
    Also creates all edges that lead from predecessor nodes to this node.

    Args:
        node:               The node in question
        dot:                The graphviz-graph
        input_res:          The input resolution of the model - required for
                            coloring critical and border layers
        color_border:       The color used for marking border layer
        color_critical:     The color used for marking critical layers
        include_rf_info:    If True the receptive field information is
                            included in the node description

    Returns:
        Nothing.

    """
    color = "white"
    if (
        node.is_border(
            input_resolution=input_res, filter_kernel_size_1=filter_kernel_size_1
        )
        and color_border
    ):
        color = "red"
    elif (
        np.all(np.asarray(node.receptive_field_min) > np.asarray(input_res))
        and color_critical
        and not node.is_border(
            input_resolution=input_res, filter_kernel_size_1=filter_kernel_size_1
        )
    ):
        color = "orange"
    elif (
        np.any(np.asarray(node.receptive_field_min) > np.asarray(input_res))
        and color_critical
        and not node.is_border(
            input_resolution=input_res, filter_kernel_size_1=filter_kernel_size_1
        )
    ):
        color = "yellow"
    l_name = node.layer_info.name
    rf_info = (
        "\\n" + f"r(min)={node.receptive_field_min}, r(max)={node.receptive_field_max}"
    )

    filters = f"\\n{node.layer_info.filters} filters"
    units = f"\\n{node.layer_info.units} units"

    feature_map_size = (
        _feature_map_size_label(
            np.asarray(input_res) // np.asarray(node.get_maximum_scale_factor())
        )
        if node.kernel_size != np.inf
        else ""
    )

    label = l_name
    if node.layer_info.filters is not None:
        label += filters
    elif node.layer_info.units is not None:
        label += units
    if include_rf_info:
        label += rf_info
    if include_fm_info:
        label += feature_map_size

    dot.node(
        f"{node.name}-{id(node)}",
        label=label,
        fillcolor=color,
        style="filled",
    )
    for pred in node.predecessors:
        dot.edge(node_id(pred), node_id(node), label="")


def visualize_architecture(
    output_node: EnrichedNetworkNode,
    model_name: str,
    input_res: int = 224,
    color_critical: bool = True,
    color_border: bool = True,
    include_rf_info: bool = True,
    filter_kernel_size_1: bool = False,
    include_fm_info: bool = True,
) -> graphviz.Digraph:
    """Visualize an architecture using graphviz
    and mark critical and border layers in the graph visualization.

    Args:
        output_node:    an EnrichedNetworkNode-instance that belong to the
                        network graph to visualize. This function can handle
                        architectures with arbitrary many output
                        and one input node.
        model_name:     the name of the model
        input_res:      the input resolution (used for determining
                        critical and border layers)
        color_critical: if True the critical layers are colored orange, True by default.
        color_border:   if True the border layers are colored red, True by default.
        include_rf_info: if True the receptive field information is included in the node
                        description

    Returns:
        A graphviz.Digraph object that can visualize the network architecture.

    """
    f = graphviz.Digraph(model_name, filename=".gv")
    f.attr(rankdir="TB")
    f.attr("node", shape="rectangle")
    all_nodes = output_node.all_layers
    for node in all_nodes:
        visualize_node(
            node,
            dot=f,
            input_res=input_res,
            color_border=color_border,
            color_critical=color_critical,
            include_rf_info=include_rf_info,
            filter_kernel_size_1=filter_kernel_size_1,
            include_fm_info=include_fm_info,
        )
    return f
