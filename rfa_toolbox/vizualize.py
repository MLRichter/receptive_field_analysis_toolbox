import graphviz

from rfa_toolbox.graphs import EnrichedNetworkNode


def node_id(node: EnrichedNetworkNode) -> str:
    return f"{node.name}-{id(node)}"


def visualize_node(
    node: EnrichedNetworkNode,
    dot: graphviz.Digraph,
    input_res: int,
    color_border: bool,
    color_critical: bool,
):
    color = "white"
    if node.is_border(input_resolution=input_res):
        color = "red"
    elif node.receptive_field_min > input_res and color_critical:
        color = "orange"
    l_name = node.layer_info.name
    rf_info = "\\n" + f"r={node.receptive_field_min}"
    filters = f"\\n{node.layer_info.filters} filters"
    units = f"\\n{node.layer_info.units} units"

    label = l_name
    if node.layer_info.filters is not None:
        label += filters
    elif node.layer_info.units is not None:
        label += units
    label += rf_info

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
) -> graphviz.Digraph:
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
        )
    return f


if __name__ == "__main__":
    from rfa_toolbox.architectures.resnet import resnet18

    model = resnet18()
    dot = visualize_architecture(model, "resnet18", 32)
    dot.view()
