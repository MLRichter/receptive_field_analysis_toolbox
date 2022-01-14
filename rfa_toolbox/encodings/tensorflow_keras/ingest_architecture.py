from json import loads
from typing import Any, Dict, List

from tensorflow.keras.models import Model

from rfa_toolbox import visualize_architecture
from rfa_toolbox.encodings.tensorflow_keras.layer_handlers import (
    AnyHandler,
    DenseHandler,
    InputHandler,
    KernelBasedHandler,
    PoolingBasedHandler,
)
from rfa_toolbox.graphs import EnrichedNetworkNode, LayerDefinition

PARSERS = [
    InputHandler(),
    KernelBasedHandler(),
    PoolingBasedHandler(),
    DenseHandler(),
    AnyHandler(),
]


def find_processable_node(
    working_layers: List[Dict[str, Any]],
    processed_nodes: Dict[str, EnrichedNetworkNode],
) -> Dict[str, Any]:
    for layer in working_layers:
        if "inbound_nodes" in layer:
            inbound_nodes = layer["inbound_nodes"]
            if len(inbound_nodes) == 0:
                return layer
            # if all inbound nose are already processed, we can process this node
            inbound_node_processed = [
                inbound_node[0] in processed_nodes for inbound_node in inbound_nodes[0]
            ]
            if all(inbound_node_processed):
                return layer
    raise ValueError("Could not find a processable node")


def obtain_layer_definition(node_dict: Dict[str, Any]) -> LayerDefinition:
    for parser in PARSERS:
        if parser.can_handle(node=node_dict):
            return parser(node_dict)
    raise ValueError(f"Could not find a parser for processing node: {node_dict}")


def create_node_from_dict(
    node_dict: Dict[str, Any], processed_nodes: Dict[str, EnrichedNetworkNode]
) -> EnrichedNetworkNode:
    predecessors = (
        []
        if not node_dict["inbound_nodes"]
        else [
            processed_nodes[inbound_node[0]]
            for inbound_node in node_dict["inbound_nodes"][0]
        ]
    )
    layer_info: LayerDefinition = obtain_layer_definition(node_dict)
    return EnrichedNetworkNode(
        name=node_dict["name"], layer_info=layer_info, predecessors=predecessors
    )


def create_graph(layers: List[Dict[str, Any]]) -> EnrichedNetworkNode:
    processed_nodes: Dict[str, EnrichedNetworkNode] = {}
    working_layers: List[Dict[str, Any]] = layers[:]
    while working_layers:
        processable_node_dict: Dict[str, Any] = find_processable_node(
            working_layers, processed_nodes
        )
        node = create_node_from_dict(
            node_dict=processable_node_dict, processed_nodes=processed_nodes
        )
        processed_nodes[node.name] = node
        working_layers.remove(processable_node_dict)
        if len(working_layers) == 0:
            return node
    raise ValueError(f"Some nodes were left unprocessed: {working_layers}")


def model_dict_to_enriched_graph(model_dict: Dict[str, Any]) -> EnrichedNetworkNode:
    if "config" not in model_dict:
        raise AttributeError("Model-json export has no config")
    layer_config = model_dict["config"]
    if "layers" not in layer_config:
        raise AttributeError("Model-json export has no layers")
    layers = layer_config["layers"]
    graph: EnrichedNetworkNode = create_graph(layers)
    return graph


def keras_model_to_dict(model: Model) -> Dict[str, Any]:
    return loads(model.to_json())


def create_graph_from_model(model: Model) -> EnrichedNetworkNode:
    model_dict = keras_model_to_dict(model)
    return model_dict_to_enriched_graph(model_dict)


if __name__ == "__main__":
    from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2

    graph: EnrichedNetworkNode = create_graph_from_model(
        InceptionResNetV2(weights=None)
    )
    visualize_architecture(graph, "InceptionResNetV2", input_res=32).view()
