from json import loads
from typing import Any, Dict, List

from tensorflow.keras.models import Model

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
    """
    Finds the first node in the list of working_layers, which is not yet processed.

    Args:
        working_layers:     all unprocessed layers
        processed_nodes:    all processed nodes, the dicts maps node-ids
                            to their EnrichedNetworkNode-instances

    Returns:
        The first node in the list of working_layers, which is not yet processed.
    """
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
    """Obtain the layer-definition from a node-dict.
    The transformations of nodes into their respective layers
    is done by handler-objects, which are registered in this module in
    the variable PARSERS.
    """
    for parser in PARSERS:
        if parser.can_handle(node=node_dict):
            return parser(node_dict)
    raise ValueError(f"Could not find a parser for processing node: {node_dict}")


def create_node_from_dict(
    node_dict: Dict[str, Any], processed_nodes: Dict[str, EnrichedNetworkNode]
) -> EnrichedNetworkNode:
    """Create the node-representation of a layer.
    Args:
        node_dict:          the node in dictionary representation, as extracted
                            from the keras-model
        processed_nodes:    a dictionary, which maps already processed nodes
                            to their EnrichedNetworkNode-instances, used for
                            obtaining predecessors
    """
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
    """Create a graph of the model from a list of layers"""
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
    """Turn a dictionary extracted from the json-representation of a Keras
    model into the rfa-toolbox specific graph representation.

    Args:
     model_dict: the json-representation of the model

    Returns:
         a node of the graph
    """
    if "config" not in model_dict:
        raise AttributeError("Model-json export has no config")
    layer_config = model_dict["config"]
    if "layers" not in layer_config:
        raise AttributeError("Model-json export has no layers")
    layers = layer_config["layers"]
    graph: EnrichedNetworkNode = create_graph(layers)
    return graph


def keras_model_to_dict(model: Model) -> Dict[str, Any]:
    """Creates a model into a dictionary based on it's json-representation"""
    return loads(model.to_json())


def create_graph_from_model(model: Model) -> EnrichedNetworkNode:
    """Create a graph model from tensorflow
    Args:
        model: the model, thus must be a Keras-model.
    """
    model_dict = keras_model_to_dict(model)
    return model_dict_to_enriched_graph(model_dict)
