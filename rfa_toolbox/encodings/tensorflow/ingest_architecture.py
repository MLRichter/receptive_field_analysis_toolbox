from json import loads
from typing import Any, Dict

from tensorflow.keras.models import Model

from rfa_toolbox.graphs import EnrichedNetworkNode


def model_dict_to_enriched_graph(model_dict: Dict[str, Any]) -> EnrichedNetworkNode:
    ...


def keras_model_to_dict(model: Model) -> Dict[str, Any]:
    return loads(model.to_json())


def create_graph_from_model(model: Model) -> EnrichedNetworkNode:
    model_dict = keras_model_to_dict(model)
    return model_dict_to_enriched_graph(model_dict)
