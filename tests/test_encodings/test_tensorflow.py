from json import loads

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16

from rfa_toolbox.encodings.tensorflow_keras.ingest_architecture import (
    create_graph_from_model,
)
from rfa_toolbox.graphs import KNOWN_FILTER_MAPPING, EnrichedNetworkNode


class TestKerasEncoding:
    def test_resnet50(self):
        model = ResNet50(weights=None)
        graph = create_graph_from_model(model)
        assert isinstance(graph, EnrichedNetworkNode)
        assert len(graph.all_layers) == len(loads(model.to_json())["config"]["layers"])

    def test_vgg16(self):
        model = VGG16(weights=None)
        graph = create_graph_from_model(model)
        assert isinstance(graph, EnrichedNetworkNode)
        assert len(graph.all_layers) == len(loads(model.to_json())["config"]["layers"])

    def test_inceptionv3(self):
        model = InceptionV3(weights=None)
        graph = create_graph_from_model(model)
        assert isinstance(graph, EnrichedNetworkNode)
        assert len(graph.all_layers) == len(loads(model.to_json())["config"]["layers"])

    def test_inceptionv3_with_infinite_rf_filter(self):
        model = InceptionV3(weights=None)
        graph = create_graph_from_model(model, filter_rf="inf")
        for node in graph.all_layers:
            assert node.receptive_field_info_filter == KNOWN_FILTER_MAPPING["inf"]
        graph = create_graph_from_model(model, filter_rf=None)
        for node in graph.all_layers:
            assert node.receptive_field_info_filter == KNOWN_FILTER_MAPPING[None]

    def test_inceptionv3_with_custom_rf_filter(self):
        model = InceptionV3(weights=None)

        def func(x):
            return x

        graph = create_graph_from_model(model, filter_rf=func)
        for node in graph.all_layers:
            assert node.receptive_field_info_filter == func
