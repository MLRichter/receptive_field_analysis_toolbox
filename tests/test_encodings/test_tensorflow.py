from json import loads

from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16

from rfa_toolbox.encodings.tensorflow_keras.ingest_architecture import (
    create_graph_from_model,
)
from rfa_toolbox.graphs import EnrichedNetworkNode


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
