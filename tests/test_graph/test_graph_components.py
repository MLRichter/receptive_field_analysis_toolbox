import json

from rfa_toolbox.network_components import ModelGraph, NetworkNode, \
    EnrichedNetworkNode, LayerDefinition


class TestLayerDefinition:

    def test_can_initialize_from_dict(self):
        layer = LayerDefinition.from_dict(**{
            "name": "test",
            "kernel_size": 3,
            "stride_size": 1}
        )
        assert layer.name == "test"
        assert layer.kernel_size == 3
        assert layer.stride_size == 1

    def test_to_dict(self):
        layer = LayerDefinition("test", 3, 1)
        assert layer.to_dict() == {
            "name": "test",
            "kernel_size": 3,
            "stride_size": 1
        }

    def test_to_dict_can_be_jsonified(self):
        layer = LayerDefinition("test", 3, 1)
        jsonified = json.dumps(layer.to_dict())
        assert isinstance(jsonified, str)

    def test_to_dict_can_be_jsonified_and_reconstituted(self):
        layer = LayerDefinition("test", 3, 1)
        jsonified = json.dumps(layer.to_dict())
        reconstituted = LayerDefinition.from_dict(**json.loads(jsonified))
        assert layer == reconstituted


class TestNetworkNode:
    ...


class TestEnrichedNetworkNode:
    ...


class TestModelGraph:
    ...
