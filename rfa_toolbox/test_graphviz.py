import pytest
from torchvision.models.efficientnet import efficientnet_b0

from rfa_toolbox import create_graph_from_pytorch_model
from rfa_toolbox.vizualize import visualize_architecture


class FakeGraph:
    def __init__(self, *args, **kwargs):
        pass

    def edge(self, *args, **kwargs):
        pass

    def node(self, *args, **kwargs):
        pass

    def attr(self, *args, **kwargs):
        pass


@pytest.fixture()
def inject_fake_digraph():
    import rfa_toolbox.vizualize as viz

    viz.graphviz.Digraph = FakeGraph


class TestVisualize:
    def test_visualize(self, inject_fake_digraph):
        model = efficientnet_b0(pretrained=False)
        graph = create_graph_from_pytorch_model(model)
        x = visualize_architecture(graph, "efficientnet")
        assert isinstance(x, FakeGraph)
