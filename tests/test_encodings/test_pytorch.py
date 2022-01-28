import pytest
import torch
from torchvision.models.alexnet import alexnet
from torchvision.models.inception import inception_v3
from torchvision.models.mnasnet import mnasnet1_3
from torchvision.models.resnet import resnet18, resnet152
from torchvision.models.vgg import vgg19

from rfa_toolbox.encodings.pytorch.ingest_architecture import make_graph
from rfa_toolbox.encodings.pytorch.intermediate_graph import Digraph
from rfa_toolbox.graphs import EnrichedNetworkNode, LayerDefinition


class TestIntermediateGraph:
    def test_check_for_lone_node(self):
        nodes = {
            "node": EnrichedNetworkNode(
                name="node", layer_info=LayerDefinition(name="conv"), predecessors=[]
            )
        }
        dot = Digraph(None)
        with pytest.warns(UserWarning):
            dot._check_for_lone_node(nodes)


class TestOnPreimplementedModels:
    def test_make_graph_mnasnet1_3(self):
        model = mnasnet1_3
        m = model()
        tm = torch.jit.trace(m, [torch.randn(1, 3, 399, 399)])
        d = make_graph(tm, ref_mod=m)
        output_node = d.to_graph()
        assert len(output_node.all_layers) == 152
        assert isinstance(output_node, EnrichedNetworkNode)

    def test_make_graph_alexnet(self):
        model = alexnet
        m = model()
        tm = torch.jit.trace(m, [torch.randn(1, 3, 399, 399)])
        d = make_graph(tm, ref_mod=m)
        output_node = d.to_graph()
        assert len(output_node.all_layers) == 22
        assert isinstance(output_node, EnrichedNetworkNode)

    def test_make_graph_resnet18(self):
        model = resnet18
        m = model()
        tm = torch.jit.trace(m, [torch.randn(1, 3, 399, 399)])
        d = make_graph(tm, ref_mod=m)
        output_node = d.to_graph()
        # nice
        assert len(output_node.all_layers) == 69
        assert isinstance(output_node, EnrichedNetworkNode)

    def test_make_graph_resnet152(self):
        model = resnet152
        m = model()
        tm = torch.jit.trace(m, [torch.randn(1, 3, 399, 399)])
        d = make_graph(tm, ref_mod=m)
        output_node = d.to_graph()
        assert len(output_node.all_layers) == 515
        assert isinstance(output_node, EnrichedNetworkNode)

    def test_inceptionv3(self):
        model = inception_v3
        m = model()
        tm = torch.jit.trace(m, [torch.randn(1, 3, 399, 399)])
        d = make_graph(tm, ref_mod=m)
        output_node = d.to_graph()
        assert len(output_node.all_layers) == 319
        assert isinstance(output_node, EnrichedNetworkNode)

    def test_make_graph_vgg19(self):
        model = vgg19
        m = model()
        tm = torch.jit.trace(m, [torch.randn(1, 3, 399, 399)])
        d = make_graph(tm, ref_mod=m)
        output_node = d.to_graph()
        assert len(output_node.all_layers) == 46
        assert isinstance(output_node, EnrichedNetworkNode)
