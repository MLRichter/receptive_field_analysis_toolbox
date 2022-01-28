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


class SomeModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.k_size = 3
        self.s_size = 1
        self.conv1 = torch.nn.Conv2d(
            64, 64, kernel_size=self.k_size, stride=self.s_size, padding=1, bias=False
        )

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.max_pool2d(
            x, kernel_size=self.k_size * 2, stride=self.s_size * 2, padding=2
        )
        return x


class TestUtils:
    def test_adding_a_handler(self):
        m = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            SomeModule(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
        )

        from rfa_toolbox import create_graph_from_pytorch_model
        from rfa_toolbox.encodings.pytorch import add_custom_layer_handler

        def kernel_size(module):
            x1 = module.k_size
            k_size2 = module.k_size * 2
            x2 = k_size2 - 1
            return x1 + x2

        def stride_size(module):
            return getattr(module, "s_size", 1) * 2

        def name_handler(module, name):
            k_size = kernel_size(module)
            s_size = stride_size(module)
            return f"{name} {k_size}x{k_size} / {s_size}"

        add_custom_layer_handler(
            class_name="SomeModule",
            name_handler=name_handler,
            kernel_size_provider=kernel_size,
            stride_size_provider=stride_size,
        )
        gr = create_graph_from_pytorch_model(m, custom_layers=["SomeModule"])
        layer = [
            layer for layer in gr.all_layers if "SomeModule" in layer.layer_info.name
        ][0]
        assert layer.layer_info.kernel_size == 8
        assert layer.layer_info.stride_size == 2
        assert layer.layer_info.name == f"SomeModule {8}x{8} / {2}"
