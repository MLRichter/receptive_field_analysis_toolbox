import numpy as np
import pytest
import torchvision

from rfa_toolbox import create_graph_from_pytorch_model
from rfa_toolbox.architectures.resnet import resnet18, resnet101
from rfa_toolbox.architectures.vgg import vgg16
from rfa_toolbox.graphs import EnrichedNetworkNode, LayerDefinition
from rfa_toolbox.utils.graph_utils import (
    input_resolution_range,
    obtain_all_critical_layers,
    obtain_all_nodes,
    obtain_border_layers,
)


@pytest.fixture()
def single_node():
    node0 = EnrichedNetworkNode(
        name="Layer0",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
        predecessors=[],
    )
    return node0


@pytest.fixture()
def sequential_network_non_square():
    node0 = EnrichedNetworkNode(
        name="Layer0",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=(3, 5), stride_size=1),
        predecessors=[],
    )
    node1 = EnrichedNetworkNode(
        name="Layer1",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=(3, 5), stride_size=1),
        predecessors=[node0],
    )
    node2 = EnrichedNetworkNode(
        name="Layer2",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=(3, 5), stride_size=1),
        predecessors=[node1],
    )
    node3 = EnrichedNetworkNode(
        name="Layer3",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=(3, 5), stride_size=1),
        predecessors=[node2],
    )
    node4 = EnrichedNetworkNode(
        name="Layer4",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=(3, 5), stride_size=1),
        predecessors=[node3],
    )
    node5 = EnrichedNetworkNode(
        name="Layer5",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=(3, 5), stride_size=1),
        predecessors=[node4],
    )
    node6 = EnrichedNetworkNode(
        name="Layer6",
        layer_info=LayerDefinition(name="Softmax", kernel_size=1, stride_size=1),
        predecessors=[node5],
    )
    return node6


@pytest.fixture()
def sequential_network():
    node0 = EnrichedNetworkNode(
        name="Layer0",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
        predecessors=[],
    )
    node1 = EnrichedNetworkNode(
        name="Layer1",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
        predecessors=[node0],
    )
    node2 = EnrichedNetworkNode(
        name="Layer2",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
        predecessors=[node1],
    )
    node3 = EnrichedNetworkNode(
        name="Layer3",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
        predecessors=[node2],
    )
    node4 = EnrichedNetworkNode(
        name="Layer4",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
        predecessors=[node3],
    )
    node5 = EnrichedNetworkNode(
        name="Layer5",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
        predecessors=[node4],
    )
    node6 = EnrichedNetworkNode(
        name="Layer6",
        layer_info=LayerDefinition(name="Softmax", kernel_size=1, stride_size=1),
        predecessors=[node5],
    )
    return node6


@pytest.fixture()
def nonsequential_network():
    node0 = EnrichedNetworkNode(
        name="Layer0",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
        predecessors=[],
    )
    node1 = EnrichedNetworkNode(
        name="Layer1",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
        predecessors=[node0],
    )
    node2 = EnrichedNetworkNode(
        name="Layer2",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
        predecessors=[node1],
    )
    node3 = EnrichedNetworkNode(
        name="Layer3",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
        predecessors=[node2],
    )
    node4 = EnrichedNetworkNode(
        name="Layer4",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
        predecessors=[node2],
    )
    node5 = EnrichedNetworkNode(
        name="Layer5",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
        predecessors=[node4, node3],
    )
    node6 = EnrichedNetworkNode(
        name="Layer6",
        layer_info=LayerDefinition(name="Softmax", kernel_size=1, stride_size=1),
        predecessors=[node5],
    )
    return node6


@pytest.fixture()
def nonsequential_network2():
    node0 = EnrichedNetworkNode(
        name="Layer0",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
        predecessors=[],
    )
    node1 = EnrichedNetworkNode(
        name="Layer1",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
        predecessors=[node0],
    )
    node2 = EnrichedNetworkNode(
        name="Layer2",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
        predecessors=[node1],
    )
    node3 = EnrichedNetworkNode(
        name="Layer3",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=(7, 12), stride_size=1),
        predecessors=[node2],
    )
    node4 = EnrichedNetworkNode(
        name="Layer4",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=(21, 4), stride_size=1),
        predecessors=[node2],
    )
    node5 = EnrichedNetworkNode(
        name="Layer5",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
        predecessors=[node4, node3],
    )
    node6 = EnrichedNetworkNode(
        name="Layer6",
        layer_info=LayerDefinition(name="Softmax", kernel_size=1, stride_size=1),
        predecessors=[node5],
    )
    return node6


class TestObtainAllNodes:
    def test_obtain_node_in_single_graph_network(self, single_node):
        node = obtain_all_nodes(single_node)
        assert node == [single_node]

    def test_obtain_node_from_non_output_node(self, sequential_network):
        nodes = obtain_all_nodes(sequential_network.predecessors[0].predecessors[0])
        assert len(nodes) == 7
        for i, node in enumerate(nodes):
            assert node.name == f"Layer{i}"

    def test_for_sequential_architecture(self, sequential_network):
        nodes = obtain_all_nodes(sequential_network)
        assert len(nodes) == 7
        for i, node in enumerate(nodes):
            assert node.name == f"Layer{i}"

    def test_for_non_sequential_architecture(self, nonsequential_network):
        nodes = obtain_all_nodes(nonsequential_network)
        assert len(nodes) == 7
        names = {f"Layer{6 - i}" for i in range(7)}
        actual_names = {node.name for node in nodes}
        assert names == actual_names


@pytest.fixture()
def vgg16_model():
    return vgg16()


@pytest.fixture()
def resnet18_model():
    return resnet18()


@pytest.fixture()
def resnet101_model():
    return resnet101()


class TestObtainAllCriticalLayers:
    def test_obtain_all_ciritcal_layers(self, vgg16_model):
        ciritical_layers = obtain_all_critical_layers(vgg16_model, 32, False)
        all_layers = obtain_all_nodes(vgg16_model)
        print(ciritical_layers)
        assert len(ciritical_layers) == 13
        for layer in all_layers:
            if layer.is_in(ciritical_layers):
                assert layer.receptive_field_min > 32
            else:
                assert layer.receptive_field_min <= 32

    def test_obtaining_critical_layers_with_filter_of_dense_layers(self, vgg16_model):
        ciritical_layers = obtain_all_critical_layers(vgg16_model, 32, True)
        all_layers = obtain_all_nodes(vgg16_model)
        print(ciritical_layers)
        assert len(ciritical_layers) == 10
        for layer in all_layers:
            if layer.is_in(ciritical_layers) and layer.kernel_size != np.inf:
                assert layer.receptive_field_min > 32
            else:
                assert (
                    layer.receptive_field_min <= 32
                    or layer.receptive_field_min == np.inf
                )

    def test_obtaining_ciritcal_layers_for_multipath_architecturs(self, resnet18_model):
        ciritical_layers = obtain_all_critical_layers(resnet18_model, 32, True)
        all_layers = obtain_all_nodes(resnet18_model)
        print(ciritical_layers)
        assert len(ciritical_layers) == 9
        for layer in all_layers:
            if layer.is_in(ciritical_layers) and layer.kernel_size != np.inf:
                assert layer.receptive_field_min > 32
            else:
                assert (
                    layer.receptive_field_min <= 32
                    or layer.receptive_field_min == np.inf
                )


class TestObtainBorderLayers:
    def test_obtaining_border_layer_if_there_is_none(self, vgg16_model):
        input_res = 42000
        borders = obtain_border_layers(
            vgg16_model, input_resolution=input_res, filter_dense=True
        )
        assert len(borders) == 0
        for border in borders:
            assert border.receptive_field_min >= input_res
            for pred in border.predecessors:
                assert pred.receptive_field_min >= input_res

    def test_obtaining_border_layer_without_filter(self, vgg16_model):
        input_res = 42000
        borders = obtain_border_layers(
            vgg16_model, input_resolution=input_res, filter_dense=False
        )
        assert len(borders) == 2
        for border in borders:
            assert border.receptive_field_min >= input_res
            for pred in border.predecessors:
                assert pred.receptive_field_min >= input_res

    def test_obtaining_border_layer_on_in_a_sequential_architecture(self, vgg16_model):
        input_res = 32
        borders = obtain_border_layers(
            vgg16_model, input_resolution=input_res, filter_dense=True
        )
        assert len(borders) == 10
        for border in borders:
            assert border.receptive_field_min >= input_res
            for pred in border.predecessors:
                assert pred.receptive_field_min >= input_res

    def test_obtaining_border_layer_if_there_are_many(self, resnet18_model):
        input_res = 32
        borders = obtain_border_layers(
            resnet18_model, input_resolution=input_res, filter_dense=True
        )
        assert len(borders) == 3
        for border in borders:
            assert border.receptive_field_min >= input_res
            for pred in border.predecessors:
                assert pred.receptive_field_min >= input_res

    def test_obtaining_border_layer_in_very_large_architecture(self, resnet101_model):
        input_res = 32
        borders = obtain_border_layers(
            resnet101_model, input_resolution=input_res, filter_dense=True
        )
        assert len(borders) == 26
        for border in borders:
            assert border.receptive_field_min >= input_res
            for pred in border.predecessors:
                assert pred.receptive_field_min >= input_res


class TestFindInputResolutionRange:
    def test_with_higher_degree_tensor(self, sequential_network):
        for i in range(10):
            cardinality = np.random.randint(1, 1000)
            r_min, r_max = input_resolution_range(
                sequential_network, cardinality=cardinality
            )
            assert len(r_max) == cardinality
            assert len(r_min) == cardinality

    def test_with_scalar_receptive_field_sizes_lower_bound(self, sequential_network):
        r_min, r_max = input_resolution_range(sequential_network, lower_bound=True)
        assert len(r_max) == 2
        assert len(r_min) == 2
        assert r_min == (11, 11)
        assert r_max == (13, 13)

    def test_with_scalar_receptive_field_sizes(self, sequential_network):
        r_min, r_max = input_resolution_range(sequential_network)
        assert len(r_max) == 2
        assert len(r_min) == 2
        assert r_min == (13, 13)
        assert r_max == (13, 13)

    def test_with_non_sequential(self, nonsequential_network2):
        r_min, r_max = input_resolution_range(nonsequential_network2)
        assert len(r_max) == 2
        assert len(r_min) == 2
        assert r_min == (27, 18)
        assert r_max == (29, 20)

    def test_with_non_square_receptive_field_sizes(self, sequential_network_non_square):
        r_min, r_max = input_resolution_range(sequential_network_non_square)
        assert len(r_max) == 2
        assert len(r_min) == 2
        assert r_min == (13, 25)
        assert r_max == (13, 25)

    def test_with_non_square_receptive_field_sizes_without_se(
        self, sequential_network_non_square
    ):
        model = torchvision.models.resnet50()
        graph = create_graph_from_pytorch_model(model)
        min_res, max_res = input_resolution_range(graph)  # (75, 75), (427, 427)
        assert len(min_res) == 2
        assert len(max_res) == 2
        assert min_res == (75, 75)
        assert max_res == (427, 427)

    def test_with_non_square_receptive_field_sizes_wit_se(
        self, sequential_network_non_square
    ):
        model = torchvision.models.efficientnet_b0()
        graph = create_graph_from_pytorch_model(model)
        min_res, max_res = input_resolution_range(
            graph, filter_all_inf_rf=True
        )  # (75, 75), (427, 427)
        assert len(min_res) == 2
        assert len(max_res) == 2
        assert min_res == (299, 299)
        assert max_res == (851, 851)
