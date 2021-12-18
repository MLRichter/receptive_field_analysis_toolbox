import json

import numpy as np
import pytest

from rfa_toolbox.network_components import (
    EnrichedNetworkNode,
    InputLayer,
    LayerDefinition,
    ModelGraph,
    NetworkNode,
    OutputLayer,
)


class TestLayerDefinition:
    def test_can_initialize_from_dict(self):
        layer = LayerDefinition.from_dict(
            **{"name": "test", "kernel_size": 3, "stride_size": 1}
        )
        assert layer.name == "test"
        assert layer.kernel_size == 3
        assert layer.stride_size == 1

    def test_to_dict(self):
        layer = LayerDefinition("test", 3, 1)
        assert layer.to_dict() == {"name": "test", "kernel_size": 3, "stride_size": 1}

    def test_to_dict_can_be_jsonified(self):
        layer = LayerDefinition("test", 3, 1)
        jsonified = json.dumps(layer.to_dict())
        assert isinstance(jsonified, str)

    def test_to_dict_can_be_jsonified_and_reconstituted(self):
        layer = LayerDefinition("test", 3, 1)
        jsonified = json.dumps(layer.to_dict())
        reconstituted = LayerDefinition.from_dict(**json.loads(jsonified))
        assert layer == reconstituted

    def test_cannot_set_illegal_stride_size_values(self):
        for i in range(10):
            with pytest.raises(ValueError):
                LayerDefinition("test", i, -i)

    def test_cannot_set_illegal_kernel_size_values(self):
        for i in range(10):
            with pytest.raises(ValueError):
                LayerDefinition("test", -i, i)

    def test_cannot_set_illegal_kernel_and_stride_size_values(self):
        for i in range(10):
            strides = np.random.randint(-100, 0)
            kernel = np.random.randint(-100, 0)

            with pytest.raises(ValueError):
                LayerDefinition("test", kernel, strides)


class TestNetworkNode:
    def test_can_initialize_from_dict_without_nodes_and_id(self):
        node = NetworkNode.from_dict(
            **{
                "name": "test",
                "layer_type": {"name": "test1", "kernel_size": 3, "stride_size": 1},
                "predecessor_list": [],
            }
        )
        assert node.name == "test"
        assert node.layer_type.name == "test1"
        assert node.layer_type.kernel_size == 3
        assert node.layer_type.stride_size == 1
        assert node.predecessor_list == []

    def test_can_initialize_from_dict_without_nodes(self):
        node = NetworkNode.from_dict(
            **{
                "id": 123,
                "name": "test",
                "layer_type": {"name": "test1", "kernel_size": 3, "stride_size": 1},
                "predecessor_list": [],
            }
        )
        assert node.name == "test"
        assert node.layer_type.name == "test1"
        assert node.layer_type.kernel_size == 3
        assert node.layer_type.stride_size == 1
        assert node.predecessor_list == []

    def test_can_initialize_from_dict(self):
        node1 = NetworkNode.from_dict(
            **{
                "id": 124,
                "name": "B",
                "layer_type": {"name": "test1", "kernel_size": 3, "stride_size": 1},
                "predecessor_list": [],
            }
        )

        node2 = NetworkNode.from_dict(
            **{
                "id": 125,
                "name": "C",
                "layer_type": {"name": "test1", "kernel_size": 3, "stride_size": 1},
                "predecessor_list": [],
            }
        )

        node = NetworkNode.from_dict(
            **{
                "id": 123,
                "name": "A",
                "layer_type": {"name": "test1", "kernel_size": 3, "stride_size": 1},
                "predecessor_list": [node1, node2],
            }
        )

        assert node.name == "A"
        assert node.layer_type.name == "test1"
        assert node.layer_type.kernel_size == 3
        assert node.layer_type.stride_size == 1
        assert node.predecessor_list == [node1, node2]

    def test_to_dict_without_predecessor(self):
        node = NetworkNode("test", LayerDefinition("test1", 3, 1))
        assert node.to_dict() == {
            "id": id(node),
            "name": "test",
            "layer_type": {"name": "test1", "kernel_size": 3, "stride_size": 1},
            "predecessor_list": [],
        }

    def test_to_dict_with_predecessor(self):
        node1 = NetworkNode("A", LayerDefinition("test1", 3, 1))
        node2 = NetworkNode("B", LayerDefinition("test2", 3, 1), [node1])

        assert node2.to_dict() == {
            "id": id(node2),
            "name": "B",
            "layer_type": {"name": "test2", "kernel_size": 3, "stride_size": 1},
            "predecessor_list": [id(node1)],
        }

    def test_to_dict_can_be_jsonified(self):
        node = NetworkNode("test", LayerDefinition("test1", 3, 1))
        jsonified = json.dumps(node.to_dict())
        assert isinstance(jsonified, str)


class TestEnrichedNetworkNode:
    def test_to_dict_without_predecessor(self):
        node = EnrichedNetworkNode(
            name="test",
            layer_type=LayerDefinition("test1", 3, 1),
            receptive_field_sizes=[8, 16, 32],
            predecessor_list=[],
        )

        assert node.to_dict() == {
            "id": id(node),
            "name": "test",
            "layer_type": {"name": "test1", "kernel_size": 3, "stride_size": 1},
            "receptive_field_sizes": [8, 16, 32],
            "predecessor_list": [],
        }

    def test_can_reconstructor_from_to_dict_without_predecessor(self):
        node = EnrichedNetworkNode.from_dict(
            **{
                "id": 123,
                "name": "test",
                "receptive_field_sizes": [8, 16, 32],
                "layer_type": {"name": "test1", "kernel_size": 3, "stride_size": 1},
                "predecessor_list": [],
            }
        )
        assert node.name == "test"
        assert node.layer_type.name == "test1"
        assert node.layer_type.kernel_size == 3
        assert node.layer_type.stride_size == 1
        assert node.receptive_field_max == 32
        assert node.receptive_field_min == 8
        assert node.receptive_field_sizes == [8, 16, 32]
        assert node.predecessor_list == []

    def test_to_dict_with_predecessor(self):
        node0 = EnrichedNetworkNode(
            name="B",
            layer_type=LayerDefinition("test0", 1, 2),
            receptive_field_sizes=[4, 8, 16],
            predecessor_list=[],
        )

        node1 = EnrichedNetworkNode(
            name="A",
            layer_type=LayerDefinition("test1", 3, 1),
            receptive_field_sizes=[8, 16, 32],
            predecessor_list=[node0],
        )

        assert node1.to_dict() == {
            "id": id(node1),
            "name": "A",
            "layer_type": {"name": "test1", "kernel_size": 3, "stride_size": 1},
            "receptive_field_sizes": [8, 16, 32],
            "predecessor_list": [id(node0)],
        }

    def test_assign_successors_works_as_exspected(self):
        node0 = EnrichedNetworkNode(
            name="A",
            layer_type=LayerDefinition("test0", 1, 2),
            receptive_field_sizes=[4, 8, 16],
            predecessor_list=[],
        )
        node1 = EnrichedNetworkNode(
            name="B",
            layer_type=LayerDefinition("test1", 3, 1),
            receptive_field_sizes=[8, 16, 32],
            predecessor_list=[node0],
        )
        node2 = EnrichedNetworkNode(
            name="C",
            layer_type=LayerDefinition("test2", 3, 1),
            receptive_field_sizes=[10, 20, 34],
            predecessor_list=[node0, node1],
        )

        assert len(node0.succecessor_list) == 2
        assert len(node1.succecessor_list) == 1
        assert len(node2.succecessor_list) == 0
        assert node0.succecessor_list[0] == node1
        assert node0.succecessor_list[1] == node2
        assert node1.succecessor_list[0] == node2

    def test_compute_border_layer_at_sequence(self):
        node0 = EnrichedNetworkNode(
            name="A",
            layer_type=LayerDefinition("test0", 1, 2),
            receptive_field_sizes=[4, 8, 16],
            predecessor_list=[],
        )
        node1 = EnrichedNetworkNode(
            name="B",
            layer_type=LayerDefinition("test1", 3, 1),
            receptive_field_sizes=[8, 16, 32],
            predecessor_list=[node0],
        )
        node2 = EnrichedNetworkNode(
            name="C",
            layer_type=LayerDefinition("test2", 3, 1),
            receptive_field_sizes=[10, 20, 34],
            predecessor_list=[node1],
        )
        node3 = EnrichedNetworkNode(
            name="D",
            layer_type=LayerDefinition("test3", 3, 1),
            receptive_field_sizes=[12, 24, 36],
            predecessor_list=[node2],
        )
        node4 = EnrichedNetworkNode(
            name="E",
            layer_type=LayerDefinition("test4", 3, 1),
            receptive_field_sizes=[14, 28, 38],
            predecessor_list=[node2, node3],
        )

        assert not node1.is_border(8)
        assert node2.is_border(8)
        assert node3.is_border(8)
        assert node4.is_border(8)

    def test_border_layer_at_fork_is_based_on_minimum_receptive_field(self):
        node0 = EnrichedNetworkNode(
            name="A",
            layer_type=LayerDefinition("test0", 1, 2),
            receptive_field_sizes=[4, 8, 16],
            predecessor_list=[],
        )
        node1 = EnrichedNetworkNode(
            name="B",
            layer_type=LayerDefinition("test1", 3, 1),
            receptive_field_sizes=[8, 16, 32],
            predecessor_list=[node0],
        )
        node2 = EnrichedNetworkNode(
            name="C",
            layer_type=LayerDefinition("test2", 3, 1),
            receptive_field_sizes=[10, 20, 34],
            predecessor_list=[node1],
        )
        node3 = EnrichedNetworkNode(
            name="D",
            layer_type=LayerDefinition("test3", 3, 1),
            receptive_field_sizes=[12, 24, 36],
            predecessor_list=[node2],
        )
        node4 = EnrichedNetworkNode(
            name="E",
            layer_type=LayerDefinition("test4", 3, 1),
            receptive_field_sizes=[14, 28, 38],
            predecessor_list=[node2, node3],
        )

        assert not node1.is_border(11)
        assert not node2.is_border(11)
        assert not node3.is_border(11)
        assert not node4.is_border(11)
        assert node4.is_border(10)


@pytest.fixture
def simple_non_enrichted_network_nodes():
    node0 = NetworkNode(name="A", layer_type=InputLayer(), predecessor_list=[])
    node1 = NetworkNode(
        name="B",
        layer_type=LayerDefinition("test1", 3, 1),
        predecessor_list=[node0],
    )
    node2 = NetworkNode(
        name="C",
        layer_type=LayerDefinition("test2", 3, 1),
        predecessor_list=[node1],
    )
    node3 = NetworkNode(
        name="D",
        layer_type=LayerDefinition("test3", 3, 1),
        predecessor_list=[node2],
    )
    node4 = NetworkNode(
        name="E",
        layer_type=LayerDefinition("test4", 3, 1),
        predecessor_list=[node3],
    )
    node5 = NetworkNode(
        name="F",
        layer_type=OutputLayer(),
        predecessor_list=[node4],
    )
    return node0, node1, node2, node3, node4, node5


@pytest.fixture
def simple_network_nodes():
    node0 = EnrichedNetworkNode(
        name="A",
        layer_type=InputLayer(),
        predecessor_list=[],
        receptive_field_sizes=[1],
    )
    node1 = EnrichedNetworkNode(
        name="B",
        layer_type=LayerDefinition("test1", 3, 1),
        receptive_field_sizes=[3],
        predecessor_list=[node0],
    )
    node2 = EnrichedNetworkNode(
        name="C",
        layer_type=LayerDefinition("test2", 3, 1),
        receptive_field_sizes=[5],
        predecessor_list=[node1],
    )
    node3 = EnrichedNetworkNode(
        name="D",
        layer_type=LayerDefinition("test3", 3, 1),
        receptive_field_sizes=[7],
        predecessor_list=[node2],
    )
    node4 = EnrichedNetworkNode(
        name="E",
        layer_type=LayerDefinition("test4", 3, 1),
        receptive_field_sizes=[9],
        predecessor_list=[node3],
    )
    node5 = EnrichedNetworkNode(
        name="F",
        layer_type=OutputLayer(),
        receptive_field_sizes=[9],
        predecessor_list=[node4],
    )
    return node0, node1, node2, node3, node4, node5


@pytest.fixture
def simple_network_nodes_with_strides():
    node0 = EnrichedNetworkNode(
        name="A",
        layer_type=InputLayer(),
        predecessor_list=[],
        receptive_field_sizes=[1],
    )
    node1 = EnrichedNetworkNode(
        name="B",
        layer_type=LayerDefinition("test1", 3, 2),
        receptive_field_sizes=[3],
        predecessor_list=[node0],
    )
    node2 = EnrichedNetworkNode(
        name="C",
        layer_type=LayerDefinition("test2", 3, 2),
        receptive_field_sizes=[7],
        predecessor_list=[node1],
    )
    node3 = EnrichedNetworkNode(
        name="D",
        layer_type=LayerDefinition("test3", 3, 1),
        receptive_field_sizes=[15],
        predecessor_list=[node2],
    )
    node4 = EnrichedNetworkNode(
        name="E",
        layer_type=LayerDefinition("test4", 3, 2),
        receptive_field_sizes=[23],
        predecessor_list=[node3],
    )
    node5 = EnrichedNetworkNode(
        name="F",
        layer_type=OutputLayer(),
        receptive_field_sizes=[23],
        predecessor_list=[node4],
    )
    return node0, node1, node2, node3, node4, node5


@pytest.fixture
def simple_network_nodes_with_kernel_size1():
    node0 = EnrichedNetworkNode(
        name="A",
        layer_type=InputLayer(),
        predecessor_list=[],
        receptive_field_sizes=[1],
    )
    node1 = EnrichedNetworkNode(
        name="B",
        layer_type=LayerDefinition("test1", 3, 1),
        receptive_field_sizes=[3],
        predecessor_list=[node0],
    )
    node2 = EnrichedNetworkNode(
        name="C",
        layer_type=LayerDefinition("test2", 1, 1),
        receptive_field_sizes=[3],
        predecessor_list=[node1],
    )
    node3 = EnrichedNetworkNode(
        name="D",
        layer_type=LayerDefinition("test3", 3, 1),
        receptive_field_sizes=[5],
        predecessor_list=[node2],
    )
    node4 = EnrichedNetworkNode(
        name="E",
        layer_type=LayerDefinition("test4", 1, 1),
        receptive_field_sizes=[5],
        predecessor_list=[node3],
    )
    node5 = EnrichedNetworkNode(
        name="F",
        layer_type=OutputLayer(),
        receptive_field_sizes=[5],
        predecessor_list=[node4],
    )
    return node0, node1, node2, node3, node4, node5


@pytest.fixture
def simple_network_nodes_with_dense_layer():
    node0 = EnrichedNetworkNode(
        name="A",
        layer_type=InputLayer(),
        predecessor_list=[],
        receptive_field_sizes=[1],
    )
    node1 = EnrichedNetworkNode(
        name="B",
        layer_type=LayerDefinition("test1", 3, 1),
        receptive_field_sizes=[3],
        predecessor_list=[node0],
    )
    node2 = EnrichedNetworkNode(
        name="C",
        layer_type=LayerDefinition("test2", 1, 1),
        receptive_field_sizes=[3],
        predecessor_list=[node1],
    )
    node3 = EnrichedNetworkNode(
        name="D",
        layer_type=LayerDefinition("test3", np.inf, 1),
        receptive_field_sizes=[np.inf],
        predecessor_list=[node2],
    )
    node4 = EnrichedNetworkNode(
        name="E",
        layer_type=LayerDefinition("test4", np.inf, 1),
        receptive_field_sizes=[np.inf],
        predecessor_list=[node3],
    )
    node5 = EnrichedNetworkNode(
        name="F",
        layer_type=OutputLayer(),
        receptive_field_sizes=[np.inf],
        predecessor_list=[node4],
    )
    return node0, node1, node2, node3, node4, node5


@pytest.fixture
def simple_non_sequential_network_nodes():
    node0 = EnrichedNetworkNode(
        name="A",
        layer_type=InputLayer(),
        predecessor_list=[],
        receptive_field_sizes=[1],
    )
    node1 = EnrichedNetworkNode(
        name="B",
        layer_type=LayerDefinition("test1", 3, 1),
        receptive_field_sizes=[3],
        predecessor_list=[node0],
    )
    node2 = EnrichedNetworkNode(
        name="C",
        layer_type=LayerDefinition("test2", 3, 1),
        receptive_field_sizes=[5],
        predecessor_list=[node1],
    )
    node3 = EnrichedNetworkNode(
        name="D",
        layer_type=LayerDefinition("test3", 3, 1),
        receptive_field_sizes=[3],
        predecessor_list=[node0],
    )
    node4 = EnrichedNetworkNode(
        name="E",
        layer_type=LayerDefinition("test4", 3, 1),
        receptive_field_sizes=[5],
        predecessor_list=[node3],
    )
    node5 = EnrichedNetworkNode(
        name="F",
        layer_type=OutputLayer(),
        receptive_field_sizes=[5],
        predecessor_list=[node2, node4],
    )
    return node0, node1, node2, node3, node4, node5


@pytest.fixture
def simple_non_sequential_network_non_enriched_nodes():
    node0 = NetworkNode(
        name="A",
        layer_type=InputLayer(),
        predecessor_list=[],
    )
    node1 = NetworkNode(
        name="B",
        layer_type=LayerDefinition("test1", 3, 1),
        predecessor_list=[node0],
    )
    node2 = NetworkNode(
        name="C",
        layer_type=LayerDefinition("test2", 3, 1),
        predecessor_list=[node1],
    )
    node3 = NetworkNode(
        name="D",
        layer_type=LayerDefinition("test3", 3, 1),
        predecessor_list=[node0],
    )
    node4 = NetworkNode(
        name="E",
        layer_type=LayerDefinition("test4", 5, 1),
        predecessor_list=[node3],
    )
    node5 = NetworkNode(
        name="F",
        layer_type=OutputLayer(),
        predecessor_list=[node2, node4],
    )
    return node0, node1, node2, node3, node4, node5


class TestModelGraphStaticMethods:
    def test_obtain_all_nodes_from_root_on_single_node_graph(
        self, simple_network_nodes
    ):
        node0, node1, node2, node3, node4, node5 = simple_network_nodes
        all_nodes = ModelGraph.obtain_all_nodes_from_root(node0)
        assert node0 in all_nodes
        assert node1 not in all_nodes
        assert node2 not in all_nodes
        assert node3 not in all_nodes
        assert node4 not in all_nodes
        assert node5 not in all_nodes

    def test_obtain_all_nodes_from_a_sequential_graph(self, simple_network_nodes):
        node0, node1, node2, node3, node4, node5 = simple_network_nodes
        all_nodes = ModelGraph.obtain_all_nodes_from_root(node5)
        assert node0 in all_nodes
        assert node1 in all_nodes
        assert node2 in all_nodes
        assert node3 in all_nodes
        assert node4 in all_nodes
        assert node5 in all_nodes

    def test_obtain_all_nodes_from_a_nonsequential_graph(
        self, simple_non_sequential_network_nodes
    ):
        node0, node1, node2, node3, node4, node5 = simple_non_sequential_network_nodes
        all_nodes = ModelGraph.obtain_all_nodes_from_root(node5)
        assert node0 in all_nodes
        assert node1 in all_nodes
        assert node2 in all_nodes
        assert node3 in all_nodes
        assert node4 in all_nodes
        assert node5 in all_nodes
        assert len(all_nodes) == 6

    def test_find_input_node_with_input_node(self, simple_non_sequential_network_nodes):
        input_node, _, _, _, _, _ = simple_non_sequential_network_nodes
        assert (
            ModelGraph._find_input_node(list(simple_non_sequential_network_nodes))
            == input_node
        )

    def test_find_input_node_without_input_node_in_graph(
        self, simple_non_sequential_network_nodes
    ):
        node0, node1, node2, node3, node4, node5 = simple_non_sequential_network_nodes
        with pytest.raises(ValueError):
            ModelGraph._find_input_node([node5, node4, node3, node2, node1])

    def test_obtain_paths_from_sequential_path(self, simple_network_nodes):
        node0, node1, node2, node3, node4, node5 = simple_network_nodes
        paths = ModelGraph.obtain_paths(node0, node5)
        assert len(paths) == 1
        assert paths[0] == [node0, node1, node2, node3, node4, node5]

    def test_obtain_paths_from_multipath_architecture(
        self, simple_non_sequential_network_nodes
    ):
        node0, node1, node2, node3, node4, node5 = simple_non_sequential_network_nodes
        paths = ModelGraph.obtain_paths(node0, node5)
        assert len(paths) == 2
        assert paths[0] == [node0, node1, node2, node5]
        assert paths[1] == [node0, node3, node4, node5]

    def test_obtain_paths_from_subgraph(self, simple_non_sequential_network_nodes):
        node0, node1, node2, node3, node4, node5 = simple_non_sequential_network_nodes
        paths = ModelGraph.obtain_paths(node1, node5)
        assert len(paths) == 1
        assert paths[0] == [node1, node2, node5]

    def test_obtain_paths_from_subgraph_with_input_node(
        self, simple_non_sequential_network_nodes
    ):
        node0, node1, node2, node3, node4, node5 = simple_non_sequential_network_nodes
        paths = ModelGraph.obtain_paths(node0, node4)
        assert len(paths) == 1
        assert paths[0] == [node0, node3, node4]

    def test_compute_receptive_field_for_node_with_kernel_greater_than_1(
        self, simple_network_nodes
    ):
        node0, node1, node2, node3, node4, node5 = simple_network_nodes
        r_l = []
        for node in simple_network_nodes:
            r_l.append(node.receptive_field_sizes[0])
            node.receptive_field_sizes.remove(r_l[-1])
        for node in simple_network_nodes:
            assert len(node.receptive_field_sizes) == 0
        all_nodes = ModelGraph.obtain_all_nodes_from_root(node5)
        for node in all_nodes:
            assert len(node.receptive_field_sizes) == 0
        graph = ModelGraph.compute_receptive_field_for_node_sequence(all_nodes)
        assert len(graph) == len(r_l)
        for layer, receptive_field in zip(graph, r_l):
            print(layer, receptive_field)
            assert layer.receptive_field_max == receptive_field
            assert layer.receptive_field_min == receptive_field
            assert len(layer.receptive_field_sizes) == 1

    def test_compute_receptive_field_for_node_with_kernel_equal_to_1(
        self, simple_network_nodes_with_kernel_size1
    ):
        (
            node0,
            node1,
            node2,
            node3,
            node4,
            node5,
        ) = simple_network_nodes_with_kernel_size1
        r_l = []
        for node in simple_network_nodes_with_kernel_size1:
            r_l.append(node.receptive_field_sizes[0])
            node.receptive_field_sizes.remove(r_l[-1])
        for node in simple_network_nodes_with_kernel_size1:
            assert len(node.receptive_field_sizes) == 0
        all_nodes = ModelGraph.obtain_all_nodes_from_root(node5)
        for node in all_nodes:
            assert len(node.receptive_field_sizes) == 0
        graph = ModelGraph.compute_receptive_field_for_node_sequence(all_nodes)
        assert len(graph) == len(r_l)
        for layer, receptive_field in zip(graph, r_l):
            print(layer, receptive_field)
            assert layer.receptive_field_max == receptive_field
            assert layer.receptive_field_min == receptive_field
            assert len(layer.receptive_field_sizes) == 1

    def test_compute_receptive_field_for_node_with_infinite_kernel_size(
        self, simple_network_nodes_with_dense_layer
    ):
        node0, node1, node2, node3, node4, node5 = simple_network_nodes_with_dense_layer
        r_l = []
        for node in simple_network_nodes_with_dense_layer:
            r_l.append(node.receptive_field_sizes[0])
            node.receptive_field_sizes.remove(r_l[-1])
        for node in simple_network_nodes_with_dense_layer:
            assert len(node.receptive_field_sizes) == 0
        all_nodes = ModelGraph.obtain_all_nodes_from_root(node5)
        for node in all_nodes:
            assert len(node.receptive_field_sizes) == 0
        graph = ModelGraph.compute_receptive_field_for_node_sequence(all_nodes)
        assert len(graph) == len(r_l)
        for layer, receptive_field in zip(graph, r_l):
            print(layer, receptive_field)
            assert layer.receptive_field_max == receptive_field
            assert layer.receptive_field_min == receptive_field
            assert len(layer.receptive_field_sizes) == 1

    def test_compute_receptive_field_for_large_stride_sizes(
        self, simple_network_nodes_with_strides
    ):
        node0, node1, node2, node3, node4, node5 = simple_network_nodes_with_strides
        r_l = []
        for node in simple_network_nodes_with_strides:
            r_l.append(node.receptive_field_sizes[0])
            node.receptive_field_sizes.remove(r_l[-1])
        for node in simple_network_nodes_with_strides:
            assert len(node.receptive_field_sizes) == 0
        all_nodes = ModelGraph.obtain_all_nodes_from_root(node5)
        for node in all_nodes:
            assert len(node.receptive_field_sizes) == 0
        graph = ModelGraph.compute_receptive_field_for_node_sequence(all_nodes)
        assert len(graph) == len(r_l)
        for layer, receptive_field in zip(graph, r_l):
            print(layer, receptive_field)
            assert layer.receptive_field_max == receptive_field
            assert layer.receptive_field_min == receptive_field
            assert len(layer.receptive_field_sizes) == 1

    def test_compute_receptive_field_for_node_sequence_in_a_non_sequential_architecture(
        self, simple_non_sequential_network_nodes
    ):
        node0, node1, node2, node3, node4, node5 = simple_non_sequential_network_nodes
        r_l = []
        for node in simple_non_sequential_network_nodes:
            r_l.append(node.receptive_field_sizes[0])
            node.receptive_field_sizes.remove(r_l[-1])
        for node in simple_non_sequential_network_nodes:
            assert len(node.receptive_field_sizes) == 0
        all_nodes = ModelGraph.obtain_all_nodes_from_root(node5)
        for node in all_nodes:
            assert len(node.receptive_field_sizes) == 0
        paths = ModelGraph.obtain_paths(node0, node5)
        for path in paths:
            ModelGraph.compute_receptive_field_for_node_sequence(path)
        all_nodes = ModelGraph.obtain_all_nodes_from_root(node5)
        assert len(all_nodes) == len(r_l)
        for layer, receptive_field in zip(all_nodes, r_l):
            print(layer, receptive_field)
            assert layer.receptive_field_max == receptive_field
            assert layer.receptive_field_min == receptive_field
            assert len(layer.receptive_field_sizes) == 1

    def test_enrich_graph_on_sequential_graph(self, simple_non_enrichted_network_nodes):
        node0, node1, node2, node3, node4, node5 = simple_non_enrichted_network_nodes
        root_node = ModelGraph.enrich_graph(node5)
        all_nodes = ModelGraph.obtain_all_nodes_from_root(root_node)
        assert len(all_nodes) == 6
        for o, l in zip(simple_non_enrichted_network_nodes, all_nodes):
            assert isinstance(l, EnrichedNetworkNode)
            assert o.layer_type.stride_size == l.layer_type.stride_size
            assert o.layer_type.kernel_size == l.layer_type.kernel_size

    def test_enrich_graph_on_sequential_graph_correct_predecessor(
        self, simple_non_enrichted_network_nodes
    ):
        node0, node1, node2, node3, node4, node5 = simple_non_enrichted_network_nodes
        root_node = ModelGraph.enrich_graph(node5)
        all_nodes = ModelGraph.obtain_all_nodes_from_root(root_node)
        assert len(all_nodes) == 6
        for o, l in zip(simple_non_enrichted_network_nodes, all_nodes):
            if l.predecessor_list:
                for predecessor in l.predecessor_list:
                    assert predecessor.is_in(all_nodes)
                    assert isinstance(predecessor, EnrichedNetworkNode)
                assert len(l.predecessor_list) == len(o.predecessor_list)

    def test_enrich_graph_on_sequential_graph_correct_successor(
        self, simple_non_enrichted_network_nodes
    ):
        node0, node1, node2, node3, node4, node5 = simple_non_enrichted_network_nodes
        root_node = ModelGraph.enrich_graph(node5)
        all_nodes = ModelGraph.obtain_all_nodes_from_root(root_node)
        assert len(all_nodes) == 6
        for o, l in zip(simple_non_enrichted_network_nodes, all_nodes):
            if l.predecessor_list:
                for predecessor in l.predecessor_list:
                    assert l.is_in(predecessor.succecessor_list)
                for successor in l.succecessor_list:
                    assert successor.is_in(all_nodes)
                    assert l.is_in(successor.predecessor_list)
                assert len(l.predecessor_list) == len(o.predecessor_list)

    def test_enrich_graph_on_non_sequential_graph(
        self, simple_non_sequential_network_non_enriched_nodes
    ):
        (
            node0,
            node1,
            node2,
            node3,
            node4,
            node5,
        ) = simple_non_sequential_network_non_enriched_nodes
        root_node = ModelGraph.enrich_graph(node5)
        all_nodes = ModelGraph.obtain_all_nodes_from_root(root_node)
        assert len(all_nodes) == 6
        for o, l in zip(simple_non_sequential_network_non_enriched_nodes, all_nodes):
            if l.predecessor_list:
                for predecessor in l.predecessor_list:
                    assert l.is_in(predecessor.succecessor_list)
                for successor in l.succecessor_list:
                    assert successor.is_in(all_nodes)
                    assert l.is_in(successor.predecessor_list)
                assert len(l.predecessor_list) == len(o.predecessor_list)

    def test_enrich_graph_on_enriched_graph_can_be_isomorph(
        self, simple_non_sequential_network_non_enriched_nodes
    ):
        (
            node0,
            node1,
            node2,
            node3,
            node4,
            node5,
        ) = simple_non_sequential_network_non_enriched_nodes
        root_node = ModelGraph.enrich_graph(node5)
        root_node2 = ModelGraph.enrich_graph(node5)
        all_nodes = ModelGraph.obtain_all_nodes_from_root(root_node)
        all_nodes2 = ModelGraph.obtain_all_nodes_from_root(root_node2)
        assert len(all_nodes) == len(all_nodes2)
        for nodeA, nodeB in zip(all_nodes, all_nodes2):
            assert id(nodeA) != id(nodeB)
            assert nodeA == nodeB

    def test_enrich_graph_on_enriched_graph_not_isomorph_for_set_receptive_field_values(
        self, simple_non_sequential_network_nodes
    ):
        node0, node1, node2, node3, node4, node5 = simple_non_sequential_network_nodes
        for node in simple_non_sequential_network_nodes:
            node.receptive_field_sizes.append(666)
        root_node = ModelGraph.enrich_graph(node5)
        all_nodes = ModelGraph.obtain_all_nodes_from_root(root_node)
        all_nodes2 = simple_non_sequential_network_nodes
        assert len(all_nodes) == len(all_nodes2)


class TestModelGraph:
    def test_to_dict(self):
        ...

    def test_to_dict_is_json_serializable(self):
        ...

    def test_from_dict(self):
        ...

    def test_graph_can_be_reconstructed_from_json_string_of_itself(self):
        ...

    def test_find_unproductive_layer_when_there_are_no_unproductive_layers(self):
        ...

    def test_find_unproductive_layer_when_there_is_an_unproductive_layer(self):
        ...

    def test_find_unproductive_layer_when_there_are_multiple_unproductive_layers(self):
        ...

    def test_find_unproductive_layer_in_a_non_sequential_architecture(
        self,
    ):
        ...

    def test_find_border_layers_when_there_are_no_border_layers(self):
        ...

    def test_find_border_layers_when_there_is_a_border_layer(self):
        ...

    def test_find_border_layers_when_there_are_multiple_border_layers(self):
        ...


class TestReceptiveFieldsOfImportantArchitectures:
    def test_compute_receptive_field_size_for_resnet18_cifar10_optimized(self):
        ...

    def test_compute_receptive_field_size_for_resnet34_cifar10_optimized(self):
        ...

    def test_compute_receptive_field_size_for_resnet18(self):
        ...

    def test_compute_receptive_field_size_for_resnet34(self):
        ...

    def test_compute_receptive_field_size_for_resnet50(self):
        ...

    def test_compute_receptive_field_size_for_vgg11(self):
        ...

    def test_compute_receptive_field_size_for_vgg13(self):
        ...

    def test_compute_receptive_field_size_for_vgg16(self):
        ...

    def test_compute_receptive_field_size_for_vgg19(self):
        ...
