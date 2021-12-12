import json

from rfa_toolbox.network_components import (
    EnrichedNetworkNode,
    LayerDefinition,
    NetworkNode,
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


class TestModelGraphStaticMethods:
    def test_obtain_all_nodes_from_root_on_single_node_graph(self):
        ...

    def test_obtain_all_nodes_from_a_sequential_graph(self):
        ...

    def test_obtain_all_nodes_from_a_nonsequential_graph(self):
        ...

    def test_find_input_node_with_input_node(self):
        ...

    def test_find_input_node_without_input_node_in_graph(self):
        ...

    def test_obtain_paths_from_sequential_path(self):
        ...

    def test_obtain_paths_from_multipath_architecture(self):
        ...

    def test_obtain_paths_from_subgraph(self):
        ...

    def test_obtain_paths_from_subgraph_with_input_node(self):
        ...

    def test_compute_receptive_field_for_node_with_kernel_greater_than_1(self):
        ...

    def test_compute_receptive_field_for_node_with_kernel_equal_to_1(self):
        ...

    def test_compute_receptive_field_for_node_with_infinite_kernel_size(self):
        ...

    def test_compute_receptive_field_for_large_stride_sizes(self):
        ...

    def test_compute_receptive_field_for_illegal_kernel_sizes(self):
        ...

    def test_compute_receptive_field_for_illegal_stride_sizes(self):
        ...

    def test_compute_receptive_field_for_illegal_growth_rate(self):
        ...

    def test_compute_receptive_field_for_node_sequence_in_a_sequential_architecture(
        self,
    ):
        ...

    def test_compute_receptive_field_for_node_sequence_in_a_non_sequential_architecture(
        self,
    ):
        ...

    def test_enrich_graph_on_sequential_graph(self):
        ...

    def test_enrich_graph_on_non_sequential_graph(self):
        ...

    def test_enrich_graph_on_enriched_graph_can_be_isomorph(self):
        ...

    def test_enrich_graph_on_oneiched_graph_not_isomorph_for_set_receptive_field_values(
        self,
    ):
        ...


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
