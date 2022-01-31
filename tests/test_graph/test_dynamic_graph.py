import json
from collections import Sequence

import numpy as np
import pytest

from rfa_toolbox.architectures.resnet import resnet18, resnet36, resnet50, resnet101
from rfa_toolbox.architectures.vgg import vgg11, vgg13, vgg16, vgg19
from rfa_toolbox.graphs import (
    KNOWN_FILTER_MAPPING,
    EnrichedNetworkNode,
    LayerDefinition,
    noop_filter,
)
from rfa_toolbox.utils.graph_utils import obtain_all_nodes


class TestLayerDefinition:
    def test_can_initialize_from_dict(self):
        layer = LayerDefinition.from_dict(
            {"name": "test", "kernel_size": 3, "stride_size": 1}
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
        reconstituted = LayerDefinition.from_dict(json.loads(jsonified))
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

    def test_set_tuple_kernel_size(self):
        layer = LayerDefinition("test", (3, 4), 1)
        assert layer.kernel_size == (3, 4)

    def test_set_tuple_stride_size(self):
        layer = LayerDefinition("test", 3, (4, 5))
        assert layer.stride_size == (4, 5)

    def test_set_tuple_kernel_and_stride_size(self):
        layer = LayerDefinition("test", (3, 4), (4, 5))
        assert layer.stride_size == (4, 5)
        assert layer.kernel_size == (3, 4)

    def test_tuples_must_be_the_same_size(self):
        for i in range(500):
            strides = np.random.randint(0, 100)
            kernel = np.random.randint(-0, 100)
            kernel_shape = np.random.randint(75, 100)
            stride_shape = np.random.randint(1, 50)

            kernel_tuple = [kernel for _ in range(kernel_shape)]
            stride_tuple = [strides for _ in range(stride_shape)]
            with pytest.raises(ValueError):
                LayerDefinition("test", kernel_tuple, stride_tuple)


class TestEnrichedNetworkNode:
    def test_simple_sequential(self):
        conv1 = EnrichedNetworkNode(
            name="Layer1",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
            predecessors=[],
        )

        conv2 = EnrichedNetworkNode(
            name="Layer2",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
            predecessors=[conv1],
        )

        conv3 = EnrichedNetworkNode(
            name="Layer3",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
            predecessors=[conv2],
        )

        conv4 = EnrichedNetworkNode(
            name="Layer4",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
            predecessors=[conv3],
        )

        conv5 = EnrichedNetworkNode(
            name="Layer5",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
            predecessors=[conv4],
        )

        assert conv1.receptive_field_min == conv1.receptive_field_max
        assert conv1.receptive_field_min == 3
        assert conv2.receptive_field_min == conv2.receptive_field_max
        assert conv2.receptive_field_min == 5
        assert conv3.receptive_field_min == conv3.receptive_field_max
        assert conv3.receptive_field_min == 7
        assert conv4.receptive_field_min == conv4.receptive_field_max
        assert conv4.receptive_field_min == 9
        assert conv5.receptive_field_min == conv5.receptive_field_max
        assert conv5.receptive_field_min == 11

    def test_simple_sequential_with_strides(self):
        conv1 = EnrichedNetworkNode(
            name="Layer1",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
            predecessors=[],
        )

        conv2 = EnrichedNetworkNode(
            name="Layer2",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=2),
            predecessors=[conv1],
        )

        conv3 = EnrichedNetworkNode(
            name="Layer3",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
            predecessors=[conv2],
        )

        conv4 = EnrichedNetworkNode(
            name="Layer4",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=2),
            predecessors=[conv3],
        )

        conv5 = EnrichedNetworkNode(
            name="Layer5",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
            predecessors=[conv4],
        )

        assert conv1.receptive_field_min == conv1.receptive_field_max
        assert conv1.receptive_field_min == 3
        assert conv2.receptive_field_min == conv2.receptive_field_max
        assert conv2.receptive_field_min == 5
        assert conv3.receptive_field_min == conv3.receptive_field_max
        assert conv3.receptive_field_min == 9
        assert conv4.receptive_field_min == conv4.receptive_field_max
        assert conv4.receptive_field_min == 13
        assert conv5.receptive_field_min == conv5.receptive_field_max
        assert conv5.receptive_field_min == 21

    def test_simple_sequential_with_strides_and_no_expansion(self):
        conv1 = EnrichedNetworkNode(
            name="Layer1",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=1, stride_size=1),
            predecessors=[],
        )

        conv2 = EnrichedNetworkNode(
            name="Layer2",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=1, stride_size=2),
            predecessors=[conv1],
        )

        conv3 = EnrichedNetworkNode(
            name="Layer3",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=1, stride_size=1),
            predecessors=[conv2],
        )

        conv4 = EnrichedNetworkNode(
            name="Layer4",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=2),
            predecessors=[conv3],
        )

        conv5 = EnrichedNetworkNode(
            name="Layer5",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=1, stride_size=1),
            predecessors=[conv4],
        )

        assert conv1.receptive_field_min == conv1.receptive_field_max
        assert conv1.receptive_field_min == 1
        assert conv2.receptive_field_min == conv2.receptive_field_max
        assert conv2.receptive_field_min == 1
        assert conv3.receptive_field_min == conv3.receptive_field_max
        assert conv3.receptive_field_min == 1
        assert conv4.receptive_field_min == conv4.receptive_field_max
        assert conv4.receptive_field_min == 5
        assert conv5.receptive_field_min == conv5.receptive_field_max
        assert conv5.receptive_field_min == 5

    def test_simple_non_sequential_with_strides_and_no_expansion(self):
        conv1 = EnrichedNetworkNode(
            name="Layer1",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
            predecessors=[],
        )

        conv2 = EnrichedNetworkNode(
            name="Layer2",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=2),
            predecessors=[conv1],
        )

        conv3 = EnrichedNetworkNode(
            name="Layer3",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=5, stride_size=2),
            predecessors=[conv1],
        )

        conv4 = EnrichedNetworkNode(
            name="Layer4",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
            predecessors=[conv2, conv3],
        )

        conv5 = EnrichedNetworkNode(
            name="Layer5",
            layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
            predecessors=[conv4],
        )

        print(conv4)

        assert conv1.receptive_field_min == conv1.receptive_field_max
        assert conv1.receptive_field_min == 3
        assert conv2.receptive_field_min == conv2.receptive_field_max
        assert conv2.receptive_field_min == 5
        assert conv3.receptive_field_min == conv3.receptive_field_max
        assert conv3.receptive_field_min == 7
        assert conv4.receptive_field_min != conv4.receptive_field_max
        assert conv4.receptive_field_min == 9
        assert conv4.receptive_field_max == 11
        assert conv5.receptive_field_min != conv5.receptive_field_max
        assert conv5.receptive_field_min == 13
        assert conv5.receptive_field_max == 15


@pytest.fixture()
def vgg11_model():
    return vgg11()


@pytest.fixture()
def vgg13_model():
    return vgg13()


@pytest.fixture()
def vgg16_model():
    return vgg16()


@pytest.fixture()
def vgg19_model():
    return vgg19()


@pytest.fixture()
def resnet18_model():
    return resnet18()


@pytest.fixture()
def resnet36_model():
    return resnet36()


@pytest.fixture()
def resnet50_model():
    return resnet50()


@pytest.fixture()
def resnet101_model():
    return resnet101()


@pytest.fixture()
def vgg11_model_rf():
    return [
        1,
        3,
        4,
        8,
        10,
        18,
        26,
        30,
        46,
        62,
        70,
        102,
        134,
        150,
        np.inf,
        np.inf,
        np.inf,
    ]


@pytest.fixture()
def vgg13_model_rf():
    return [
        1,
        3,
        5,
        6,
        10,
        14,
        16,
        24,
        32,
        36,
        52,
        68,
        76,
        108,
        140,
        156,
        np.inf,
        np.inf,
        np.inf,
    ]


@pytest.fixture()
def vgg16_model_rf():
    return [
        1,
        3,
        5,
        6,
        10,
        14,
        16,
        24,
        32,
        40,
        44,
        60,
        76,
        92,
        100,
        132,
        164,
        196,
        212,
        np.inf,
        np.inf,
        np.inf,
    ]


@pytest.fixture()
def vgg19_model_rf():
    return [
        1,
        3,
        5,
        6,
        10,
        14,
        16,
        24,
        32,
        40,
        48,
        52,
        68,
        84,
        100,
        116,
        124,
        156,
        188,
        220,
        252,
        268,
        np.inf,
        np.inf,
        np.inf,
    ]


class TestReceptiveFieldsOfImportantArchitectures:
    def test_compute_receptive_field_size_for_vgg11(self, vgg11_model, vgg11_model_rf):
        rf_nodes = obtain_all_nodes(vgg11_model)
        rf = [node.receptive_field_min for node in rf_nodes]
        assert rf == vgg11_model_rf

    def test_compute_receptive_field_size_for_vgg13(self, vgg13_model, vgg13_model_rf):
        rf_nodes = obtain_all_nodes(vgg13_model)
        rf = [node.receptive_field_min for node in rf_nodes]
        assert rf == vgg13_model_rf

    def test_compute_receptive_field_size_for_vgg16(self, vgg16_model, vgg16_model_rf):
        rf_nodes = obtain_all_nodes(vgg16_model)
        rf = [node.receptive_field_min for node in rf_nodes]
        assert rf == vgg16_model_rf

    def test_compute_receptive_field_size_for_vgg19(self, vgg19_model, vgg19_model_rf):
        rf_nodes = obtain_all_nodes(vgg19_model)
        rf = [node.receptive_field_min for node in rf_nodes]
        assert rf == vgg19_model_rf

    def test_compute_receptive_field_size_for_resnet18(self, resnet18_model):
        rf_nodes = obtain_all_nodes(resnet18_model)
        rf = {node.name: node.receptive_field_max for node in rf_nodes}
        print(rf_nodes[0])
        assert rf["Stage3-Block1-Addition"] == 435


class TestReceptiveFieldSizesForArchitecturesWithTupleKernel:
    def test_creating_graph_with_tuple_kernels(self):
        conv1 = EnrichedNetworkNode(
            name="Conv1",
            layer_info=LayerDefinition(
                name="Conv3x3", kernel_size=(3, 3), stride_size=1, filters=64
            ),
            predecessors=[],
        )
        conv2 = EnrichedNetworkNode(
            name="Conv2",
            layer_info=LayerDefinition(
                name="Conv3x3", kernel_size=(3, 3), stride_size=1, filters=128
            ),
            predecessors=[conv1],
        )

        conv3 = EnrichedNetworkNode(
            name="Conv3",
            layer_info=LayerDefinition(
                name="Conv3x3", kernel_size=(5, 5), stride_size=1, filters=256
            ),
            predecessors=[conv1],
        )

        conv4 = EnrichedNetworkNode(
            name="Conv4",
            layer_info=LayerDefinition(
                name="Conv3x3", kernel_size=(3, 3), stride_size=1, filters=256
            ),
            predecessors=[conv2, conv3],
        )

        out = EnrichedNetworkNode(
            name="Softmax",
            layer_info=LayerDefinition(name="Fully Connected", units=1000),
            predecessors=[conv4],
        )
        print(conv1)
        print(conv2)
        print(conv3)
        print(conv4)
        assert conv1.receptive_field_min == (3, 3)
        assert conv1.receptive_field_max == (3, 3)
        assert conv2.receptive_field_min == (5, 5)
        assert conv2.receptive_field_max == (5, 5)
        assert conv3.receptive_field_min == (7, 7)
        assert conv3.receptive_field_max == (7, 7)
        assert conv4.receptive_field_min == (7, 7)
        assert conv4.receptive_field_max == (9, 9)
        assert out.receptive_field_min == np.inf
        assert out.receptive_field_max == np.inf

    def test_creating_graph_with_tuple_kernels_and_strides(self):
        conv1 = EnrichedNetworkNode(
            name="Conv1",
            layer_info=LayerDefinition(
                name="Conv3x3", kernel_size=(3, 3), stride_size=(1, 2), filters=64
            ),
            predecessors=[],
        )
        conv2 = EnrichedNetworkNode(
            name="Conv2",
            layer_info=LayerDefinition(
                name="Conv3x3", kernel_size=(3, 3), stride_size=(1, 1), filters=128
            ),
            predecessors=[conv1],
        )

        conv3 = EnrichedNetworkNode(
            name="Conv3",
            layer_info=LayerDefinition(
                name="Conv3x3", kernel_size=(5, 5), stride_size=(1, 1), filters=256
            ),
            predecessors=[conv1],
        )

        conv4 = EnrichedNetworkNode(
            name="Conv4",
            layer_info=LayerDefinition(
                name="Conv3x3", kernel_size=(3, 3), stride_size=(1, 1), filters=256
            ),
            predecessors=[conv2, conv3],
        )

        out = EnrichedNetworkNode(
            name="Softmax",
            layer_info=LayerDefinition(name="Fully Connected", units=1000),
            predecessors=[conv4],
        )
        assert conv1.receptive_field_min == (3, 3)
        assert conv1.receptive_field_max == (3, 3)
        assert conv2.receptive_field_min == (5, 7)
        assert conv2.receptive_field_max == (5, 7)
        assert conv3.receptive_field_min == (7, 11)
        assert conv3.receptive_field_max == (7, 11)
        assert conv4.receptive_field_min == (7, 11)
        assert conv4.receptive_field_max == (9, 15)
        assert out.receptive_field_min == np.inf
        assert out.receptive_field_max == np.inf

    def test_find_border_layer_into_tuple_based_model(self):
        conv1 = EnrichedNetworkNode(
            name="Conv1",
            layer_info=LayerDefinition(
                name="Conv3x3", kernel_size=(3, 3), stride_size=(1, 2), filters=64
            ),
            predecessors=[],
        )
        conv2 = EnrichedNetworkNode(
            name="Conv2",
            layer_info=LayerDefinition(
                name="Conv3x3", kernel_size=(3, 3), stride_size=(1, 1), filters=128
            ),
            predecessors=[conv1],
        )

        conv3 = EnrichedNetworkNode(
            name="Conv3",
            layer_info=LayerDefinition(
                name="Conv3x3", kernel_size=(5, 5), stride_size=(1, 1), filters=256
            ),
            predecessors=[conv1],
        )

        conv4 = EnrichedNetworkNode(
            name="Conv4",
            layer_info=LayerDefinition(
                name="Conv3x3", kernel_size=(3, 3), stride_size=(1, 1), filters=256
            ),
            predecessors=[conv2, conv3],
        )

        _ = EnrichedNetworkNode(
            name="Softmax",
            layer_info=LayerDefinition(name="Fully Connected", units=1000),
            predecessors=[conv4],
        )

        assert not conv4.is_border(32)
        assert not conv4.is_border((32, 32))
        assert conv4.is_border(2)
        assert conv4.is_border((2, 2))


@pytest.fixture()
def list_of_predecessors_noop():
    l0 = EnrichedNetworkNode(
        name="Conv0",
        layer_info=LayerDefinition(
            name="Conv3x3",
            kernel_size=1,
            stride_size=1,
            filters=64,
        ),
        predecessors=[],
        receptive_field_info_filter=KNOWN_FILTER_MAPPING[None],
    )

    l1 = EnrichedNetworkNode(
        name="Conv1",
        layer_info=LayerDefinition(
            name="Conv3x3",
            kernel_size=(3, 3),
            stride_size=(1, 1),
            filters=64,
        ),
        predecessors=[l0],
        receptive_field_info_filter=noop_filter,
    )

    l2 = EnrichedNetworkNode(
        name="Conv2",
        layer_info=LayerDefinition(
            name="Conv3x3",
            kernel_size=3,
            stride_size=(1, 1),
            filters=64,
        ),
        predecessors=[l0],
        receptive_field_info_filter=noop_filter,
    )

    l3 = EnrichedNetworkNode(
        name="Conv3",
        layer_info=LayerDefinition(
            name="Conv3x3",
            kernel_size=3,
            stride_size=1,
            filters=64,
        ),
        predecessors=[l0],
        receptive_field_info_filter=noop_filter,
    )

    l4 = EnrichedNetworkNode(
        name="Conv4",
        layer_info=LayerDefinition(
            name="Conv3x3",
            kernel_size=(np.inf, 3),
            stride_size=(1, 1),
            filters=64,
        ),
        predecessors=[l0],
        receptive_field_info_filter=noop_filter,
    )

    l5 = EnrichedNetworkNode(
        name="Conv5",
        layer_info=LayerDefinition(
            name="Conv3x3",
            kernel_size=(np.inf, np.inf),
            stride_size=(1, 1),
            filters=64,
        ),
        predecessors=[l0],
        receptive_field_info_filter=noop_filter,
    )

    l6 = EnrichedNetworkNode(
        name="Conv6",
        layer_info=LayerDefinition(
            name="Conv3x3",
            kernel_size=np.inf,
            stride_size=(1, 1),
            filters=64,
        ),
        predecessors=[l0],
        receptive_field_info_filter=noop_filter,
    )

    return [l1, l2, l3, l4, l5, l6]


@pytest.fixture()
def model_with_scalar_noop_infinity_predecessors(list_of_predecessors_noop):
    return EnrichedNetworkNode(
        name="Conv7",
        layer_info=LayerDefinition(
            name="Conv3x3",
            kernel_size=(1, 1),
            stride_size=(1, 1),
            filters=64,
        ),
        predecessors=list_of_predecessors_noop,
        receptive_field_info_filter=noop_filter,
    )


@pytest.fixture()
def list_of_predecessors_inf_filter():
    l0 = EnrichedNetworkNode(
        name="Conv0",
        layer_info=LayerDefinition(
            name="Conv3x3",
            kernel_size=1,
            stride_size=1,
            filters=64,
        ),
        predecessors=[],
        receptive_field_info_filter=KNOWN_FILTER_MAPPING["inf"],
    )

    l1 = EnrichedNetworkNode(
        name="Conv1",
        layer_info=LayerDefinition(
            name="Conv3x3",
            kernel_size=(3, 3),
            stride_size=(1, 1),
            filters=64,
        ),
        predecessors=[l0],
        receptive_field_info_filter=KNOWN_FILTER_MAPPING["inf"],
    )

    l2 = EnrichedNetworkNode(
        name="Conv2",
        layer_info=LayerDefinition(
            name="Conv3x3",
            kernel_size=3,
            stride_size=(1, 1),
            filters=64,
        ),
        predecessors=[l0],
        receptive_field_info_filter=KNOWN_FILTER_MAPPING["inf"],
    )

    l3 = EnrichedNetworkNode(
        name="Conv3",
        layer_info=LayerDefinition(
            name="Conv3x3",
            kernel_size=3,
            stride_size=1,
            filters=64,
        ),
        predecessors=[l0],
        receptive_field_info_filter=KNOWN_FILTER_MAPPING["inf"],
    )

    l4 = EnrichedNetworkNode(
        name="Conv4",
        layer_info=LayerDefinition(
            name="Conv3x3",
            kernel_size=(np.inf, 3),
            stride_size=(1, 1),
            filters=64,
        ),
        predecessors=[l0],
        receptive_field_info_filter=KNOWN_FILTER_MAPPING["inf"],
    )

    l5 = EnrichedNetworkNode(
        name="Conv5",
        layer_info=LayerDefinition(
            name="Conv3x3",
            kernel_size=(np.inf, np.inf),
            stride_size=(1, 1),
            filters=64,
        ),
        predecessors=[l0],
        receptive_field_info_filter=KNOWN_FILTER_MAPPING["inf"],
    )

    l6 = EnrichedNetworkNode(
        name="Conv6",
        layer_info=LayerDefinition(
            name="Conv3x3",
            kernel_size=np.inf,
            stride_size=(1, 1),
            filters=64,
        ),
        predecessors=[l0],
        receptive_field_info_filter=KNOWN_FILTER_MAPPING["inf"],
    )

    return [l1, l2, l3, l4, l5, l6]


@pytest.fixture()
def model_with_scalar_inf_filter_infinity_predecessors(list_of_predecessors_inf_filter):
    return EnrichedNetworkNode(
        name="Conv7",
        layer_info=LayerDefinition(
            name="Conv3x3",
            kernel_size=(1, 1),
            stride_size=(1, 1),
            filters=64,
        ),
        predecessors=list_of_predecessors_inf_filter,
        receptive_field_info_filter=KNOWN_FILTER_MAPPING["inf"],
    )


class TestFiltering:
    def test_noop_filter_does_nothing(
        self, model_with_scalar_noop_infinity_predecessors
    ):
        node = model_with_scalar_noop_infinity_predecessors
        print(node.receptive_field_info)
        print(node.all_layers)
        assert len(node.all_layers) - 2 == len(node.receptive_field_sizes)

    def test_inf_filters_scalar_infinities(self, list_of_predecessors_noop):
        r = EnrichedNetworkNode(
            name="Conv7",
            layer_info=LayerDefinition(
                name="Conv3x3",
                kernel_size=1,
                stride_size=1,
                filters=64,
            ),
            predecessors=list_of_predecessors_noop,
            receptive_field_info_filter=KNOWN_FILTER_MAPPING["inf"],
        )
        for rf in r.receptive_field_sizes:
            assert rf != np.inf

    def test_inf_filters_tuple_infinities(self, list_of_predecessors_noop):
        r = EnrichedNetworkNode(
            name="Conv7",
            layer_info=LayerDefinition(
                name="Conv3x3",
                kernel_size=1,
                stride_size=1,
                filters=64,
            ),
            predecessors=list_of_predecessors_noop,
            receptive_field_info_filter=KNOWN_FILTER_MAPPING["inf"],
        )
        for rf in r.receptive_field_sizes:
            if isinstance(rf, Sequence):
                assert rf != (np.inf, np.inf)

    def test_inf_filters_partial_infinity_tuples(self, list_of_predecessors_noop):
        r = EnrichedNetworkNode(
            name="Conv7",
            layer_info=LayerDefinition(
                name="Conv3x3",
                kernel_size=1,
                stride_size=1,
                filters=64,
            ),
            predecessors=list_of_predecessors_noop,
            receptive_field_info_filter=KNOWN_FILTER_MAPPING["inf"],
        )
        for rf in r.receptive_field_sizes:
            if isinstance(rf, Sequence):
                assert rf != (np.inf, 3)

    def test_inf_behaves_like_noop_if_everything_is_filtered(
        self, list_of_predecessors_noop
    ):
        node = EnrichedNetworkNode(
            name="Conv7",
            layer_info=LayerDefinition(
                name="Conv3x3",
                kernel_size=1,
                stride_size=1,
                filters=64,
            ),
            predecessors=[
                list_of_predecessors_noop[3],
                list_of_predecessors_noop[4],
                list_of_predecessors_noop[5],
            ],
            receptive_field_info_filter=KNOWN_FILTER_MAPPING["inf"],
        )
        assert len(node.receptive_field_info) == 3

    def test_inf_filter_on_full_network_noop(self, list_of_predecessors_inf_filter):
        node = EnrichedNetworkNode(
            name="Conv7",
            layer_info=LayerDefinition(
                name="Conv3x3",
                kernel_size=1,
                stride_size=1,
                filters=64,
            ),
            predecessors=[
                list_of_predecessors_inf_filter[3],
                list_of_predecessors_inf_filter[4],
                list_of_predecessors_inf_filter[5],
            ],
            receptive_field_info_filter=KNOWN_FILTER_MAPPING["inf"],
        )
        assert len(node.receptive_field_info) == 3

    def test_inf_filter_on_full_network(self, list_of_predecessors_inf_filter):
        node = EnrichedNetworkNode(
            name="Conv7",
            layer_info=LayerDefinition(
                name="Conv3x3",
                kernel_size=1,
                stride_size=1,
                filters=64,
            ),
            predecessors=list_of_predecessors_inf_filter,
            receptive_field_info_filter=KNOWN_FILTER_MAPPING["inf"],
        )
        assert len(node.receptive_field_info) == 3
        for rf in node.receptive_field_sizes:
            if isinstance(rf, Sequence):
                assert rf != (np.inf, 3)
                assert rf != (np.inf, np.inf)
            assert rf != np.inf
