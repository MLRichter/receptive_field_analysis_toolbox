from typing import Optional

from rfa_toolbox.graphs import EnrichedNetworkNode, LayerDefinition


def conv_batch_norm_relu(
    predecessor: EnrichedNetworkNode, idx: int, filters: Optional[int] = None
) -> EnrichedNetworkNode:
    return EnrichedNetworkNode(
        name=f"{idx}-Conv3x3-BatchNorm-ReLU",
        layer_info=LayerDefinition(
            name="Conv3x3-BatchNorm-ReLU", kernel_size=3, stride_size=1, filters=filters
        ),
        predecessors=[predecessor],
    )


def max_pooling(predecessor: EnrichedNetworkNode, idx: int) -> EnrichedNetworkNode:
    return EnrichedNetworkNode(
        name=f"{idx}-MaxPooling",
        layer_info=LayerDefinition(name="MaxPooling", kernel_size=2, stride_size=2),
        predecessors=[predecessor],
    )


def head(pool: EnrichedNetworkNode) -> EnrichedNetworkNode:
    readout = EnrichedNetworkNode(
        name="Readout",
        layer_info=LayerDefinition(
            name="DenseLayer", kernel_size=None, stride_size=None
        ),
        predecessors=[pool],
    )
    dense = EnrichedNetworkNode(
        name="Dense",
        layer_info=LayerDefinition(
            name="DenseLayer", kernel_size=None, stride_size=None
        ),
        predecessors=[readout],
    )
    softmax = EnrichedNetworkNode(
        name="Softmax",
        layer_info=LayerDefinition(
            name="DenseLayer", kernel_size=None, stride_size=None
        ),
        predecessors=[dense],
    )

    return softmax


def vgg11() -> EnrichedNetworkNode:
    input_node = EnrichedNetworkNode(
        name="input",
        layer_info=LayerDefinition(name="Input", kernel_size=1, stride_size=1),
        predecessors=[],
    )
    # stage 1
    conv = conv_batch_norm_relu(input_node, 0)
    pool = max_pooling(conv, 0)

    # stage 2
    conv = conv_batch_norm_relu(pool, 1)
    pool = max_pooling(conv, 1)

    # stage 3
    conv = conv_batch_norm_relu(pool, 2)
    conv = conv_batch_norm_relu(conv, 3)
    pool = max_pooling(conv, 2)

    # stage 4
    conv = conv_batch_norm_relu(pool, 4)
    conv = conv_batch_norm_relu(conv, 5)
    pool = max_pooling(conv, 3)

    # stage 5
    conv = conv_batch_norm_relu(pool, 6)
    conv = conv_batch_norm_relu(conv, 7)
    pool = max_pooling(conv, 4)

    # head
    output = head(pool)

    return output


def vgg13() -> EnrichedNetworkNode:
    input_node = EnrichedNetworkNode(
        name="input",
        layer_info=LayerDefinition(name="Input", kernel_size=1, stride_size=1),
        predecessors=[],
    )
    # stage 1
    conv = conv_batch_norm_relu(input_node, 0)
    conv = conv_batch_norm_relu(conv, 1)
    pool = max_pooling(conv, 0)

    # stage 2
    conv = conv_batch_norm_relu(pool, 2)
    conv = conv_batch_norm_relu(conv, 3)
    pool = max_pooling(conv, 1)

    # stage 3
    conv = conv_batch_norm_relu(pool, 4)
    conv = conv_batch_norm_relu(conv, 5)
    pool = max_pooling(conv, 2)

    # stage 4
    conv = conv_batch_norm_relu(pool, 6)
    conv = conv_batch_norm_relu(conv, 7)
    pool = max_pooling(conv, 3)

    # stage 5
    conv = conv_batch_norm_relu(pool, 8)
    conv = conv_batch_norm_relu(conv, 9)
    pool = max_pooling(conv, 4)

    # head
    output = head(pool)

    return output


def vgg16() -> EnrichedNetworkNode:
    input_node = EnrichedNetworkNode(
        name="input",
        layer_info=LayerDefinition(name="Input", kernel_size=1, stride_size=1),
        predecessors=[],
    )
    # stage 1
    conv = conv_batch_norm_relu(input_node, 0)
    conv = conv_batch_norm_relu(conv, 1)
    pool = max_pooling(conv, 0)

    # stage 2
    conv = conv_batch_norm_relu(pool, 2)
    conv = conv_batch_norm_relu(conv, 3)
    pool = max_pooling(conv, 1)

    # stage 3
    conv = conv_batch_norm_relu(pool, 4)
    conv = conv_batch_norm_relu(conv, 5)
    conv = conv_batch_norm_relu(conv, 6)
    pool = max_pooling(conv, 2)

    # stage 4
    conv = conv_batch_norm_relu(pool, 7)
    conv = conv_batch_norm_relu(conv, 8)
    conv = conv_batch_norm_relu(conv, 9)
    pool = max_pooling(conv, 3)

    # stage 5
    conv = conv_batch_norm_relu(pool, 10)
    conv = conv_batch_norm_relu(conv, 11)
    conv = conv_batch_norm_relu(conv, 12)
    pool = max_pooling(conv, 4)

    # head
    output = head(pool)

    return output


def vgg19() -> EnrichedNetworkNode:
    input_node = EnrichedNetworkNode(
        name="input",
        layer_info=LayerDefinition(name="Input", kernel_size=1, stride_size=1),
        predecessors=[],
    )
    # stage 1
    conv = conv_batch_norm_relu(input_node, 0)
    conv = conv_batch_norm_relu(conv, 1)
    pool = max_pooling(conv, 0)

    # stage 2
    conv = conv_batch_norm_relu(pool, 2)
    conv = conv_batch_norm_relu(conv, 3)
    pool = max_pooling(conv, 1)

    # stage 3
    conv = conv_batch_norm_relu(pool, 4)
    conv = conv_batch_norm_relu(conv, 5)
    conv = conv_batch_norm_relu(conv, 6)
    conv = conv_batch_norm_relu(conv, 7)
    pool = max_pooling(conv, 2)

    # stage 4
    conv = conv_batch_norm_relu(pool, 8)
    conv = conv_batch_norm_relu(conv, 9)
    conv = conv_batch_norm_relu(conv, 10)
    conv = conv_batch_norm_relu(conv, 11)
    pool = max_pooling(conv, 3)

    # stage 5
    conv = conv_batch_norm_relu(pool, 12)
    conv = conv_batch_norm_relu(conv, 13)
    conv = conv_batch_norm_relu(conv, 14)
    conv = conv_batch_norm_relu(conv, 15)
    pool = max_pooling(conv, 4)

    # head
    output = head(pool)

    return output
