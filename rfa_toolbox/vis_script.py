from rfa_toolbox.architectures.vgg import conv_batch_norm_relu, head, max_pooling
from rfa_toolbox.graphs import EnrichedNetworkNode, LayerDefinition
from rfa_toolbox.vizualize import visualize_architecture


def vgg19() -> EnrichedNetworkNode:
    input_node = EnrichedNetworkNode(
        name="input",
        layer_info=LayerDefinition(name="Input", kernel_size=1, stride_size=1),
        predecessors=[],
    )
    # stage 1
    conv = conv_batch_norm_relu(input_node, 0, filters=64)
    conv = conv_batch_norm_relu(conv, 1, filters=64)
    pool = max_pooling(conv, 0)

    # stage 2
    conv = conv_batch_norm_relu(pool, 2, filters=128)
    conv = conv_batch_norm_relu(conv, 3, filters=128)
    pool = max_pooling(conv, 1)

    # stage 3
    conv = conv_batch_norm_relu(pool, 4, filters=256)
    conv = conv_batch_norm_relu(conv, 5, filters=256)
    conv = conv_batch_norm_relu(conv, 6, filters=256)
    conv = conv_batch_norm_relu(conv, 7, filters=256)
    pool = max_pooling(conv, 2)

    # stage 4
    conv = conv_batch_norm_relu(pool, 8, filters=512)
    conv = conv_batch_norm_relu(conv, 9, filters=512)
    conv = conv_batch_norm_relu(conv, 10, filters=512)
    conv = conv_batch_norm_relu(conv, 11, filters=512)
    pool = max_pooling(conv, 3)

    # stage 5
    conv = conv_batch_norm_relu(pool, 12, filters=512)
    conv = conv_batch_norm_relu(conv, 13, filters=512)
    conv = conv_batch_norm_relu(conv, 14, filters=512)
    conv = conv_batch_norm_relu(conv, 15, filters=512)
    pool = max_pooling(conv, 4)

    # head
    output = head(pool)

    return output


def vgg19_hybrid() -> EnrichedNetworkNode:
    input_node = EnrichedNetworkNode(
        name="input",
        layer_info=LayerDefinition(name="Input", kernel_size=1, stride_size=1),
        predecessors=[],
    )
    # stage 1
    conv = conv_batch_norm_relu(input_node, 0)
    conv = conv_batch_norm_relu(conv, 1)
    # pool = max_pooling(conv, 0)

    # stage 2
    conv = conv_batch_norm_relu(conv, 2)
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


def vgg19_perf() -> EnrichedNetworkNode:
    input_node = EnrichedNetworkNode(
        name="input",
        layer_info=LayerDefinition(name="Input", kernel_size=1, stride_size=1),
        predecessors=[],
    )
    # stage 1
    conv = conv_batch_norm_relu(input_node, 0)
    conv = conv_batch_norm_relu(conv, 1)
    # pool = max_pooling(conv, 0)

    # stage 2
    conv = conv_batch_norm_relu(conv, 2)
    conv = conv_batch_norm_relu(conv, 3)
    # pool = max_pooling(conv, 1)

    # stage 3
    conv = conv_batch_norm_relu(conv, 4)
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


def vgg19_perf2() -> EnrichedNetworkNode:
    input_node = EnrichedNetworkNode(
        name="input",
        layer_info=LayerDefinition(name="Input", kernel_size=1, stride_size=1),
        predecessors=[],
    )
    # stage 1
    conv = conv_batch_norm_relu(input_node, 0, filters=64)
    conv = conv_batch_norm_relu(conv, 1, filters=64)
    conv = conv_batch_norm_relu(conv, 2, filters=128)
    conv = conv_batch_norm_relu(conv, 3, filters=128)
    conv = conv_batch_norm_relu(conv, 4, filters=256)
    conv = conv_batch_norm_relu(conv, 5, filters=256)
    pool = max_pooling(conv, 1)

    # stage 2
    conv = conv_batch_norm_relu(pool, 6, filters=256)
    conv = conv_batch_norm_relu(conv, 7, filters=256)
    conv = conv_batch_norm_relu(conv, 8, filters=512)
    conv = conv_batch_norm_relu(conv, 9, filters=512)
    conv = conv_batch_norm_relu(conv, 10, filters=512)
    conv = conv_batch_norm_relu(conv, 11, filters=512)
    pool = max_pooling(conv, 2)

    # stage 3
    conv = conv_batch_norm_relu(pool, 12, filters=512)
    conv = conv_batch_norm_relu(conv, 13, filters=512)
    conv = conv_batch_norm_relu(conv, 14, filters=512)
    conv = conv_batch_norm_relu(conv, 15, filters=512)
    pool = max_pooling(conv, 4)

    # head
    output = head(pool)

    return output


if __name__ == "__main__":
    m = vgg19_perf2

    dot = visualize_architecture(m(), "vgg19_perf", input_res=16).view()
