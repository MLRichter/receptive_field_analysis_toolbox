from typing import Callable, List

from rfa_toolbox.graphs import EnrichedNetworkNode, LayerDefinition


def conv_batch_norm_relu(
    predecessor: EnrichedNetworkNode, idx: str, strides: int = 1
) -> EnrichedNetworkNode:
    return EnrichedNetworkNode(
        name=f"{idx}-Conv3x3-BatchNorm-ReLU",
        layer_info=LayerDefinition(
            name="Conv3x3-BatchNorm-ReLU", kernel_size=3, stride_size=strides
        ),
        predecessors=[predecessor],
    )


def conv_batch_norm_relu_squeeze(
    predecessor: EnrichedNetworkNode, idx: str, strides: int = 1
) -> EnrichedNetworkNode:
    return EnrichedNetworkNode(
        name=f"{idx}-Conv1x1-BatchNorm-ReLU",
        layer_info=LayerDefinition(
            name="Conv1x1-BatchNorm-ReLU", kernel_size=1, stride_size=strides
        ),
        predecessors=[predecessor],
    )


def addition(predecessor: List[EnrichedNetworkNode], idx: str) -> EnrichedNetworkNode:
    return EnrichedNetworkNode(
        name=f"{idx}-Addition",
        layer_info=LayerDefinition(name="Addition", kernel_size=1, stride_size=1),
        predecessors=predecessor,
    )


def skip_downsample(predecessor: EnrichedNetworkNode, idx: str) -> EnrichedNetworkNode:
    return EnrichedNetworkNode(
        name=f"{idx}-SkipDownsample",
        layer_info=LayerDefinition(
            name="1x1Conv-Projection", kernel_size=1, stride_size=2
        ),
        predecessors=[predecessor],
    )


def stem() -> EnrichedNetworkNode:
    conv = EnrichedNetworkNode(
        name="StemConv",
        layer_info=LayerDefinition(name="Conv7x7", kernel_size=7, stride_size=2),
        predecessors=[],
    )
    pool = EnrichedNetworkNode(
        name="StemPool",
        layer_info=LayerDefinition(name="MaxPool3x3", kernel_size=3, stride_size=2),
        predecessors=[conv],
    )
    return pool


def small_stem() -> EnrichedNetworkNode:
    conv = EnrichedNetworkNode(
        name="StemConv",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=1),
        predecessors=[],
    )
    return conv


def mediun_stem() -> EnrichedNetworkNode:
    conv = EnrichedNetworkNode(
        name="StemConv",
        layer_info=LayerDefinition(name="Conv3x3", kernel_size=3, stride_size=2),
        predecessors=[],
    )
    return conv


def residual_block(
    input_node: EnrichedNetworkNode, idx: int, i: int, strides: int = 1
) -> EnrichedNetworkNode:
    conv1 = conv_batch_norm_relu(input_node, f"Stage{idx}-Block{i}-{0}", strides)
    conv2 = conv_batch_norm_relu(conv1, f"Stage{idx}-Block{i}-{1}", 1)

    residual = (
        input_node
        if strides == 1
        else skip_downsample(predecessor=input_node, idx=f"BlockSkip{idx}-")
    )
    add = addition([residual, conv2], f"Stage{idx}-Block{i}")

    return add


def bottleneck(
    input_node: EnrichedNetworkNode, idx: int, i: int, strides: int = 1
) -> EnrichedNetworkNode:
    conv1 = conv_batch_norm_relu_squeeze(
        input_node, f"Stage{idx}-Block{i}-{0}", strides
    )
    print(f"\tStage{idx}-Block{i}-{0}")
    conv2 = conv_batch_norm_relu(conv1, f"Stage{idx}-Block{i}-{1}", 1)
    print(f"\tStage{idx}-Block{i}-{1}")
    conv3 = conv_batch_norm_relu_squeeze(conv2, f"Stage{idx}-Block{i}-{2}", 1)
    print(f"\tStage{idx}-Block{i}-{2}")

    residual = (
        input_node
        if strides == 1
        else skip_downsample(predecessor=input_node, idx=f"BlockSkip{idx}-")
    )
    add = addition([residual, conv3], f"Stage{idx}-Block{i}")
    print(f"\tStage{idx}-Block{i}-Add")

    return add


def head(feature_extractor: EnrichedNetworkNode) -> EnrichedNetworkNode:
    readout = EnrichedNetworkNode(
        name="GlobalAveragePooling",
        layer_info=LayerDefinition(
            name="Global Average Pooling", kernel_size=1, stride_size=None
        ),
        predecessors=[feature_extractor],
    )
    print("GAP")
    softmax = EnrichedNetworkNode(
        name="Softmax",
        layer_info=LayerDefinition(
            name="DenseLayer", kernel_size=None, stride_size=None
        ),
        predecessors=[readout],
    )
    print("Softmax")
    return softmax


def stage(
    predecessor: EnrichedNetworkNode,
    idx: int,
    num_blocks: int,
    strides: int,
    block: Callable[[EnrichedNetworkNode, int, int, int], EnrichedNetworkNode],
) -> EnrichedNetworkNode:
    current_block = block(predecessor, idx, 0, strides)
    for i in range(1, num_blocks):
        print("Bulding stage", idx, "block", i)
        current_block = block(current_block, idx, i, 1)
    return current_block


def resnet(
    stage_config: List[int],
    block: Callable[[EnrichedNetworkNode, int, int], EnrichedNetworkNode],
    stem_factory=stem,
) -> EnrichedNetworkNode:
    stem_out = stem_factory()
    c_block = stem_out
    for i, c_stage in enumerate(stage_config):
        c_block = stage(
            predecessor=c_block,
            idx=i,
            num_blocks=c_stage,
            strides=1 if i == 0 else 2,
            block=block,
        )
    output = head(c_block)
    return output


def resnet18() -> EnrichedNetworkNode:
    return resnet(stage_config=[2, 2, 2, 2], block=residual_block)


def resnet36() -> EnrichedNetworkNode:
    return resnet(stage_config=[3, 4, 6, 3], block=residual_block)


def resnet50() -> EnrichedNetworkNode:
    return resnet(stage_config=[3, 4, 6, 3], block=bottleneck)


def resnet101() -> EnrichedNetworkNode:
    return resnet(stage_config=[3, 4, 23, 3], block=bottleneck)


def resnet152() -> EnrichedNetworkNode:
    return resnet(stage_config=[3, 8, 36, 3], block=bottleneck)
