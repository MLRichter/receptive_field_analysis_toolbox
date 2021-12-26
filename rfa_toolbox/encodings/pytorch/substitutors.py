import re

from attr import attrib, attrs

from rfa_toolbox.encodings.pytorch.domain import NodeSubstitutor
from rfa_toolbox.graphs import EnrichedNetworkNode


@attrs(auto_attribs=True, frozen=True, slots=True)
class PatternSubstitutor(NodeSubstitutor):

    substring: re.Pattern = attrib(converter=re.compile)

    def can_handle(self, name: str) -> bool:
        return bool(self.substring.match(name))

    def __call__(self, node: EnrichedNetworkNode):
        if node.layer_info.kernel_size != 1 or node.layer_info.stride_size != 1:
            raise ValueError(
                f"{node} is an illegal argument for being substituted with {self}, "
                f"since kernel or stride size is larger than 1!"
            )
        for pred in node.predecessors:
            pred.succecessors.remove(node)
            pred.succecessors.extend(node.succecessors)
        for succ in node.succecessors:
            succ.predecessors.remove(node)
            succ.predecessors.extend(node.predecessors)
        node.all_layers.remove(node)
        return


def output_substitutor() -> PatternSubstitutor:
    return PatternSubstitutor(substring=r"inp_\d+")


def input_substitutor() -> PatternSubstitutor:
    return PatternSubstitutor(substring=r"out_\d+")


def numeric_substitutor() -> PatternSubstitutor:
    return PatternSubstitutor(substring=r"\d")
