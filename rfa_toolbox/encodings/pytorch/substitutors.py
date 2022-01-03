import re

from attr import attrib, attrs

from rfa_toolbox.encodings.pytorch.domain import NodeSubstitutor
from rfa_toolbox.graphs import EnrichedNetworkNode


@attrs(auto_attribs=True, frozen=True, slots=True)
class PatternSubstitutor(NodeSubstitutor):
    """A simple substitutor that removes nodes from the graph, based on
    on a regular expression matching the name of the node.

    Args:
        substring:  a string that can be interpreted as a regular expression.

    """

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
    """A factory producing a Substitutor-Instance, which removes
    functionless output-nodes from the compute-graph.
    """
    return PatternSubstitutor(substring=r"inp_\d+")


def input_substitutor() -> PatternSubstitutor:
    """A factory producing a Substitutor-Instance, which removes
    functionless input-nodes from the compute-graph.
    """
    return PatternSubstitutor(substring=r"out_\d+")


def numeric_substitutor() -> PatternSubstitutor:
    """A factory producing a Substitutor-Instance, which removes
    nodes from the compute-graph that have an resolvable name.
    """
    return PatternSubstitutor(substring=r"\d")
