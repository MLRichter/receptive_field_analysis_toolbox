from rfa_toolbox.graphs import EnrichedNetworkNode, LayerDefinition

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

import torch


class LayerInfoHandler(Protocol):
    """Creates a LayerDefinition from the model and a resolvable string."""

    def can_handle(self, name: str) -> bool:
        """Checks if this handler can process the
        node in the compute graph of the model.

        Args:
            name: the name of the compute graph.

        Returns:
            True if the node can be processed into a
            valid LayerDefinition by this handler.
        """
        ...

    def __call__(
        self, model: torch.nn.Module, resolvable_string: str, name: str
    ) -> LayerDefinition:
        """Transform a node in the JIT-compiled compute-graph into a valid LayerDefinition
        for further processing and visualization.

        Args:
            model:              the entire model in a non-jit-compiled versionen
            resolvable_string:  a string extracted from the jit-compiled
                                version of the model.
                                the string is formatted in a way that allows
                                the extraction of the module the compute-node
                                in the jit-compiled version is referencing.
                                This string is only resolvable if and only if
                                the node is actually referencing a module.
                                This is not the case if the model uses for
                                example calls from the  functional-library
                                of PyTorch.
                                This is commonly the case for stateless-elements
                                of a model like activation functions or
                                "logistical" operation like concatenations,
                                and additions etc.
                                that are necessary for the flow of information.
                                In such cases the resolvable string only holds some
                                information of the position of the function call
                                within the forward-call and the name of the
                                called function the name of the node in question.

        Returns:
            A LayerDefinition that reflects the properties of the layer.
        """
        ...


class NodeSubstitutor(Protocol):
    """This object handles the conditional substituion of adjacent
    nodes into one single node in the network graph.
    Substitution is not necessary. However, when extracting the
    compute-graph from the JIT-compiler artifact-nodes caused by
    technical details of the module structure may cause an unecessarily
    busy graph-structure.
    """

    def can_handle(self, name: str) -> bool:
        """Check if this substituor can handle the node in question.

        Args:
            name: the name of the EnrichedNetworkNode-object

        Returns:
            True if this substituor can substitute the node in question.

        """
        ...

    def __call__(self, node: EnrichedNetworkNode) -> EnrichedNetworkNode:
        """Remove the node in question from the compute graph.

        Args:
            node: the node to remove

        Returns:
            another node from the repaired compute-graph.

        """
        ...
