from typing import Callable, Dict, List, Optional, Tuple, Union

from rfa_toolbox.graphs import (
    KNOWN_FILTER_MAPPING,
    EnrichedNetworkNode,
    ReceptiveFieldInfo,
)

try:
    import torch
except ImportError:
    pass

from rfa_toolbox.encodings.pytorch.intermediate_graph import Digraph


def _check_white_list(submodule_type, fq_submodule_name, classes_to_visit):
    return (
        classes_to_visit is None
        and (
            not fq_submodule_name.startswith("torch.nn")
            or fq_submodule_name.startswith("torch.nn.modules.container")
        )
    ) or (
        classes_to_visit is not None
        and (
            submodule_type in classes_to_visit or fq_submodule_name in classes_to_visit
        )
    )


def _check_black_list(submodule_type, fq_submodule_name, classes_to_not_visit):
    return (classes_to_not_visit is None) or (
        classes_to_not_visit is not None
        and (
            submodule_type not in classes_to_not_visit
            and fq_submodule_name not in classes_to_not_visit
        )
    )


def _obtain_variable_names(graph: torch._C.Graph) -> Dict[str, str]:
    result = {}
    for node in graph.nodes():
        x, y = str(node).split(" : ")
        key, value = x, y
        result[key] = value
    return result


def _obtain_node_key(node: str) -> str:
    return str(node).split(" : ")[0]


def _obtain_variable_from_node_string_atten(
    line: str, graph_row_dict: Dict[str, str]
) -> List[str]:
    row = graph_row_dict[line]
    layer_call = row.split("aten::")[-1]
    variable_pruned_left = layer_call.split("(")[1]
    variable_pruned_right = variable_pruned_left.split("),")[0]
    variable_name = variable_pruned_right.split(",")
    return [x.replace(" ", "") for x in variable_name]


def _resolve_variable(
    variable: str, graph_row_dict: Dict[str, str]
) -> Union[List[int], int]:
    if "prim::ListConstruct" in variable:
        variable_pruned_left = variable.split("prim::ListConstruct(")[1]
        variable_pruned_right = variable_pruned_left.split("),")[0]
        variable_name = variable_pruned_right.split(",")
        result = [
            _resolve_variable(graph_row_dict[x.replace(" ", "")], graph_row_dict)
            for x in variable_name
        ]
    elif "prim::Constant" in variable and "prim::Constant()" not in variable:
        variable_pruned_left = variable.split("prim::Constant[value=")[1]
        variable_pruned_right = variable_pruned_left.split("]")[0]
        variable = int(variable_pruned_right)
        result = variable
    else:
        result = None
    return result


def _resolve_variables(variables: List[str], graph_row_dict: Dict[str, str]):
    result = []
    for var in variables:
        if var in graph_row_dict:
            var_val = _resolve_variable(graph_row_dict[var], graph_row_dict)
            result.append(var_val)
        else:
            result.append(None)
    return result


def make_graph(
    mod,
    classes_to_visit=None,
    classes_found=None,
    dot=None,
    prefix="",
    input_preds=None,
    parent_dot=None,
    ref_mod=None,
    classes_to_not_visit=None,
    filter_rf=KNOWN_FILTER_MAPPING[None],
):
    """
    This code was adapted from this blog article:
    lernapparat.de/visualize-pytorch-models
    """
    preds = {}

    def find_name(i, self_input, suffix=None):
        if i == self_input:
            return suffix
        try:
            cur = i.node().s("name")
        except RuntimeError:
            return suffix + "-unknownType" if suffix is not None else "unknownType"
        if suffix is not None:
            cur = cur + "." + suffix
        of = next(i.node().inputs())
        return find_name(of, self_input, suffix=cur)

    # compute graph from the JIT
    gr = mod.graph
    self_input = next(gr.inputs())
    self_type = self_input.type().str().split(".")[-1]
    preds[self_input] = (set(), set())  # inps, ops

    if dot is None:
        dot = Digraph(
            ref_mod=ref_mod,
            format="svg",
            graph_attr={"label": self_type, "labelloc": "t"},
            filter_rf=filter_rf,
        )
        # dot.attr('node', shape='box')

    seen_inpnames = set()
    seen_edges = set()

    def add_edge(dot, n1, n2):
        if (n1, n2) not in seen_edges:
            seen_edges.add((n1, n2))
            dot.edge(n1, n2)

    def make_edges(pr, inpname, name, op, edge_dot=dot):
        if op:
            if inpname not in seen_inpnames:
                seen_inpnames.add(inpname)
                label_lines = [[]]
                line_len = 0
                for w in op:
                    if line_len >= 20:
                        label_lines.append([])
                        line_len = 0
                    label_lines[-1].append(w)
                    line_len += len(w) + 1
                edge_dot.node(
                    inpname,
                    label="\n".join([" ".join(w) for w in label_lines]),
                    shape="box",
                    style="rounded",
                )
                for p in pr:
                    add_edge(edge_dot, p, inpname)
            add_edge(edge_dot, inpname, name)
        else:
            for p in pr:
                add_edge(edge_dot, p, name)

    for nr, i in enumerate(list(gr.inputs())[1:]):
        name = prefix + "inp_" + i.debugName()
        preds[i] = {name}, set()
        dot.node(name, shape="ellipse")
        if input_preds is not None:
            pr, op = input_preds[nr]
            make_edges(pr, "inp_" + name, name, op, edge_dot=parent_dot)

    def is_relevant_type(t):
        kind = t.kind()
        if kind == "TensorType":
            return True
        if kind in ("ListType", "OptionalType"):
            return is_relevant_type(t.getElementType())
        if kind == "TupleType":
            return any([is_relevant_type(tt) for tt in t.elements()])
        return False

    variable_names = _obtain_variable_names(graph=gr)

    for n in gr.nodes():
        # this seems to be uninteresting for resnet-style models
        only_first_ops = {"aten::expand_as"}
        rel_inp_end = 1 if n.kind() in only_first_ops else None
        # rel_inp_end
        # this filters input and outputs and gets rid of all irrelevant stuff
        # relevant is any module that either contains a TensorType-Module or is
        # such a Module
        relevant_inputs = [
            i for i in list(n.inputs())[:rel_inp_end] if is_relevant_type(i.type())
        ]
        relevant_outputs = [o for o in n.outputs() if is_relevant_type(o.type())]
        if n.kind() == "prim::CallMethod":

            # full module name
            fq_submodule_name = ".".join(
                [
                    nc
                    for nc in list(n.inputs())[0].type().str().split(".")
                    if not nc.startswith("__")
                ]
            )
            # the name of the module class
            submodule_type = list(n.inputs())[0].type().str().split(".")[-1]

            # name
            submodule_name = find_name(list(n.inputs())[0], self_input)
            # print()
            # print(fq_submodule_name)
            # print(submodule_type)
            # print(submodule_name)
            # print()
            name = prefix + "." + n.output().debugName()
            # print(name)
            label = prefix + submodule_name + " (" + submodule_type + ")"
            printable = prefix + submodule_name
            pr = ""
            for elem in printable.split("."):
                if elem.isnumeric():
                    elem = "[" + elem + "]."
                pr += elem

            if classes_found is not None:
                classes_found.add(fq_submodule_name)
            if _check_white_list(
                submodule_type, fq_submodule_name, classes_to_visit
            ) and _check_black_list(
                submodule_type, fq_submodule_name, classes_to_not_visit
            ):
                # go into subgraph
                sub_prefix = prefix + submodule_name + "."
                with dot.subgraph(name="cluster_" + name) as sub_dot:
                    # sub_dot.attr(label=label)
                    submod = mod
                    # iterate to the lowest submodule hirarchy
                    for i, k in enumerate(submodule_name.split(".")):
                        submod = getattr(submod, k)
                        # create subgraph for the submodule
                    make_graph(
                        submod,
                        dot=sub_dot,
                        prefix=sub_prefix,
                        input_preds=[preds[i] for i in list(n.inputs())[1:]],
                        parent_dot=dot,
                        classes_to_visit=classes_to_visit,
                        classes_found=classes_found,
                        classes_to_not_visit=classes_to_not_visit,
                        filter_rf=filter_rf,
                    )
                    # creating a mapping from the c-values
                    # to the output of the respective subgraph
                    for i, o in enumerate(n.outputs()):
                        # print(i, sub_prefix + f'out_{i}', type(o))
                        preds[o] = {sub_prefix + f"out_{i}"}, set()
            else:
                # here the basic node (Conv2D, BatchNorm etc.) are created.
                dot.node(name, label=label, shape="box")
                for i in relevant_inputs:
                    # create edges between predecessor and node
                    pr, op = preds[i]
                    make_edges(pr, prefix + i.debugName(), name, op)
                # register the node in the preds dict
                for o in n.outputs():
                    # print(o, name)
                    preds[o] = {name}, set()
        elif n.kind() == "prim::CallFunction":
            # this code is not touched by ResNet
            funcname = list(n.inputs())[0].type().__repr__().split(".")[-1]
            name = prefix + "." + n.output().debugName()
            label = funcname
            dot.node(name, label=label, shape="box")
            for i in relevant_inputs:
                pr, op = preds[i]
                make_edges(pr, prefix + i.debugName(), name, op)
            for o in n.outputs():
                preds[o] = {name}, set()
        elif "aten::" in n.kind() and "pool" in n.kind() and "adaptive" not in n.kind():
            name = prefix + "." + n.output().debugName()
            label = n.kind().split("::")[-1]
            key = _obtain_node_key(str(n))
            vars = _obtain_variable_from_node_string_atten(key, variable_names)
            resolved = _resolve_variables(vars, variable_names)
            dot.node(
                name,
                label=label,
                shape="box",
                kernel_size=resolved[1],
                stride_size=resolved[2],
            )
            for i in relevant_inputs:
                pr, op = preds[i]
                make_edges(pr, prefix + i.debugName(), name, op)
            for o in n.outputs():
                preds[o] = {name}, set()
        else:
            unseen_ops = {
                "prim::ListConstruct",
                "prim::TupleConstruct",
                "aten::index",
                "aten::size",
                "aten::slice",
                "aten::unsqueeze",
                "aten::squeeze",
                "aten::to",
                "aten::view",
                "aten::permute",
                "aten::transpose",
                "aten::contiguous",
                "aten::permute",
                "aten::Int",
                "prim::TupleUnpack",
                "prim::ListUnpack",
                "aten::unbind",
                "aten::select",
                "aten::detach",
                "aten::stack",
                "aten::reshape",
                "aten::split_with_sizes",
                # "aten::cat",
                "aten::expand",
                "aten::expand_as",
                "aten::_shape_as_tensor",
            }

            absorbing_ops = (
                "aten::size",
                "aten::_shape_as_tensor",
            )  # probably also partially absorbing ops. :/
            if True:
                # if this is neither a function nor a module with usefull nodes
                # for example addition-functions
                label = n.kind().split("::")[-1].rstrip("_")
                pr, op = set(), set()
                for i in relevant_inputs:
                    apr, aop = preds[i]
                    # union of pr and apr / op and aop
                    pr |= apr
                    op |= aop
                # if pr and n.kind() not in unseen_ops:
                #    print(n.kind(), n)
                if n.kind() in absorbing_ops:
                    pr, op = set(), set()
                elif (
                    len(relevant_inputs) > 0
                    and len(relevant_outputs) > 0
                    and n.kind() not in unseen_ops
                ):
                    op.add(label)
                for o in n.outputs():
                    # print(o, pr, op)
                    preds[o] = pr, op

    for i, o in enumerate(gr.outputs()):
        name = prefix + f"out_{i}"
        dot.node(name, shape="ellipse")
        pr, op = preds[o]
        make_edges(pr, "inp_" + name, name, op)
    return dot


def create_graph_from_model(
    model: torch.nn.Module,
    filter_rf: Optional[
        Union[
            Callable[[Tuple[ReceptiveFieldInfo, ...]], Tuple[ReceptiveFieldInfo, ...]],
            str,
        ]
    ] = None,
    input_res: Tuple[int, int, int, int] = (1, 3, 399, 399),
    custom_layers: Optional[List[str]] = None,
    display_se_modules: bool = False,
) -> EnrichedNetworkNode:
    """Create a graph of enriched network nodes from a PyTorch-Model.

    Args:
        model:          a PyTorch-Model.
        filter_rf:      a function that filters receptive field sizes.
                        Disabled by default.
        input_res:      input-tuple shape that can be processed by the model.
                        Needs to be a 4-Tuple of shape (batch_size,
                        color_channels, height, width) for CNNs.
                        Needs to be a 2-Tuple of shape (batch_size,
                        num_features) for fully connected networks.
        custom_layers:  Class-names of custom layers, like DropPath
                        or Involutions, which are not part of
                        torch.nn. Keep in mind that unknown layers
                        will defaulted to have no effect on the
                        receptive field size. You may need to
                        implement some additional layer handlers.
        display_se_modules: False by default. If True, displays the structure
                        inside Squeeze-and-Excitation modules and considers their
                        maximum receptive field size infinite, which is technically
                        closer to the truth but irrelevant in practice.

    Returns:
        The EnrichedNetworkNodeGraph
    """
    custom_layers = (
        ["ConvNormActivation"]
        if custom_layers is None
        else custom_layers + ["ConvNormActivation"]
    )
    if not display_se_modules:
        custom_layers.append("SqueezeExcitation")
    filter_func = (
        filter_rf
        if (not isinstance(filter_rf, str) and filter_rf is not None)
        else KNOWN_FILTER_MAPPING[filter_rf]
    )
    tm = torch.jit.trace(model, (torch.randn(*input_res),))
    return make_graph(
        tm, filter_rf=filter_func, ref_mod=model, classes_to_not_visit=custom_layers
    ).to_graph()
