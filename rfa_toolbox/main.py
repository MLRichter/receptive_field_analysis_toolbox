import graphviz


def add(n1, n2):
    return n1 + n2


def find_border_layer():
    ...


"""https://graphviz.org/Gallery/directed/hello.html"""
"""https://graphviz.org/Gallery/directed/fsm.html"""


if __name__ == "__main__":
    f = graphviz.Digraph("finite_state_machine", filename="fsm.gv")
    f.attr(rankdir="TB", size="10,5")

    f.attr("node", shape="rectangle")
    f.node("Input", label="Input")
    f.node("1-Conv3x3", label="Conv3x3")
    f.node("1-BatchNorm", label="BatchNorm")
    f.node("1-ReLU", label="ReLU")
    f.node("2-Conv3x3", label="Conv3x3")
    f.node("2-BatchNorm", label="BatchNorm")
    f.node("2-ReLU", label="ReLU")
    f.node("Add", label="+")

    f.edge("Input", "1-Conv3x3", label="")
    f.edge("1-Conv3x3", "1-BatchNorm", label="")
    f.edge("1-BatchNorm", "1-ReLU", label="")
    f.edge("1-ReLU", "2-Conv3x3", label="")
    f.edge("2-Conv3x3", "2-BatchNorm", label="")
    f.edge("2-BatchNorm", "2-ReLU", label="")
    f.edge("Input", "Add", label="")
    f.edge("2-ReLU", "Add", label="")

    f.view()
