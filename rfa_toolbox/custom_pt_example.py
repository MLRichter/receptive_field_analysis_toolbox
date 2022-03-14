from torch import nn

from rfa_toolbox import create_graph_from_pytorch_model, visualize_architecture

mdl = nn.Sequential(
    nn.Conv2d(
        kernel_size=(11, 11), in_channels=3, out_channels=32, stride=1, padding=1
    ),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(kernel_size=(7, 7), in_channels=32, out_channels=64, stride=1, padding=1),
    nn.ReLU(),
    nn.Conv2d(kernel_size=(5, 5), in_channels=64, out_channels=64, stride=2, padding=1),
    nn.ReLU(),
    nn.Conv2d(
        kernel_size=(5, 5), in_channels=64, out_channels=128, stride=1, padding=1
    ),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(in_features=128, out_features=10),
)


def main():
    # Create graph from model
    graph = create_graph_from_pytorch_model(mdl)
    # Visualize the graph
    dot = visualize_architecture(graph, "Example", input_res=32, include_fm_info=False)
    dot.unflatten(stagger=1)
    dot.render("Example")


if __name__ == "__main__":
    main()
