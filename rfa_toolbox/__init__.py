__version__ = "1.7.0"
try:
    # flake8: noqa: F401
    from rfa_toolbox.encodings.pytorch.ingest_architecture import (
        create_graph_from_model as create_graph_from_pytorch_model,
    )
except ImportError:

    def create_graph_from_pytorch_model(*args, **kwargs):
        raise ImportError("This function is not available, torch not installed")


try:
    # flake8: noqa: F401
    from rfa_toolbox.encodings.tensorflow_keras.ingest_architecture import (
        create_graph_from_model as create_graph_from_tensorflow_model,
    )
except ImportError:

    def create_graph_from_tensorflow_model(*args, **kwargs):
        raise ImportError("This function is not available, tensorflow not installed")


# flake8: noqa: F401
from rfa_toolbox.utils.graph_utils import input_resolution_range

# flake8: noqa: F401
from rfa_toolbox.vizualize import visualize_architecture
