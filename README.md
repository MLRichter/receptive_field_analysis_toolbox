# ReceptiveFieldAnalysisToolbox

<p align="center">
  <a href="https://github.com/MLRichter/receptive_field_analysis_toolbox/actions?query=workflow%3ACI">
    <img src="https://img.shields.io/github/workflow/status/MLRichter/receptive_field_analysis_toolbox/CI/main?label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://receptive-field-analysis-toolbox.readthedocs.io/en/latest/index.html">
    <img src="https://img.shields.io/readthedocs/receptive-field-analysis-toolbox.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
    <a href="https://codecov.io/gh/MLRichter/receptive_field_analysis_toolbox">
        <img src="https://codecov.io/gh/MLRichter/receptive_field_analysis_toolbox/branch/main/graph/badge.svg?token=3K52NPEAEU"/>
    </a>
</p>
<p align="center">
  <a href="https://python-poetry.org/">
    <img src="https://img.shields.io/badge/packaging-poetry-299bd7?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAASCAYAAABrXO8xAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAJJSURBVHgBfZLPa1NBEMe/s7tNXoxW1KJQKaUHkXhQvHgW6UHQQ09CBS/6V3hKc/AP8CqCrUcpmop3Cx48eDB4yEECjVQrlZb80CRN8t6OM/teagVxYZi38+Yz853dJbzoMV3MM8cJUcLMSUKIE8AzQ2PieZzFxEJOHMOgMQQ+dUgSAckNXhapU/NMhDSWLs1B24A8sO1xrN4NECkcAC9ASkiIJc6k5TRiUDPhnyMMdhKc+Zx19l6SgyeW76BEONY9exVQMzKExGKwwPsCzza7KGSSWRWEQhyEaDXp6ZHEr416ygbiKYOd7TEWvvcQIeusHYMJGhTwF9y7sGnSwaWyFAiyoxzqW0PM/RjghPxF2pWReAowTEXnDh0xgcLs8l2YQmOrj3N7ByiqEoH0cARs4u78WgAVkoEDIDoOi3AkcLOHU60RIg5wC4ZuTC7FaHKQm8Hq1fQuSOBvX/sodmNJSB5geaF5CPIkUeecdMxieoRO5jz9bheL6/tXjrwCyX/UYBUcjCaWHljx1xiX6z9xEjkYAzbGVnB8pvLmyXm9ep+W8CmsSHQQY77Zx1zboxAV0w7ybMhQmfqdmmw3nEp1I0Z+FGO6M8LZdoyZnuzzBdjISicKRnpxzI9fPb+0oYXsNdyi+d3h9bm9MWYHFtPeIZfLwzmFDKy1ai3p+PDls1Llz4yyFpferxjnyjJDSEy9CaCx5m2cJPerq6Xm34eTrZt3PqxYO1XOwDYZrFlH1fWnpU38Y9HRze3lj0vOujZcXKuuXm3jP+s3KbZVra7y2EAAAAAASUVORK5CYII=" alt="Poetry">
  </a>
  <a href="https://github.com/ambv/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square" alt="black">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/rfa_toolbox/">
    <img src="https://img.shields.io/pypi/v/rfa_toolbox.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/rfa_toolbox.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/rfa_toolbox.svg?style=flat-square" alt="License">
</p>

This is RFA-Toolbox, a simple and easy-to-use library that allows you to optimize your neural network architectures
using receptive field analysis (RFA) and graph visualizations of your architecture.

## Installation

Install this via pip (or your favourite package manager):

`pip install rfa_toolbox`

## What is Receptive Field Analysis?

Receptive Field Analysis (RFA) is simple yet effective way to optimize the efficiency of any neural architecture without
training it.

## Usage

This library allows you to look for certain inefficiencies withing your convolutional neural network setup without
ever training the model.
You can do this simply by importing your architecture into the format of RFA-Toolbox and then use the in-build functions
to visualize your architecture using GraphViz.
Unproductive (i.e. useless) layers marked in red and layers that run into a risk
of being unproductive marked in orange.
This is especially useful if you plan to train your model on resolutions that are substantially lower than the
design-resolution of most models.
As an alternative, you can also use the graph from RFA-Toolbox directly to automatically detect and remove
unproductive layers within your NAS-system without the need for any visualization.

### Examples

There are multiple ways to import your model into RFA-Toolbox for analysis, with additional ways being added in future
releases.

#### PyTorch

The simplest way of importing model is by directly extracting the compute-graph of your model from the PyTorch model
itself.
Here is a simple example:

```python
import torchvision
from rfa_toolbox import create_graph_from_pytorch_model, visualize_architecture
model = torchvision.models.alexnet()
graph = create_graph_from_pytorch_model(model)
visualize_architecture(
    graph, f"alexnet_32_pixel", input_res=32
).render(f"alexnet_32_pixel")
```

This will create a graph of your model and visualize it using GraphViz and color all layers that are predicted to be
unproductive for an input resolution of 32x32 pixels:
![rf_stides.PNG](https://github.com/MLRichter/receptive_field_analysis_toolbox/blob/main/images/alexnet.PNG?raw=true)

Keep in mind that the Graph is reverse-engineerd from the PyTorch JIT-compiler, therefore no looping-logic is
allowed within the forward pass of the model.

#### Custom

If you are not able to automatically import your model from PyTorch or you just want some visualization, you can
also directly implement the model with the propriatary-Graph-format of RFA-Toolbox.
This is similar to coding a compute-graph in a declarative style like in TensorFlow 1.x.

```python
from rfa_toolbox import visualize_architecture
from rfa_toolbox.graphs import EnrichedNetworkNode, LayerDefinition


conv1 = EnrichedNetworkNode(
    name="Conv1",
    layer_info=LayerDefinition(
        name="Conv3x3",
        kernel_size=3, stride_size=1,
        filters=64
    ),
    predecessors=[]
)
conv2 = EnrichedNetworkNode(
    name="Conv2",
    layer_info=LayerDefinition(
        name="Conv3x3",
        kernel_size=3, stride_size=1,
        filters=128
    ),
    predecessors=[conv1]
)

conv3 = EnrichedNetworkNode(
    name="Conv3",
    layer_info=LayerDefinition(
        name="Conv3x3",
        kernel_size=3, stride_size=1,
        filters=256
    ),
    predecessors=[conv1]
)

conv4 = EnrichedNetworkNode(
    name="Conv4",
    layer_info=LayerDefinition(
        name="Conv3x3",
        kernel_size=3, stride_size=1,
        filters=256
    ),
    predecessors=[conv2, conv3]
)

out = EnrichedNetworkNode(
    name="Softmax",
    layer_info=LayerDefinition(
        name="Fully Connected",
        units=1000
    ),
    predecessors=[conv4]
)
visualize_architecture(
    out, f"example_model", input_res=32
).render(f"example_model")

```

This will produce the following graph:
![simple_conv.PNG](https://github.com/MLRichter/receptive_field_analysis_toolbox/blob/main/images/simple_conv.png)
Note that the output layer is marked as a critical layer that is maybe unproductive. This is because dense layers
have technically an infinite receptive field size and therefore using more than a single dense layer in your output
head is generally not a good idea.

### A quick primer on the Receptive Field

To understand how it works, we first need to understand what a receptive field is, and it affects what the network is doing.
Every layer in a (convolutional) neural network has a receptive field. It can be considered the "field of view" of this layer.
In more precise terms, we define a receptive field as the area influencing the output of a single position of the
convolutional kernel.
Here is a simple, 1-dimensional example:
![rf.PNG](https://github.com/MLRichter/receptive_field_analysis_toolbox/blob/main/images/rf.PNG?raw=true)
As you can see, the first layer of this simple architecture can only evaluate the information the input pixels directly
under his kernel, which has a size of three pixels. Of course, in most convolutional neural architectures, this kernel
would be a square area, since most CNN-models process images.
Another observation we can make from this example is that the receptive field size is actually expanding from layer
to layer.
This is happening, because the consecutive layers also have kernel sizes greater than 1 pixel, which means that
they combine multiple adjacent positions on the feature map into a single position in their output, expanding the receptive
field of the previous layer in the process.
In other words, every consecutive layer adds additional context to each feature map position by expanding the receptive field.
This ultimately allows networks to go from detecting small and simple patterns to big and very complicated ones.

The effective size of the kernel is not the only factor influence the growth of the receptive field size.
Another important factor is the stride size:
![rf_stides.PNG](https://github.com/MLRichter/receptive_field_analysis_toolbox/blob/main/images/rf_strides.PNG?raw=true)
The stride size is the size of the step between the individual kernel position. Commonly, every possible
position is evaluated, which is not affecting the receptive field size in any way.
When the stride size is greater than one however valid positions of the kernel are skipped, which reduces
the size of the feature map. Since now information on the feature map is now condensed on fewer feature map positions,
the growth of the receptive field is multiplied for future layers.
In real-world architectures, this is typically the case when downsampling layers like convolutions with a stride size of 2
are used.

### Why does the Receptive Field Matter?

At this point you may be wondering why the receptive field of all things is useful for optimizing an
architecture.
The short answer to this is: because it influences where the network can process patterns of a certain size.
Simply speaking each convolutional layer is only able to detect a certain size of pattern because of its receptive field size.
Interestingly this also means that there is an upper limited to the usefulness of expanding the receptive field.
At the latest, this is the case when the receptive field of a layer is BIGGER than the input image, since no novel
context be added at this point.
For convolutional layers this is in fact a big issue, because layers pas this "Border Layer" are unproductive and
do not contribute anymore to the inference process.
If you are interested in the details of this phenomenon I recommend that you read these papers that investigate this phenomenon
in greater detail:

- [(Input) Size Matters for Convolutional Neural Network Classifier](https://link.springer.com/chapter/10.1007/978-3-030-86340-1_11)
- [Should You Go Deeper? Optimizing Convolutional Neural Network Architectures without Training](https://arxiv.org/abs/2106.12307)
  (published at the 20th IEEE Internation Conference for Machine Learning Application - ICMLA)

### Optimizing Architecture using Receptive Field Analysis

So far, we learned that the expansion of the receptive field allows a layer to enrich the data with more context
and when this is no longer possible, the layer is not able to contribute to the quality of the output of the model.

Of course, being able to predict why and which layer will become dead weight during training is highly useful, since
we can now adjust the design of the architecture to fit our input resolution better.

Let's take for example the good old AlexNet architecture from 2012, which is a very simple CNN-model.
Let's assume we want to train Cifar10 on AlexNet, which has a 32 pixel input resolution, which is lower than the 224
pixel design resolution of the network.
When we apply Receptive receptive field analysis, wen can see that most convolutional layer will in fact not contribute
to the inference process (unproductive layers marked red, probable unproductive layers marked orange):
![rf_stides.PNG](https://github.com/MLRichter/receptive_field_analysis_toolbox/blob/main/images/alexnet.PNG?raw=true)

We can clearly see that most of the network will not contribute anything useful to the quality of the output, since
their receptive field sizes war way to large.

From here on we have multiple ways of optimizing the setup.
Of course, we can simply increase the resolution, to involve more layers in the inference process, but that is usually
very expensive from a computational point of view.
If we are satisfied with the performance of the model, we may simply replace all unproductive layers with a simple output
head and save a lot of computation time.
On the other hand we could fiddle around with he downsampling layers and kernel sizes of the convolutional layers down,
such that only the final 1 or 2 layers have a receptive field that can grasp the entire image.
This is likely to be the more computational expensive choice, but it utilize the parameters in the model better, likely
resulting in better predictive performance.

If you want to see a deeper dive into these optimization strategies using receptive field analysis,
I recommend you reading [Should You Go Deeper? Optimizing Convolutional Neural Network Architectures without Training](https://arxiv.org/abs/2106.12307)

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

This package was created with
[Cookiecutter](https://github.com/audreyr/cookiecutter) and the
[browniebroke/cookiecutter-pypackage](https://github.com/browniebroke/cookiecutter-pypackage)
project template.
