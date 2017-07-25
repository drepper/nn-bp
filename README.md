Demonstration of Neural Network Learning
========================================

Visual representation of the internals of a neural network during the
learning phase.  The code needs to be recompiled for different
configurations.  The input has two dimensions and the value is
categorical with two or more classes.  Best to pick two classes or three
at most since otherwise the coloring will not work well.

Compile with `make` to get the program `nn-bp-gtkmm`.  This uses GTK 3.  If
you do not have it installed the code will not compile.  Similarly the Eigen3
library is needed.

Once started the fresh, randomized neural network is shown.

![Startup](/startup.png)

In this case three classes are used with overlapping areas.  This is a hard
problem to solve.

The two small images on the right side represent the first (and only hidden) layer
and the last layer (the output) of the network.  By clicking on the little image
the big display can be changed to that layer.

Below the image the geometry of the network is shown with the currently displayed
layer highlighted in orange.

The buttons on the lower right side allow to advance the training.  The current
Epoch number is shown and can be advanced by 1, 10, or 100.  After every button
press the graphics are updated to represent the current state.  After 500 Epochs
the result could look like this:

![Epoch 500](/epoch500.png)

Three outputs are generated corresponding to the number of classes.  The plus and
circle classes are recognized nicely when they do not overlap with other classes.
The cross class is reconized only in the bottom part, not in the top part.  This
is a limitation of the geometry of the neural network.  By playing with the
parameters (e.g., large layers or more layers) and looking at the internal layers
it is possible to get an understanding for what is going on in a neural
network.
