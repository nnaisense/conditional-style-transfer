## Known issues

##### Trying to run training script, I get: "... RuntimeError: one of the variables needed for gradient computation has been modified by an inplace operation ..."

We have encountered this issue after upgrading PyTorch. This work has been done with PyTorch 1.4.0. Currently we are running on PyTorch 1.6.0. However, trying to train a model we get the above error. Downgrading to PyTorch 1.4.0 resolvs the issue.

##### Trying to run training script, I get: "ConnectionError: Error connecting to Visdom server"

We use [Visdom](https://github.com/facebookresearch/visdom) to display some examples and statistics during training. In order to run our training script, you therefore need to install Visdom (typically `pip install visdom`) and then run a visdom server by running `visdom` command.