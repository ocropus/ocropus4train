import glob, os, re, sys

import numpy as np
import torch
import torch.jit
from einops.layers.torch import Rearrange
from torch import nn
from torchmore2 import combos, flex

from . import ocrmodels as ocrmodels_old
from . import ocrlayers

default_device = torch.device(os.environ.get("device", "cuda:0"))
noutput = 53


def find_constructor(name, module):
    fname = "make_" + name
    if hasattr(module, fname):
        return getattr(module, fname)
    else:
        return None


def make(name, *args, device=default_device, **kw):
    f = find_constructor(name, sys.modules[__name__]) or find_constructor(
        name, ocrmodels_old
    )
    # model = f(*args, **kw)
    model = call_function_with_valid_kwargs(f, *args, kwargs_dict=kw)
    if device is not None:
        model.to(device)
    model.model_name = name
    return model


def extract_save_info(fname):
    fname = re.sub(r".*/", "", fname)
    match = re.search(r"([0-9]{3})+-([0-9]{9})", fname)
    if match:
        return int(match.group(1)), float(match.group(2)) * 1e-6
    else:
        return 0, -1


def load_latest(model, pattern=None, error=False):
    if pattern is None:
        name = model.model_name
        pattern = f"models/{name}-*.pth"
    saves = sorted(glob.glob(pattern))
    if error:
        assert len(saves) > 0, f"no {pattern} found"
    elif len(saves) == 0:
        print(f"no {pattern} found", file=sys.stderr)
        return 0, -1
    else:
        print(f"loading {saves[-1]}", file=sys.stderr)
        model.load_state_dict(torch.load(saves[-1]))
        return extract_save_info(saves[-1])


def call_function_with_valid_kwargs(func, *args, kwargs_dict):
    # Get the valid keyword arguments for the target function
    valid_kwargs = set(func.__code__.co_varnames[: func.__code__.co_argcount])

    # Initialize dictionaries for valid and invalid keyword arguments
    valid_kwargs_dict = {}
    invalid_kwargs_dict = {}

    # Iterate over the keyword arguments and separate them into valid and invalid dictionaries
    for key, value in kwargs_dict.items():
        if key in valid_kwargs:
            valid_kwargs_dict[key] = value
        else:
            invalid_kwargs_dict[key] = value
            print(
                f"warning: {key} is not a valid argument for {func.__name__} function."
            )

    # Call the target function with the valid keyword arguments
    result = func(**valid_kwargs_dict)

    # Return the result and the invalid keyword arguments dictionary
    return result


class SumReduce(nn.Module):
    def __init__(self, axis=2):
        super(SumReduce, self).__init__()
        self.axis = axis

    def forward(self, a):
        return a.sum(self.axis)


class MaxReduce(nn.Module):
    def __init__(self, axis=2):
        super(MaxReduce, self).__init__()
        self.axis = axis

    def forward(self, a):
        return a.max(self.axis)[0]


class TextInput(nn.Module):
    def __init__(self):
        super(TextInput, self).__init__()

    def forward(self, x):
        # check image dimensions
        assert x.ndim == 4, x.shape
        assert x.shape[1] == 1, x.shape
        assert x.shape[-2] <= 48, x.shape
        # check that the input is USM preprocessed
        assert x.amin() < -0.0001, (x.amin(), x.mean(), x.std(), x.amax())
        assert abs(float(x.mean())) < 0.2, (x.amin(), x.mean(), x.std(), x.amax())
        return x


class SegInput(nn.Module):
    def __init__(self):
        super(SegInput, self).__init__()

    def forward(self, x):
        # check image dimensions
        assert x.ndim == 4, x.shape
        assert x.shape[1] == 1, x.shape
        # check that the input is USM preprocessed
        assert x.amin() < -0.0001, (x.amin(), x.mean(), x.std(), x.amax())
        assert abs(float(x.mean())) < 0.2, (x.amin(), x.mean(), x.std(), x.amax())
        return x


def project_and_lstm_v2(d, noutput, num_layers=1):
    return [
        SumReduce(2),
        Rearrange("b d l -> l b d"),
        flex.LSTM(d, bidirectional=True, num_layers=num_layers),
        Rearrange("l b d -> b d l"),
        flex.Conv1d(noutput, 1),
    ]


def project_and_conv1d_v2(d, noutput, r=5):
    return [
        MaxReduce(2),
        flex.Conv1d(d, r),
        flex.BatchNorm1d(),
        nn.ReLU(),
        flex.Conv1d(noutput, 1),
    ]


def make_lstm_resnet_v2(noutput=noutput, blocksize=5):
    model = nn.Sequential(
        TextInput(),
        *combos.conv2d_block(64, 3, mp=(2, 1)),
        *combos.resnet_blocks(blocksize, 64),
        *combos.conv2d_block(128, 3, mp=(2, 1)),
        *combos.resnet_blocks(blocksize, 128),
        *combos.conv2d_block(256, 3, mp=2),
        *combos.resnet_blocks(blocksize, 256),
        *combos.conv2d_block(256, 3),
        *project_and_lstm_v2(100, noutput),
    )
    flex.shape_inference(model, torch.randn((2, 1, 48, 512)))
    return model


def make_seg_unet_v2(noutput=4, dropout=0.0, levels=5, complexity=64, final=4):
    size = [int(complexity * (2.0**x)) for x in np.linspace(0, 3, levels)]
    model = nn.Sequential(
        SegInput(),
        *combos.conv2d_block(64, 3, repeat=3),
        combos.make_unet(size, sub=flex.BDHW_LSTM(size[-1])),
        *combos.conv2d_block(64, 3, repeat=2),
        flex.BDHW_LSTM(final),
        # *combos.conv2d_block(64, 3, repeat=2),
        flex.Conv2d(noutput, 5),
    )
    flex.shape_inference(model, torch.randn((2, 1, 256, 256)))
    return model


def make_lstm_resnet_v3(noutput=noutput, blocksize=5, kinds="unknown"):
    model = nn.Sequential(
        TextInput(),
        *combos.conv2d_block(64, 3, mp=(2, 1)),
        *combos.resnet_blocks(blocksize, 64),
        *combos.conv2d_block(128, 3, mp=(2, 1)),
        *combos.resnet_blocks(blocksize, 128),
        *combos.conv2d_block(256, 3, mp=2),
        *combos.resnet_blocks(blocksize, 256),
        *combos.conv2d_block(256, 3),
        *project_and_lstm_v2(100, noutput),
    )
    model = ocrlayers.CTCRecognizer(model, noutput=noutput, kinds=kinds)
    flex.shape_inference(model, torch.randn((2, 1, 48, 512)))
    return model


def make_seg_unet_v3(noutput=4, dropout=0.0, levels=5, complexity=64, final=4, kinds="unknown"):
    size = [int(complexity * (2.0**x)) for x in np.linspace(0, 3, levels)]
    model = nn.Sequential(
        SegInput(),
        *combos.conv2d_block(64, 3, repeat=3),
        combos.make_unet(size, sub=flex.BDHW_LSTM(size[-1])),
        *combos.conv2d_block(64, 3, repeat=2),
        flex.BDHW_LSTM(final),
        # *combos.conv2d_block(64, 3, repeat=2),
        flex.Conv2d(noutput, 5),
    )
    model = ocrlayers.PixSegmenter(model, noutput=noutput, kinds=kinds)
    flex.shape_inference(model, torch.randn((2, 1, 256, 256)))
    return model
