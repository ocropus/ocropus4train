import glob, os, re, sys

import numpy as np
import torch
import torch.jit
from einops.layers.torch import Rearrange
from einops import rearrange
from torch import nn
from torchmore2 import combos, flex
import torch.nn.functional as F

from . import ocrmodels as ocrmodels_old
from . import ocrlayers

default_device = torch.device(os.environ.get("device", "cuda:0"))
noutput = 53

def reorder(x: torch.Tensor, old: str, new: str, set_order: bool = True) -> torch.Tensor:
    """Reorder dimensions according to strings.
    E.g., reorder(x, "BLD", "LBD")
    """
    assert isinstance(old, str) and isinstance(new, str)
    for c in old:
        assert new.find(c) >= 0
    for c in new:
        assert old.find(c) >= 0
    permutation = [old.find(c) for c in new]
    assert len(old) == x.ndim, (old, x.size())
    result = x.permute(permutation).contiguous()
    return result


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

class Sum2(nn.Module):
  def forward(self, x):
    return x.sum(2)

import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

def preencoder_v1(nembed):
    model = nn.Sequential(
        *combos.conv2d_block(50, 3, mp=(2, 1)),
        *combos.conv2d_block(100, 3, mp=(2, 1)),
        *combos.conv2d_block(150, 3, mp=2),
        Sum2(),
        Rearrange("B D L -> L B D"),
        flex.LSTM(100, bidirectional=True, num_layers=1),
        Rearrange("L B D -> B D L"),
        flex.Conv1d(nembed, 1),
    )
    flex.shape_inference(model, torch.randn((2, 1, 256, 256)))
    return model
    
class OcroTrans(nn.Module):

  def __init__(self, preencoder=None, ncodes=96, nembed=256, nseq=256, nhead=4, nelayers=2, ndlayers=2):
    super().__init__()
    self.ncodes = ncodes
    self.pe = PositionalEncoding(nembed)
    self.preencoder = preencoder or preencoder_v1(nseq)
    self.lin = nn.Linear(nseq, nembed)
    self.embed = nn.Linear(ncodes, nembed)
    self.tr = nn.Transformer(
        d_model=nembed,
        nhead=nhead,
        num_encoder_layers=nelayers,
        num_decoder_layers=ndlayers,
        dim_feedforward=4*nembed,
    )
    self.decode = nn.Linear(nembed, ncodes)

  def forward(self, x, y):
    assert x.ndim == 4  # b d h w
    assert y.ndim == 2  # b l
    assert y.amax() < self.ncodes and y.amin() >= 0

    m = self.preencoder(x)  # -> b d l
    # m = Rearrange("b d l -> l b d")(m)
    m = reorder(m, "BDL", "LBD")
    m = self.lin(m)
    self.pe(m)

    start = torch.zeros((y.shape[0], 1), dtype=torch.long, device=y.device)
    y = torch.cat([start, y], dim=1)
    y = F.one_hot(y, self.ncodes).float()
    y = self.embed(y)
    # y = Rearrange("b l d -> l b d")(y)
    y = reorder(y, "BLD", "LBD")
    y = self.pe(y)

    # mask = nn.Transformer.generate_square_subsequent_mask(y.shape[0], device=y.device)
    sz = y.shape[0]
    mask = torch.triu(torch.full((sz, sz), float('-inf'), device=y.device), diagonal=1)
    z = self.tr.forward(m, y, tgt_mask=mask)
    z = self.decode(z)

    # return Rearrange("l b d -> b d l")(z[:-1])
    return reorder(z[:-1], "LBD", "BDL")

def make_tf_v1(noutput=noutput, blocksize=5):
    model = OcroTrans()
    model = ocrlayers.TransformerRecognizer(model, noutput=noutput)
    return model