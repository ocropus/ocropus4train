import torch
from torch import nn
from torchmore import flex, layers, combos
import torch.nn.functional as F
import os
import sys
import glob
import re
import numpy as np
import kornia
import torch.jit

default_device = torch.device(os.environ.get("device", "cuda:0"))
noutput = 53

def make(name, *args, device=default_device, **kw):
    f = eval("make_"+name)
    # model = f(*args, **kw)
    model = call_function_with_valid_kwargs(f, *args, kwargs_dict=kw)
    if device is not None:
        model.to(device)
    model.model_name = name
    return model

def extract_save_info(fname):
    fname = re.sub(r'.*/', '', fname)
    match = re.search(r'([0-9]{3})+-([0-9]{9})', fname)
    if match:
        return int(match.group(1)), float(match.group(2))*1e-6
    else:
        return 0, -1

def load_latest(model, pattern=None, error=False):
    if pattern is None:
        name = model.model_name
        pattern = f"models/{name}-*.pth"
    saves = sorted(glob.glob(pattern))
    if error:
        assert len(saves)>0, f"no {pattern} found"
    elif len(saves)==0:
        print(f"no {pattern} found", file=sys.stderr)
        return 0, -1
    else:
        print(f"loading {saves[-1]}", file=sys.stderr)
        model.load_state_dict(torch.load(saves[-1]))
        return extract_save_info(saves[-1])
    
def call_function_with_valid_kwargs(func, *args, kwargs_dict):
    # Get the valid keyword arguments for the target function
    valid_kwargs = set(func.__code__.co_varnames[:func.__code__.co_argcount])

    # Initialize dictionaries for valid and invalid keyword arguments
    valid_kwargs_dict = {}
    invalid_kwargs_dict = {}

    # Iterate over the keyword arguments and separate them into valid and invalid dictionaries
    for key, value in kwargs_dict.items():
        if key in valid_kwargs:
            valid_kwargs_dict[key] = value
        else:
            invalid_kwargs_dict[key] = value
            print(f"warning: {key} is not a valid argument for {func.__name__} function.")

    # Call the target function with the valid keyword arguments
    result = func(**valid_kwargs_dict)

    # Return the result and the invalid keyword arguments dictionary
    return result


#
# New Layers
#


class UnsharpMask(nn.Module):
    def __init__(self, sigma: float):
        super().__init__()
        self.sigma = sigma

    @torch.jit.export
    def ensure_size(self, a: torch.Tensor, size: int) -> torch.Tensor:
        assert a.ndim == 4
        height, width = a.shape[-2:]
        pad_height = max(0, size - height)
        pad_width = max(0, size - width)
        padded_tensor = F.pad(a, (0, pad_width, 0, pad_height), mode='constant', value=0.0)
        return padded_tensor

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        size = int(4*self.sigma) + 1
        a = self.ensure_size(a, size+1)
        return a - kornia.filters.gaussian_blur2d(a, (size, size), (self.sigma, self.sigma))
    
    def extra_repr(self):
        return f"sigma={self.sigma}"



class HeightTo(nn.Module):
    """Ensure that the input height is equal to the given height."""
    def __init__(self, height, upscale=True):
        super(HeightTo, self).__init__()
        self.height = height
        self.upscale = upscale

    def __repr__(self):
        return f"HeightTo({self.height})"

    def __str__(self):
        return repr(self)

    def forward(self, a):
        assert a.ndim == 4
        zoom = float(self.height) / float(a.shape[2])
        if zoom < 1.0 or (zoom > 1.0 and self.upscale):
            result = F.interpolate(a, scale_factor=zoom, recompute_scale_factor=False)        
            return result
        else:
            return a
        

class CenterNormalize(nn.Module):
    def __init__(self, target=48):
        super(CenterNormalize, self).__init__()
        self.target = target

    def forward(self, a):
        # a is a batch of (b, c, h, w) images
        for i in range(a.shape[0]):
            img = a[i]
            img = img - img.min()
            img = img / img.max()
            if img.mean() > 0.5:
                img = 1.0 - img
            proj = img.sum(2).sum(0)
            # compute the center of mass and stddev of proj
            mean = (proj * torch.arange(proj.shape[0], device=proj.device)).sum() / proj.sum()
            std = torch.sqrt((proj * (torch.arange(proj.shape[0], device=proj.device) - mean)**2).sum() / proj.sum())
            raise Exception("unimplemeted FIXME")

class GrayDocument(nn.Module):
    """Ensure that the output is a single channel image.

    Images are normalized and a small amount of noise is added."""

    def __init__(self, noise=0.0, autoinvert=True):
        super(GrayDocument, self).__init__()
        self.noise = noise
        self.autoinvert = autoinvert
        self.val = nn.Parameter(torch.zeros(1))

    def __repr__(self):
        return f"GrayDocument(noise={self.noise}, autoinvert={self.autoinvert}, device={self.val.device})"

    def __str__(self):
        return repr(self)

    def forward(self, a):
        assert a.ndim == 3 or a.ndim == 4
        assert isinstance(a, torch.Tensor)
        if a.dtype == torch.uint8:
            a = a.float() / 255.0
        if a.ndim == 4:
            a = torch.mean(a, 1)
        if a.ndim == 3:
            a = a.unsqueeze(1)
        for i in range(a.shape[0]):
            a[i] -= a[i].min().item()
            a[i] /= max(0.5, a[i].max().item())
            if self.autoinvert and a[i].mean().item() > 0.5:
                a[i] = 1.0 - a[i]
            if self.noise > 0:
                d, h, w = a[i].shape
                a[i] += self.noise * torch.randn(d, h, w, device=a.device)
            a[i] = a[i].clip(0, 1)
        a = a.to(self.val.device)
        return a


################################################################
# ## layer combinations
# ###############################################################

#ocr_output = "BLD"
ocr_output = "BDL"

def project_and_lstm(d, noutput, num_layers=1):
    return [
        layers.Fun("lambda x: x.sum(2)"), # BDHW -> BDW
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(d, bidirectional=True, num_layers=num_layers),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
        layers.Reorder("BDL", ocr_output)
    ]

def project_and_conv1d(d, noutput, r=5):
    return [
        layers.Fun("lambda x: x.max(2)[0]"),
        flex.Conv1d(d, r),
        flex.BatchNorm1d(),
        nn.ReLU(),
        flex.Conv1d(noutput, 1),
        layers.Reorder("BDL", ocr_output)
    ]


################################################################
### entire OCR models
################################################################

def make_conv_only(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(100, 3, mp=2, repeat=2),
        *combos.conv2d_block(200, 3, mp=2, repeat=2),
        *combos.conv2d_block(300, 3, mp=2, repeat=2),
        *combos.conv2d_block(400, 3, repeat=2),
        *project_and_conv1d(800, noutput)
    )
    flex.shape_inference(model, (1, 1, 48, 300))
    return model

def make_conv_resnet(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(64, 3, mp=2),
        *combos.resnet_blocks(5, 64),
        *combos.conv2d_block(128, 3, mp=(2, 1)),
        *combos.resnet_blocks(5, 128),
        *combos.conv2d_block(192, 3, mp=2),
        *combos.resnet_blocks(5, 192),
        *combos.conv2d_block(256, 3, mp=(2, 1)),
        *combos.resnet_blocks(5, 256),
        *combos.conv2d_block(512, 3),
        *project_and_conv1d(512, noutput)
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model

def make_lstm_ctc(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(50, 3, mp=(2, 1)),
        *combos.conv2d_block(100, 3, mp=(2, 1)),
        *combos.conv2d_block(150, 3, mp=2),
        *project_and_lstm(100, noutput)
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model

def make_lstm_normalized(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1),
                     sizes=[None, 1, 80, None]),
        *combos.conv2d_block(50, 3, mp=(2, 1)),
        *combos.conv2d_block(100, 3, mp=(2, 1)),
        *combos.conv2d_block(150, 3, mp=2),
        layers.Reshape(0, [1, 2], 3),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(100, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
        layers.Reorder("BDL", ocr_output))
    flex.shape_inference(model, (1, 1, 80, 200))
    return model

def make_lstm_transpose(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(50, 3, repeat=2),
        *combos.conv2d_block(100, 3, repeat=2),
        *combos.conv2d_block(150, 3, repeat=2),
        *combos.conv2d_block(200, 3, repeat=2),
        layers.Fun("lambda x: x.sum(2)"), # BDHW -> BDW
        flex.ConvTranspose1d(800, 1, stride=2), # <-- undo too tight spacing
        #flex.BatchNorm1d(), nn.ReLU(),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(100, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
        layers.Reorder("BDL", ocr_output)
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model

def make_lstm_keep(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        layers.KeepSize(
            mode="nearest",
            dims=[3],
            sub=nn.Sequential(
                *combos.conv2d_block(50, 3, repeat=2),
                *combos.conv2d_block(100, 3, repeat=2),
                *combos.conv2d_block(150, 3, repeat=2),
                layers.Fun("lambda x: x.sum(2)") # BDHW -> BDW
            )
        ),
        flex.Conv1d(500, 5, padding=2),
        flex.BatchNorm1d(),
        nn.ReLU(),
        layers.Reorder("BDL", "LBD"),
        flex.LSTM(200, bidirectional=True),
        layers.Reorder("LBD", "BDL"),
        flex.Conv1d(noutput, 1),
        layers.Reorder("BDL", ocr_output)
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model

def make_lstm_resnet(noutput=noutput, blocksize=5):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(64, 3, mp=(2, 1)),
        *combos.resnet_blocks(blocksize, 64),
        *combos.conv2d_block(128, 3, mp=(2, 1)),
        *combos.resnet_blocks(blocksize, 128),
        *combos.conv2d_block(256, 3, mp=2),
        *combos.resnet_blocks(blocksize, 256),
        *combos.conv2d_block(256, 3),
        *project_and_lstm(100, noutput)
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model

def make_lstm_resnet_f(noutput=noutput, blocksize=5):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        UnsharpMask(16.0),
        *combos.conv2d_block(64, 3, mp=(2, 1)),
        *combos.resnet_blocks(blocksize, 64),
        *combos.conv2d_block(128, 3, mp=(2, 1)),
        *combos.resnet_blocks(blocksize, 128),
        *combos.conv2d_block(256, 3, mp=2),
        *combos.resnet_blocks(blocksize, 256),
        *combos.conv2d_block(256, 3),
        *project_and_lstm(100, noutput)
    )
    flex.shape_inference(model, (1, 1, 128, 512))
    return model

def make_lstm_unet(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(64, 3, repeat=3),
        combos.make_unet([64, 128, 256, 512]),
        *combos.conv2d_block(128, 3, repeat=2),
        *project_and_lstm(100, noutput)
    )
    flex.shape_inference(model, (1, 1, 128, 256))
    return model

def make_lstm2_ctc(noutput=noutput):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(100, 3, mp=2, repeat=2),
        *combos.conv2d_block(200, 3, mp=2, repeat=2),
        *combos.conv2d_block(300, 3, mp=2, repeat=2),
        *combos.conv2d_block(400, 3, repeat=2),
        flex.Lstm2(400),
        *project_and_conv1d(800, noutput)
    )
    flex.shape_inference(model, (1, 1, 48, 300))
    return model

def make_lstm2_ctc_v2(noutput=noutput):
    model = nn.Sequential(
        GrayDocument(autoinvert=True),
        HeightTo(64, upscale=False),
        *combos.conv2d_block(100, 3, mp=2, repeat=2),
        *combos.conv2d_block(200, 3, mp=2, repeat=2),
        *combos.conv2d_block(300, 3, mp=2, repeat=2),
        *combos.conv2d_block(400, 3, repeat=2),
        flex.Lstm2(400),
        *project_and_conv1d(800, noutput)
    )
    flex.shape_inference(model, (1, 1, 48, 300))
    return model

def make_seg_conv(noutput=4):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        layers.KeepSize(sub=nn.Sequential(
                *combos.conv2d_block(50, 3, mp=2, repeat=3),
                *combos.conv2d_block(100, 3, mp=2, repeat=3),
                *combos.conv2d_block(200, 3, mp=2, repeat=3)
            )
        ),
        *combos.conv2d_block(400, 5),
        flex.Conv2d(noutput, 3)
    )
    flex.shape_inference(model, (1, 1, 256, 256))
    return model

def make_seg_lstm(noutput=4):
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        layers.KeepSize(sub=nn.Sequential(
                *combos.conv2d_block(50, 3, mp=2, repeat=3),
                *combos.conv2d_block(100, 3, mp=2, repeat=3),
                *combos.conv2d_block(200, 3, mp=2, repeat=3),
                flex.BDHW_LSTM(200)
            )
        ),
        *combos.conv2d_block(400, 5),
        flex.Conv2d(noutput, 3)
    )
    flex.shape_inference(model, (1, 1, 256, 256))
    return model

def make_seg_unet(noutput=4, dropout=0.0, levels=7):
    size = [int(64*(2.0**x)) for x in np.linspace(0, 3, levels)]
    model = nn.Sequential(
        layers.Input("BDHW", range=(0, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(64, 3, repeat=3),
        combos.make_unet(size),
        *combos.conv2d_block(64, 3, repeat=2),
        flex.Conv2d(noutput, 5)
    )
    flex.shape_inference(model, (1, 1, 256, 256))
    return model

def make_seg_unet2(noutput=4, dropout=0.0, levels=5, complexity=64, final=4):
    size = [int(complexity*(2.0**x)) for x in np.linspace(0, 3, levels)]
    model = nn.Sequential(
        layers.Input("BDHW", range=(-1, 1), sizes=[None, 1, None, None]),
        *combos.conv2d_block(64, 3, repeat=3),
        combos.make_unet(size, sub=flex.BDHW_LSTM(size[-1])),
        *combos.conv2d_block(64, 3, repeat=2),
        flex.BDHW_LSTM(final),
        # *combos.conv2d_block(64, 3, repeat=2),
        flex.Conv2d(noutput, 5)
    )
    flex.shape_inference(model, (1, 1, 256, 256))
    return model

def make_seg_unet2f(noutput=4, dropout=0.0, levels=5, complexity=64, final=4):
    size = [int(complexity*(2.0**x)) for x in np.linspace(0, 3, levels)]
    model = nn.Sequential(
        layers.Input("BDHW", range=(-1, 1), sizes=[None, 1, None, None]),
        UnsharpMask(16.0),
        *combos.conv2d_block(64, 3, repeat=3),
        combos.make_unet(size, sub=flex.BDHW_LSTM(size[-1])),
        *combos.conv2d_block(64, 3, repeat=2),
        flex.BDHW_LSTM(final),
        # *combos.conv2d_block(64, 3, repeat=2),
        flex.Conv2d(noutput, 5)
    )
    flex.shape_inference(model, (1, 1, 256, 256))
    return model

