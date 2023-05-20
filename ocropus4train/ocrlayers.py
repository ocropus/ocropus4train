import os

import torch, torch.jit
from torch import nn

default_device = torch.device(os.environ.get("device", "cuda:0"))
noutput = 53


class CTCRecognizer(nn.Module):
    def __init__(self, model, **kw):
        super().__init__()
        self.meta = dict(__kind__="CTCRecognizer", **kw)
        self.model = model
        self.channels = kw.get("channels", 1)
        self.usm = kw.get("usm", 16.0)

    def forward(self, x):
        assert x.dtype in [torch.float16, torch.float32, torch.float64]
        assert x.ndim == 4
        assert x.shape[1] == self.channels
        assert x.abs().max() <= 200.0
        if self.usm > 0:
            assert x.amin() < 0.0
        else:
            assert x.amin() >= 0.0 and x.amax() <= 1.0
        return self.model(x)


class TransformerRecognizer(nn.Module):
    def __init__(self, model, **kw):
        super().__init__()
        self.meta = dict(__kind__="CTCRecognizer", **kw)
        self.model = model
        self.channels = kw.get("channels", 1)
        self.usm = kw.get("usm", 16.0)

    def forward(self, x, y=None):
        assert x.dtype in [torch.float16, torch.float32, torch.float64]
        assert x.ndim == 4
        assert x.shape[1] == self.channels
        assert x.abs().max() <= 200.0
        if self.usm > 0:
            assert x.amin() < 0.0
        else:
            assert x.amin() >= 0.0 and x.amax() <= 1.0
        return self.model(x, y)


class PixSegmenter(nn.Module):
    def __init__(self, model, **kw):
        super().__init__()
        self.meta = dict(__kind__="PixSegmenter", **kw)
        self.model = model
        self.channels = kw.get("channels", 1)
        self.usm = kw.get("usm", 16.0)

    def forward(self, x):
        assert x.dtype in [torch.float16, torch.float32, torch.float64]
        assert x.ndim == 4
        assert x.shape[1] == self.channels
        assert x.abs().max() <= 200.0
        if self.usm > 0:
            assert x.amin() < 0.0
        else:
            assert x.amin() >= 0.0 and x.amax() <= 1.0
        return self.model(x)

class MultiPixSegmenter(nn.Module):
    def __init__(self, model, **kw):
        super().__init__()
        self.meta = dict(__kind__="PixSegmenter", **kw)
        self.model = model
        self.channels = kw.get("channels", 1)
        self.usm = kw.get("usm", 16.0)

    def forward(self, x):
        assert x.dtype in [torch.float16, torch.float32, torch.float64]
        assert x.ndim == 4
        assert x.shape[1] == self.channels
        assert x.abs().max() <= 200.0
        if self.usm > 0:
            assert x.amin() < 0.0
        else:
            assert x.amin() >= 0.0 and x.amax() <= 1.0
        return self.model(x)

class RectLinearizer(nn.Module):
    def __init__(self, model, **kw):
        super().__init__()
        self.meta = dict(__kind__="RectLinearizer", **kw)
        self.model = model
        self.channels = kw.get("channels", 1)
        self.usm = kw.get("usm", 16.0)

    def forward(self, x, y=None):
        assert x.dtype in [torch.float16, torch.float32, torch.float64]
        assert x.ndim == 4
        assert x.shape[1] == self.channels
        assert x.abs().max() <= 200.0
        if self.usm > 0:
            assert x.amin() < 0.0
        else:
            assert x.amin() >= 0.0 and x.amax() <= 1.0
        return self.model(x, y)


class PageTypeClassifier(nn.Module):
    def __init__(self, model, **kw):
        super().__init__()
        self.meta = dict(__kind__="PageTypeClassifier", **kw)
        self.model = model
        self.channels = kw.get("channels", 1)
        self.usm = kw.get("usm", 16.0)

    def forward(self, x, y=None):
        assert x.dtype in [torch.float16, torch.float32, torch.float64]
        assert x.ndim == 4
        assert x.shape[1] == self.channels
        assert x.abs().max() <= 200.0
        if self.usm > 0:
            assert x.amin() < 0.0
        else:
            assert x.amin() >= 0.0 and x.amax() <= 1.0
        return self.model(x, y)


class PageScaleDetector(nn.Module):
    def __init__(self, model, lo=1.0, hi=500.0, **kw):
        super().__init__()
        self.meta = dict(__kind__="PageScaleDetector", lo=lo, hi=hi, **kw)
        self.model = model
        self.channels = kw.get("channels", 1)
        self.usm = kw.get("usm", 16.0)

    def forward(self, x, y=None):
        assert x.dtype in [torch.float16, torch.float32, torch.float64]
        assert x.ndim == 4
        assert x.shape[1] == self.channels
        assert x.abs().max() <= 200.0
        if self.usm > 0:
            assert x.amin() < 0.0
        else:
            assert x.amin() >= 0.0 and x.amax() <= 1.0
        return self.model(x, y)


class PageSkewDetector(nn.Module):
    def __init__(self, model, lo=-5.0, hi=5.0, binarized=False, **kw):
        super().__init__()
        self.meta = dict(
            __kind__="PageSkewDetector", lo=lo, hi=hi, binarized=binarized, **kw
        )
        self.model = model
        self.channels = kw.get("channels", 1)
        self.usm = kw.get("usm", 16.0)

    def forward(self, x, y=None):
        assert x.dtype in [torch.float16, torch.float32, torch.float64]
        assert x.ndim == 4
        assert x.shape[1] == self.channels
        assert x.abs().max() <= 200.0
        if self.usm > 0:
            assert x.amin() < 0.0
        else:
            assert x.amin() >= 0.0 and x.amax() <= 1.0
        return self.model(x, y)


class PagePreprocessor(nn.Module):
    def __init__(self, model, binarizer=False, upscaler=False, **kw):
        super().__init__()
        self.meta = dict(
            __kind__="PagePreprocessor",
            usm=usm,
            binarizer=binarizer,
            upscaler=upscaler,
            **kw
        )
        self.channels = kw.get("channels", 1)
        self.usm = kw.get("usm", 16.0)

    def forward(self, x):
        assert x.dtype in [torch.float16, torch.float32, torch.float64]
        assert x.ndim == 4
        assert x.shape[1] == self.channels
        assert x.abs().max() <= 200.0
        if self.usm > 0:
            assert x.amin() < 0.0
        else:
            assert x.amin() >= 0.0 and x.amax() <= 1.0
        return self.model(x)
    
class AutoDevice(torch.nn.Module):
    def __init__(self, module, name=None):
        super().__init__()
        self.module = module
        self.device = None
        self.name = name

    def to(self, device):
        super().to(device)
        self.device = device
        self.module.to(device)
        return self

    def forward(self, x):
        if self.device is None:
            return self.module(x)
        else:
            return self.module(x.to(self.device)).to(x.device)
