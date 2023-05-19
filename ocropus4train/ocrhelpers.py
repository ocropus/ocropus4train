import glob, os, sys, time
from functools import wraps

import editdistance
import matplotlib.pyplot as plt
import numpy as np
import torch
from numpy import *
from scipy import ndimage as ndi
from torch import nn, optim
from torchmore2 import layers
import torch.nn.functional as F

plt.rc("image", cmap="gray")
plt.rc("image", interpolation="nearest")
import matplotlib.pyplot as plt
import scipy.ndimage as ndi


def extend_to(l, n):
    if len(l) >= n:
        return l
    return l + [l[-1]] * (n - len(l))


class NanError(Exception):
    pass


class NanChecker:
    def __call__(self, module, input):
        if torch.isnan(input[0]).any():
            raise NanError("NaN in forward pass")


def add_nan_checker(model):
    nan_checker = NanChecker()
    model.register_forward_pre_hook(nan_checker)


def plotting_inside_notebook() -> bool:
    """
    Returns True if Matplotlib is configured to display plots inline in a Jupyter notebook,
    False otherwise (i.e., if plots are being displayed in a separate window).
    """
    # Check if running inside a Jupyter notebook
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            # Check if the matplotlib backend is set to inline
            if "inline" in plt.get_backend():
                return True
            else:
                return False
        else:
            return False
    except NameError:
        # get_ipython() function not defined, not running in a notebook
        return False


def latest_file(pattern):
    files = glob.glob(pattern)
    if not files:
        return None
    latest_file = max(files, key=os.path.getmtime)
    return latest_file


class DefaultCharset:
    def __init__(self, chars="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"):
        if isinstance(chars, str):
            chars = list(chars)
        self.chars = [""] + chars

    def __len__(self):
        return len(self.chars)

    def encode_char(self, c):
        try:
            index = self.chars.index(c)
        except ValueError:
            index = len(self.chars) - 1
        return max(index, 1)

    def encode(self, s):
        assert isinstance(s, str)
        return [self.encode_char(c) for c in s]

    def decode(self, l):
        assert isinstance(l, list)
        return "".join([self.chars[k] for k in l])


def RUN(x):
    """Run a command and output the result."""
    print(x, ":", os.popen(x).read().strip())


def scale_to(a, shape):
    """Scale a numpy array to a given target size."""
    scales = array(a.shape, "f") / array(shape, "f")
    result = ndi.affine_transform(a, diag(scales), output_shape=shape, order=1)
    return result


def tshow(a, order, b=0, ax=None, **kw):
    """Display a torch array with imshow."""
    from matplotlib.pyplot import gca

    ax = ax or gca()
    if set(order) == set("BHWD"):
        a = layers.reorder(a.detach().cpu(), order, "BHWD")[b].numpy()
    elif set(order) == set("HWD"):
        a = layers.reorder(a.detach().cpu(), order, "HWD").numpy()
    elif set(order) == set("HW"):
        a = layers.reorder(a.detach().cpu(), order, "HW").numpy()
    else:
        raise ValueError(f"{order}: unknown order")
    if a.shape[-1] == 1:
        a = a[..., 0]
    ax.imshow(a, **kw)


def asnp(a):
    """Convert to numpy."""
    if isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    else:
        assert isinstance(a, np.ndarray)
        return a


def method(cls):
    """A decorator allowing methods to be added to classes."""

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        setattr(cls, func.__name__, wrapper)
        return func

    return decorator


def ctc_decode(probs, sigma=1.0, threshold=0.7, kind=None, full=False):
    """A simple decoder for CTC-trained OCR recognizers.

    :probs: d x l sequence classification output
    """
    assert probs.ndim == 2, probs.shape
    probs = asnp(probs.T)
    assert (
        abs(probs.sum(1) - 1) < 1e-4
    ).all(), f"input not normalized; did you apply .softmax()? {probs.sum(1)}"
    probs = ndi.gaussian_filter(probs, (sigma, 0))
    probs /= probs.sum(1)[:, newaxis]
    labels, n = ndi.label(probs[:, 0] < threshold)
    mask = tile(labels[:, newaxis], (1, probs.shape[1]))
    mask[:, 0] = 0
    maxima = ndi.maximum_position(probs, mask, arange(1, amax(mask) + 1))
    if not full:
        return [c for r, c in sorted(maxima)]
    else:
        return [(r, c, probs[r, c]) for r, c in sorted(maxima)]


def pack_for_ctc(seqs):
    """Pack a list of sequences for nn.CTCLoss."""
    allseqs = torch.cat(seqs).long()
    alllens = torch.tensor([len(s) for s in seqs]).long()
    return (allseqs, alllens)


def collate4ocr(samples):
    """Collate image+sequence samples into batches.

    This returns an image batch and a compressed sequence batch using CTCLoss conventions.
    """
    images, seqs = zip(*samples)
    images = [im.unsqueeze(2) if im.ndimension() == 2 else im for im in images]
    w, h, d = map(max, zip(*[x.shape for x in images]))
    result = torch.zeros((len(images), w, h, d), dtype=torch.float)
    for i, im in enumerate(images):
        w, h, d = im.shape
        if im.dtype == torch.uint8:
            im = im.float() / 255.0
        result[i, :w, :h, :d] = im
    allseqs = torch.cat(seqs).long()
    alllens = torch.tensor([len(s) for s in seqs]).long()
    return (result, (allseqs, alllens))

def collate4trans(samples):
    """Collate image+sequence samples into batches.
    This returns an image batch and a compressed sequence batch using CTCLoss conventions.
    """
    images, seqs = zip(*samples)
    images = [im.unsqueeze(2) if im.ndimension() == 2 else im for im in images]
    w, h, d = map(max, zip(*[x.shape for x in images]))
    result = torch.zeros((len(images), w, h, d), dtype=torch.float)
    for i, im in enumerate(images):
        w, h, d = im.shape
        if im.dtype == torch.uint8:
            im = im.float() / 255.0
        result[i, :w, :h, :d] = im
    maxlen = max(len(s) for s in seqs)
    allseqs = torch.zeros((len(seqs), maxlen+1), dtype=torch.long)
    for i, s in enumerate(seqs):
        allseqs[i, :len(s)] = s
    return (result, allseqs)


def model_device(model):
    """Find the device of a model."""
    return next(model.parameters()).device


device = None


def get_maxcount(dflt=999999999):
    """Get maxcount from a file if available."""
    if os.path.exists("__MAXCOUNT__"):
        with open("__MAXCOUNT__") as stream:
            maxcount = int(stream.read().strip())
        print(f"__MAXCOUNT__ {maxcount}", file=sys.stderr)
    else:
        maxcount = int(os.environ.get("maxcount", dflt))
        if maxcount != dflt:
            print(f"maxcount={maxcount}", file=sys.stderr)
    return maxcount


def CTCLossBDL(log_softmax=True):
    """Compute CTC Loss on BDL-order tensors.

    This is a wrapper around nn.CTCLoss that does a few things:
    - it accepts the output as a plain tensor (without lengths)
    - it forforms a softmax
    - it accepts output tensors in BDL order (regular CTC: LBD)
    """
    ctc_loss = nn.CTCLoss()

    def lossfn(outputs, targets):
        assert isinstance(targets, tuple) and len(targets) == 2
        assert targets[0].amin() >= 1, targets
        assert targets[0].amax() < outputs.size(1), targets
        assert not torch.isnan(outputs).any()  # FIXME
        # layers.check_order(outputs, "BDL")
        b, d, l = outputs.size()
        olens = torch.full((b,), l).long()
        if log_softmax:
            outputs = outputs.log_softmax(1)
        assert not torch.isnan(outputs).any()  # FIXME
        outputs = layers.reorder(outputs, "BDL", "LBD")
        targets, tlens = targets
        assert tlens.size(0) == b
        assert tlens.sum() == targets.size(0)
        result = ctc_loss(outputs.cpu(), targets.cpu(), olens.cpu(), tlens.cpu())
        if torch.isnan(result):
            raise ValueError("NaN loss")
        return result

    return lossfn


def softmax1(x):
    """Softmax on second dimension."""
    return x.softmax(1)


class SavingForTrainer(object):
    """Saving mixin for Trainers."""

    def __init__(self):
        super().__init__()
        self.savedir = os.environ.get("savedir", "./models")
        self.loss_horizon = 100
        self.loss_scale = 1.0
        self.save_jit = True

    def save_epoch(self, epoch):
        if not hasattr(self.model, "model_name"):
            return
        if not self.savedir or self.savedir == "":
            return
        if not os.path.exists(self.savedir):
            return
        if not hasattr(self, "losses") or len(self.losses) < self.loss_horizon:
            return
        base = self.model.model_name
        ierr = int(1e6 * mean(self.losses[-self.loss_horizon :]) * self.loss_scale)
        ierr = min(999999999, ierr)
        loss = "%09d" % ierr
        epoch = "%03d" % epoch
        fname = f"{self.savedir}/{base}-{epoch}-{loss}.pth"
        print(f"saving {fname}", file=sys.stderr)
        torch.save(self.model.state_dict(), fname)
        if self.save_jit:
            jitted = torch.jit.script(self.model)
            torch.jit.save(jitted, f"{self.savedir}/{base}-{epoch}-{loss}.jit")

    def load(self, fname):
        print(f"loading {fname}", file=sys.stderr)
        self.model.load_state_dict(torch.load(fname))

    def load_best(self, key="latest"):
        import glob

        assert hasattr(self.model, "model_name")
        pattern = f"{self.savedir}/{self.model.model_name}-*.pth"
        files = glob.glob(pattern)
        if len(files) == 0:
            print("no load file found")
            return False

        def lossof(fname):
            return fname.split(".")[-2].split("-")[-1]

        def epochof(fname):
            return int(fname.split(".")[-2].split("-")[-2])

        key = epochof if key == "latest" else lossof
        files = sorted(files, key=key)
        fname = files[-1]
        print("loading", fname)
        self.load(fname)
        self.epoch = epochof(fname) + 1
        return True

class ReporterForTrainer(object):
    """Report mixin for Trainers."""

    def __init__(self):
        super().__init__()
        self.last_display = time.time() - 999999
        self.in_notebook = plotting_inside_notebook()
        self.fig = None
        self.extra = hasattr(self, "report_extra")

    def report_simple(self):
        avgloss = mean(self.losses[-100:]) if len(self.losses) > 0 else 0.0
        print(
            f"{self.epoch:3d} {self.count:9d} {avgloss:10.4f}",
            " " * 10,
            file=sys.stderr,
            end="\r",
            flush=True,
        )

    def report_end(self):
        if int(os.environ.get("noreport", 0)):
            return
        from IPython import display

        display.clear_output(wait=True)

    def report_inputs(self, ax, inputs):
        ax.set_title(f"{self.epoch} {self.count}")
        ax.imshow(inputs[0, 0].detach().cpu(), cmap="gray")

    def report_losses(self, ax, losses):
        if len(losses) < 100:
            return
        losses = ndi.gaussian_filter(losses, 10.0)
        losses = losses[::10]
        losses = ndi.gaussian_filter(losses, 10.0)
        ax.plot(losses)
        ax.set_ylim((0.9 * amin(losses), median(losses) * 3))

    def report_outputs(self, ax, outputs):
        pass

    def report(self):
        import matplotlib.pyplot as plt
        from IPython import display

        if int(os.environ.get("noreport", 0)):
            return
        if time.time() - self.last_display < self.every:
            return
        current_lr = self.optimizer.param_groups[0]["lr"]
        print(f"{self.epoch:3d} {self.count:9d} {mean(self.losses[-100:]):10.4f} (lr={current_lr:10.4f})", file=sys.stderr)
        self.last_display = time.time()
        if self.fig is None:
            self.fig = plt.figure(figsize=(10, 8))
            self.fig.show()
            if self.extra:
                for i in range(4):
                    self.fig.add_subplot(2, 2, i + 1)
            else:
                for i in range(3):
                    self.fig.add_subplot(3, 1, i + 1)
        self.axs = self.fig.get_axes()
        for ax in self.axs:
            ax.cla()
        inputs, targets, outputs = self.last_batch
        self.report_inputs(self.axs[0], inputs)
        self.report_outputs(self.axs[1], outputs)
        self.report_losses(self.axs[2], self.losses)
        if len(self.axs) == 4:
            self.report_extra(self.axs[3], inputs, targets, outputs)
        if self.in_notebook:
            display.clear_output(wait=True)
            display.display(self.fig)
        else:
            renderer = self.fig.canvas.get_renderer()
            self.fig.draw(renderer)
        plt.pause(0.01)


class BaseTrainer(ReporterForTrainer, SavingForTrainer):
    def __init__(
        self,
        model,
        *,
        lossfn=None,
        probfn=softmax1,
        lr=1e-4,
        every=3.0,
        device="cuda",
        savedir=True,
        maxgrad=10.0,
        mode=None,
        **kw,
    ):
        super().__init__()
        self.model = model.to(device)
        self.device = device
        if lossfn is None:
            if mode == "ctc":
                lossfn = CTCLossBDL()
            elif mode == "tf":
                lossfn = nn.CrossEntropyLoss()
        self.lossfn = lossfn
        self.probfn = probfn
        self.every = every
        self.losses = []
        self.last_lr = None
        self.set_lr(lr)
        self.clip_gradient = maxgrad
        self.charset = None
        self.maxcount = get_maxcount()
        self.epoch = 0
        self.input_range = (-2.0, 2.0)
        self.mode = mode

    def set_lr(self, lr, momentum=0.9):
        """Set the learning rate.

        Keeps track of current learning rate and only allocates a new optimizer if it changes."""
        if lr != self.last_lr:
            self.optimizer = optim.SGD(
                self.model.parameters(), lr=lr, momentum=momentum
            )
            self.last_lr = lr

    def train_batch(self, inputs, targets):
        """All the steps necessary for training a batch.

        Stores the last batch in self.last_batch.
        Adds the loss to self.losses.
        Clips the gradient if self.clip_gradient is not None.
        """
        assert inputs.ndim == 4, inputs.shape
        assert inputs.shape[1] in [1, 3], inputs.shape
        self.last_batch = (inputs, None, targets)
        self.model.train()
        assert (
            inputs.amin() >= self.input_range[0]
            and inputs.amax() <= self.input_range[1]
        )
        if self.mode == "ctc":
            assert isinstance(targets, tuple) and len(targets) == 2, targets
            outputs = self.model.forward(inputs.to(self.device))
            assert targets[0].amax() <= outputs.size(1), (targets[0].amax(), outputs.size(1))
            loss = self.compute_loss(outputs.cpu(), (targets[0].cpu(), targets[1].cpu()))
        elif self.mode == "tf":
            assert isinstance(targets, torch.Tensor)
            assert targets.shape[0] == inputs.shape[0]
            outputs = self.model.forward(inputs.to(self.device), targets.to(self.device))
            assert targets.amax() <= outputs.size(1)
            loss = self.compute_loss(outputs, targets.to(self.device))
        else:
            outputs = self.model.forward(inputs.to(self.device))
            loss = self.compute_loss(outputs, targets.to(self.device))
        if torch.isnan(loss):
            raise ValueError("loss is nan")
        self.optimizer.zero_grad()
        loss.backward()
        if self.clip_gradient is not None:
            nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradient)
        self.optimizer.step()
        self.last_batch = (inputs, targets, outputs)
        return loss.detach().item()

    def compute_loss(self, outputs, targets):
        """Call the loss function. Override for special cases."""
        return self.lossfn(outputs, targets)

    def probs_batch(self, inputs):
        """Compute probability outputs for the batch. Uses `probfn`."""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.forward(inputs.to(self.device))
        return self.probfn(outputs.detach().cpu())

    def train(
        self, loader, epochs=1, learning_rates=None, total=None, cont=False, every=None
    ):
        """Train over a dataloader for the given number of epochs."""
        if every:
            self.every = every
        assert isinstance(learning_rates, list)
        epochs = max(epochs, len(learning_rates))
        epochs = min(epochs, 100000)
        learning_rates = extend_to(learning_rates, epochs)
        start_epoch = self.epoch
        for epoch in range(start_epoch, epochs):
            print("starting epoch", epoch)
            lr = learning_rates[epoch]
            self.set_lr(lr)
            self.epoch = epoch
            self.count = 0
            for sample in loader:
                images, targets = sample
                loss = self.train_batch(images, targets)
                if loss is None:
                    continue
                self.report()
                self.losses.append(float(loss))
                self.count += 1
                if len(self.losses) >= self.maxcount:
                    print("reached maxcount (inner)")
                    break
            if len(self.losses) >= self.maxcount:
                print("reached maxcount")
                break
            self.save_epoch(epoch)
        print("finished with all epochs")
        self.report_end()


class LineTrainer(BaseTrainer):
    """Specialized Trainer for training line recognizers with CTC."""

    def __init__(self, model, charset=None, **kw):
        super().__init__(model, **kw)
        self.charset = charset

    def report_outputs(self, ax, outputs):
        """Plot the posteriors for each class and location."""
        # layers.check_order(outputs, "BDL")
        pred = outputs[0].detach().cpu().softmax(0)
        assert pred.ndim == 2, pred.shape
        # assert pred.shape[0] == len(self.charset)+1, (pred.shape, len(self.charset)+1)
        if self.mode == "ctc":
            decoded = ctc_decode(pred)
        elif self.mode == "tf":
            decoded = [c for c in pred.argmax(0) if c != 0]
        else:
            raise ValueError(self.mode)
        if self.charset:
            ax.set_title(self.charset.decode(decoded))
        for i in range(pred.shape[0]):
            ax.plot(pred[i].numpy())

    def errors(self, loader):
        """Compute OCR errors using edit distance."""
        total = 0
        errors = 0
        for inputs, targets in loader:
            targets, tlens = targets
            predictions = self.predict_batch(inputs)
            start = 0
            for p, l in zip(predictions, tlens):
                t = targets[start : start + l].tolist()
                errors += editdistance.distance(p, t)
                total += len(t)
                start += l
                if total > self.maxcount:
                    break
            if total > self.maxcount:
                break
        return errors, total

    def predict_batch(self, inputs, **kw):
        """Predict and decode a batch."""
        probs = self.probs_batch(inputs)
        result = [ctc_decode(p, **kw) for p in probs]
        return result


class SegTrainer(BaseTrainer):
    """Segmentation trainer: image to pixel classes."""

    def __init__(self, model, margin=16, masked=4, lossfn=None, **kw):
        lossfn = lossfn or nn.CrossEntropyLoss()
        """Like regular trainer but allows margin specification."""
        super().__init__(model, lossfn=lossfn, **kw)
        self.margin = margin
        self.masked = masked

    def compute_loss(self, outputs, targets):
        if targets.ndim == 3:
            return self.compute_loss_ce(outputs, targets)
        elif targets.ndim == 4:
            return self.compute_loss_bce(outputs, targets)
        else:
            raise ValueError(targets.shape)

    def compute_loss_bce(self, outputs, targets):
        """Compute loss taking a margin into account."""
        b, d, h, w = outputs.shape
        b1, d1, h1, w1 = targets.shape
        assert b == b1 and d == d1, (outputs.shape, targets.shape)
        assert h <= h1 and w <= w1 and h1 - h < 5 and w1 - w < 5, (
            outputs.shape,
            targets.shape,
        )
        targets = targets[:, :, :h, :w]
        # lsm = outputs.log_softmax(1)
        if self.masked >= 0:
            mask = ndi.maximum_filter(
                targets.sum(axis=1).numpy() > 0, (0, self.masked, self.masked)
            )
            mask = torch.tensor(mask, dtype=torch.uint8)
            outputs = outputs * mask.to(outputs.device).unsqueeze(1)
            targets = targets * mask.to(targets.device).unsqueeze(1)
        if self.margin > 0:
            m = self.margin
            outputs = outputs[:, :, m:-m, m:-m]
            targets = targets[:, :, m:-m, m:-m]
        loss = self.lossfn(outputs, targets.to(outputs.device))
        return loss
    
    def compute_loss_ce(self, outputs, targets):
        """Compute loss taking a margin into account."""
        b, d, h, w = outputs.shape
        b1, h1, w1 = targets.shape
        assert h <= h1 and w <= w1 and h1 - h < 5 and w1 - w < 5, (
            outputs.shape,
            targets.shape,
        )
        targets = targets[:, :h, :w]
        # lsm = outputs.log_softmax(1)
        if self.masked >= 0:
            mask = ndi.maximum_filter(
                targets.numpy() > 0, (0, self.masked, self.masked)
            )
            mask = torch.tensor(mask, dtype=torch.uint8)
            outputs = outputs * mask.to(outputs.device).unsqueeze(1)
            targets = targets * mask.to(targets.device)
        if self.margin > 0:
            m = self.margin
            outputs = outputs[:, :, m:-m, m:-m]
            targets = targets[:, m:-m, m:-m]
        loss = self.lossfn(outputs, targets.to(outputs.device))
        return loss

    def report_outputs(self, ax, outputs):
        """Display the RGB output posterior probabilities."""
        p = outputs.detach().cpu().softmax(1)
        b, d, h, w = outputs.size()
        result = asnp(p)[0].transpose(1, 2, 0)
        result -= amin(result)
        result /= amax(result)
        if result.shape[2] >= 4:
            result = result[:, :, 1:4]
        ax.imshow(result)
        ax.plot([w // 2, w // 2], [0, h], color="white", alpha=0.5)

    def report_extra(self, ax, inputs, targets, outputs):
        p = outputs.detach().cpu().softmax(1)
        b, d, h, w = p.size()
        if d == 3:
            colors = "r g b".split()
        elif d == 4:
            colors = "black r g b".split()
        elif d == 7:
            colors = "black r g b c m y".split()
        for i in range(min(d, len(colors))):
            ax.plot(p[0, i, :, w // 2], color=colors[i])
