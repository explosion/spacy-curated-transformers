import contextlib
import shutil
import tempfile
from pathlib import Path

import torch
from thinc.api import NumpyOps


@contextlib.contextmanager
def make_tempdir():
    d = Path(tempfile.mkdtemp())
    yield d
    shutil.rmtree(str(d))


# Wrapper around torch.testing.assert_close with custom tolerances.
def torch_assertclose(
    a: torch.Tensor, b: torch.Tensor, *, atol: float = 1e-05, rtol: float = 1e-05
):
    torch.testing.assert_close(
        a,
        b,
        atol=atol,
        rtol=rtol,
    )


def xp_assert_array_equal(a, b):
    # Always convert to CPU since Cupy will strangely complain
    # about requiring a Numpy conversion before calling assert_array_close.
    if not isinstance(a, list):
        a = [a]
    if not isinstance(b, list):
        b = [b]
    ops = NumpyOps()
    for x, y in zip(a, b):
        ops.xp.testing.assert_array_equal(ops.asarray(x), ops.asarray(y))
