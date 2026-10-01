"""Keep Linux orphaned job descendants owned across exec into the Node harness."""

import ctypes
import os
import sys

if sys.platform == "linux":
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "Could not enable harness child supervision")
    os.environ["GAMMA_LINUX_SUBREAPER"] = str(os.getpid())

os.execvp(sys.argv[1], sys.argv[1:])
