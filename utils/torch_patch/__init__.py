"""
Utilities for runtime patching of selected PyTorch behaviors used by GeoGRPO.

Currently provides helpers to relax CUDA graph capture error handling so that
TorchInductor can safely call ``torch.cuda.synchronize`` while a graph capture
is in progress (required when vLLM records mixed prefill-decode graphs).
"""

from __future__ import annotations

import threading

_RELAXED_PATCH_LOCK = threading.Lock()
_RELAXED_PATCHED = False


def enable_relaxed_cuda_graph_capture() -> None:
    """
    Force ``torch.cuda.graph`` to always capture with ``capture_error_mode``
    set to ``\"relaxed\"``.

    TorchInductor's CUDA graph tree recorder occasionally calls
    ``torch.cuda.synchronize`` while a capture is ongoing. With the default
    (``\"thread_local\"``) capture mode this raises
    ``cudaErrorStreamCaptureUnsupported``. Relaxed mode allows such calls,
    which unblocks vLLM's mixed prefill CUDA graph capture without needing
    invasive patches inside PyTorch itself.
    """
    global _RELAXED_PATCHED
    with _RELAXED_PATCH_LOCK:
        if _RELAXED_PATCHED:
            return

        import torch.cuda  # Local import to avoid initializing CUDA too early.
        from torch.cuda import graphs as cuda_graphs

        base_graph_cls = cuda_graphs.graph

        class _RelaxedGraph(base_graph_cls):  # type: ignore[misc]
            def __init__(self, cuda_graph, pool=None, stream=None,
                         capture_error_mode="global"):
                super().__init__(cuda_graph,
                                 pool=pool,
                                 stream=stream,
                                 capture_error_mode="relaxed")

        _RelaxedGraph.__name__ = base_graph_cls.__name__
        _RelaxedGraph.__qualname__ = base_graph_cls.__qualname__
        _RelaxedGraph.__doc__ = base_graph_cls.__doc__
        _RelaxedGraph.__module__ = base_graph_cls.__module__

        cuda_graphs.graph = _RelaxedGraph
        torch.cuda.graph = _RelaxedGraph

        _RELAXED_PATCHED = True
