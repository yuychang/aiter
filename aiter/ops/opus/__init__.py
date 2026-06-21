# SPDX-License-Identifier: MIT
# Copyright (C) 2025-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Import-safe public API for Opus kernels."""

from ._arch import _detect_arch

_GEMM_SUPPORTED = {"gfx950", "gfx942"}
_GEMM_HINT = (
    "opus_gemm supports gfx950 (MFMA 16x16x32 / ds_read_b64_tr / 160 KiB "
    "LDS) and gfx942 (MFMA 16x16x16 / ds_read_b128 / 64 KiB LDS). Set "
    "GPU_ARCHS to one of these (or run on a matching device) to use this "
    "module."
)

_gemm_arch_ok, _gemm_detected_arch = _detect_arch(_GEMM_SUPPORTED)


def _unsupported_arch_stub(name: str, supported: set[str], detected, hint: str):
    def _stub(*_args, **_kwargs):
        raise RuntimeError(
            f"{name} requires GPU arch in {sorted(supported)}; "
            f"detected {detected!r}. {hint}"
        )

    _stub.__name__ = name
    _stub.__qualname__ = name
    _stub.__doc__ = f"Stub: unavailable on {detected!r}."
    return _stub


def _unsupported_arch_stubs(names: tuple[str, ...], supported, detected, hint):
    return tuple(
        _unsupported_arch_stub(name, supported, detected, hint) for name in names
    )


if _gemm_arch_ok:
    from .gemm_op_a16w16 import (  # noqa: E402
        opus_gemm_a16w16_tune,
        gemm_a16w16_opus,
        opus_gemm_workspace_init,
    )
else:
    # Keep `import aiter` working on unsupported GPUs; fail only if called.
    (
        gemm_a16w16_opus,
        opus_gemm_a16w16_tune,
        opus_gemm_workspace_init,
    ) = _unsupported_arch_stubs(
        (
            "gemm_a16w16_opus",
            "opus_gemm_a16w16_tune",
            "opus_gemm_workspace_init",
        ),
        _GEMM_SUPPORTED,
        _gemm_detected_arch,
        _GEMM_HINT,
    )

__all__ = [
    "opus_gemm_a16w16_tune",
    "gemm_a16w16_opus",
    "opus_gemm_workspace_init",
]
