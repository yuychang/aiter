#!/usr/bin/env python3
"""Build the CK / CK-Tile a8w8 bpreshuffle *tune* modules (CPU only, no GPU).

The tuner builds them lazily on its first CK/cktile task, inside an mp worker
under the 1800 s per-task watchdog; compiling every instance takes longer than
that, so build them up front. Mirrors the build path in
aiter/jit/core.py::compile_ops for a pybind module.

    prebuild_tune_modules.py MODULE [MODULE ...]
"""
import os
import sys
import time

from aiter.jit import core


def build(md_name):
    try:
        core.get_module(md_name)
        print(f"[prebuild] {md_name}: already built", flush=True)
        return
    except ModuleNotFoundError:
        pass
    d = core.get_args_of_build(md_name)
    t0 = time.time()
    print(f"[prebuild] {md_name}: building ...", flush=True)
    clang = d.get("hip_clang_path")
    prev_clang = os.environ.get("HIP_CLANG_PATH")
    if clang and os.path.exists(clang):
        os.environ["HIP_CLANG_PATH"] = clang
    core.build_module(
        md_name,
        d["srcs"],
        d["flags_extra_cc"],
        d["flags_extra_hip"],
        d["blob_gen_cmd"],
        d["extra_include"],
        d["extra_ldflags"],
        d["verbose"],
        d["is_python_module"],
        d["is_standalone"],
        d["torch_exclude"],
        d.get("third_party", []),
        d.get("hipify", False),
        flags_extra_hip_per_source=d.get("flags_extra_hip_per_source", {}),
    )
    if prev_clang is None:
        os.environ.pop("HIP_CLANG_PATH", None)
    else:
        os.environ["HIP_CLANG_PATH"] = prev_clang
    core.get_module(md_name)
    print(f"[prebuild] {md_name}: built in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    for name in sys.argv[1:]:
        build(name)
