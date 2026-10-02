# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import pandas as pd

from aiter.ops import gemm_op_a8w8 as gemm_mod

Q = "torch.float8_e4m3fn"
LARGE = {
    "libtype": "flydsl",
    "splitK": 0,
    "kernelName": "flydsl_bpreshuflle_128x256x128_F8_F8_B16_1x2x4x2_default",
}
SMALL = {"libtype": "ck", "splitK": 0, "kernelName": "small_m"}


def _clear():
    gemm_mod.get_GEMM_config_with_quant_type.cache_clear()
    gemm_mod._GEMM_QUANT_TYPE_CACHE.clear()
    gemm_mod._GEMM_QUANT_TYPE_HAS_GFX.clear()
    gemm_mod._GEMM_QUANT_TYPE_MAX_M.clear()


def test_largest_tuned_m_is_used_only_above_the_table():
    cache = {
        ("gfx950", 256, 128, 3072, 512, Q): SMALL,
        ("gfx950", 256, 32768, 3072, 512, Q): LARGE,
        ("gfx950", 256, 64, 128, 7168, Q): SMALL,
    }
    index = gemm_mod._index_largest_tuned_m(cache, has_gfx=True)
    above = gemm_mod._config_for_m_above_table(
        index, True, "gfx950", 256, 99000, 3072, 512, Q
    )
    assert above["kernelName"] == LARGE["kernelName"]
    assert (
        gemm_mod._config_for_m_above_table(
            index, True, "gfx950", 256, 32768, 3072, 512, Q
        )
        is None
    )
    assert (
        gemm_mod._config_for_m_above_table(
            index, True, "gfx950", 256, 100, 3072, 512, Q
        )
        is None
    )
    assert (
        gemm_mod._config_for_m_above_table(
            index, True, "gfx950", 256, 200000, 6144, 7168, Q
        )
        is None
    )


def test_lookup_reuses_largest_row_and_keeps_small_misses(tmp_path, monkeypatch):
    path = tmp_path / "tuned.csv"
    pd.DataFrame(
        [
            {
                "gfx": "gfx950",
                "cu_num": 256,
                "M": 128,
                "N": 3072,
                "K": 512,
                "q_dtype_w": Q,
                **SMALL,
                "kernelId": 1,
            },
            {
                "gfx": "gfx950",
                "cu_num": 256,
                "M": 32768,
                "N": 3072,
                "K": 512,
                "q_dtype_w": Q,
                **LARGE,
                "kernelId": 727,
            },
        ]
    ).to_csv(path, index=False)

    monkeypatch.setattr(gemm_mod, "get_gfx", lambda: "gfx950")
    monkeypatch.setattr(gemm_mod, "get_cu_num", lambda: 256)

    def padded(m, n, k, gl):
        if m == 100 and gl == 0:
            return 128
        return m + 3

    monkeypatch.setattr(gemm_mod, "get_padded_m", padded)
    _clear()
    try:
        exact = gemm_mod.get_GEMM_config_with_quant_type(128, 3072, 512, Q, str(path))
        padded_hit = gemm_mod.get_GEMM_config_with_quant_type(
            100, 3072, 512, Q, str(path)
        )
        above = gemm_mod.get_GEMM_config_with_quant_type(
            40960, 3072, 512, Q, str(path)
        )
        small_miss = gemm_mod.get_GEMM_config_with_quant_type(
            50, 3072, 512, Q, str(path)
        )
        absent = gemm_mod.get_GEMM_config_with_quant_type(
            90000, 6144, 7168, Q, str(path)
        )
    finally:
        _clear()

    assert exact["kernelName"] == SMALL["kernelName"]
    assert padded_hit["kernelName"] == SMALL["kernelName"]
    assert above["libtype"] == "flydsl"
    assert above["kernelName"] == LARGE["kernelName"]
    assert small_miss is None
    assert absent is None
