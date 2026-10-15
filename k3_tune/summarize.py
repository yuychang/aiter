#!/usr/bin/env python3
"""Before/after table: production default-config time vs tuned time per shape.

default_us  production op (aiter.gemm_a8w8_bpreshuffle / tuned_gemm.gemm_a16w16)
            with the shipped config, i.e. what serving runs today
            (``<tuner> --run_config`` BEFORE tuning; includes the "reuse the
            largest row" fallback for a8w8)
tuned_us    the tuner's own kernel time for the winning candidate
after_us    the same production op re-run with the tuned CSV
            (``<tuner> --run_config TUNED.csv``)
speedup     default_us / after_us, or default_us / tuned_us if the AFTER run
            did not produce a number

    summarize.py --kind a8w8|bf16 --untuned U.csv --tuned T.csv \
                 --before before.log [--after after.log] [--md out.md]
"""

import argparse
import re
import sys
from pathlib import Path

import gen_shapes as g
import pandas as pd

ROW = re.compile(r"^\((\d+), (\d+), (\d+),[^)]*\)\s*\|(.*)$")


def parse_bench(path):
    out = {}
    if not path:
        return out
    try:
        lines = Path(path).read_text(errors="ignore").splitlines()
    except OSError:
        return out
    for line in lines:
        m = ROW.match(line.strip())
        if not m:
            continue
        fields = [f.strip() for f in m.group(4).split("|")]
        if len(fields) < 2:
            continue
        try:
            us = float(fields[-2])
        except ValueError:
            us = None
        out[tuple(int(x) for x in m.group(1, 2, 3))] = (us, fields[-1])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=["a8w8", "bf16"], required=True)
    ap.add_argument("--untuned", required=True)
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after")
    ap.add_argument("--md")
    a = ap.parse_args()

    shapes = pd.read_csv(a.untuned)
    try:
        tuned = pd.read_csv(a.tuned)
    except (OSError, pd.errors.EmptyDataError):
        tuned = pd.DataFrame(columns=["M", "N", "K", "libtype", "kernelName", "us"])
    if "gfx" in tuned.columns:
        tuned = tuned[(tuned.gfx == g.GFX) & (tuned.cu_num == g.CU_NUM)]
    tuned = {(r.M, r.N, r.K): r for r in tuned.itertuples()}
    before, after = parse_bench(a.before), parse_bench(a.after)

    rows = []
    for r in shapes.sort_values(["N", "K", "M"]).itertuples():
        key = (r.M, r.N, r.K)
        t = tuned.get(key)
        d_us, d_st = before.get(key, (None, ""))
        a_us, a_st = after.get(key, (None, ""))
        k_us = float(t.us) if t is not None else None
        name = ""
        if t is not None:
            name = (
                f"{t.libtype}:{t.kernelName}"
                if isinstance(t.kernelName, str)
                else str(t.libtype)
            )
        ref = a_us if a_us and a_us > 0 else k_us
        sp = d_us / ref if d_us and d_us > 0 and ref else None
        rows.append(
            (
                r.M,
                r.N,
                r.K,
                d_us,
                k_us,
                a_us,
                sp,
                d_st or "-",
                a_st or "-",
                name or "NOT TUNED",
            )
        )

    def f(x, p=1):
        return "N/A" if x is None or math.isnan(x) else f"{x:.{p}f}"

    hdr = [
        "M",
        "N",
        "K",
        "default_us",
        "tuned_us",
        "after_us",
        "speedup",
        "dflt_st",
        "aft_st",
        "winner",
    ]
    table = [hdr] + [
        [
            str(m),
            str(n),
            str(k),
            f(d),
            f(ku),
            f(au),
            f(sp, 2) + "x" if sp else "N/A",
            ds,
            as_,
            w,
        ]
        for m, n, k, d, ku, au, sp, ds, as_, w in rows
    ]
    widths = [max(len(r[i]) for r in table) for i in range(len(hdr))]
    lines = [f"== {a.kind}: default config vs tuned =="]
    for i, r in enumerate(table):
        lines.append(
            "  ".join(
                c.ljust(w) if j == len(r) - 1 else c.rjust(w)
                for j, (c, w) in enumerate(zip(r, widths))
            )
        )
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))
    ok = [
        (d, (au if au and au > 0 else ku))
        for _, _, _, d, ku, au, _, _, _, _ in rows
        if d and d > 0 and ((au and au > 0) or ku)
    ]
    if ok:
        import math

        gm = math.exp(sum(math.log(d / t) for d, t in ok) / len(ok))
        lines.append(
            f"shapes with both numbers: {len(ok)}/{len(rows)}; geomean speedup {gm:.2f}x; "
            f"sum default {sum(d for d, _ in ok):.0f}us -> tuned {sum(t for _, t in ok):.0f}us"
        )
    missing = [r for r in rows if r[9] == "NOT TUNED"]
    if missing:
        lines.append(
            f"WARNING: {len(missing)} shape(s) have no tuned row: "
            + ", ".join(f"M{r[0]}xN{r[1]}xK{r[2]}" for r in missing)
        )

    existing = g.merged_tuned(
        (
            "a8w8_bpreshuffle_tuned_gemm.csv"
            if a.kind == "a8w8"
            else "bf16_tuned_gemm.csv"
        ),
        "a8w8_bpreshuffle_tuned_gemm" if a.kind == "a8w8" else "bf16_tuned_gemm",
    )
    new = pd.read_csv(a.tuned) if tuned else pd.DataFrame()
    if len(new):
        keys = ["M", "N", "K"] + (
            ["q_dtype_w"]
            if a.kind == "a8w8"
            else ["bias", "dtype", "outdtype", "scaleAB", "bpreshuffle"]
        )
        new = (
            new[(new.gfx == g.GFX) & (new.cu_num == g.CU_NUM)]
            if "gfx" in new.columns
            else new
        )
        dup = pd.merge(existing[keys].drop_duplicates(), new[keys], on=keys)
        lines.append(
            "merge collision check vs existing tuned tables: "
            + (
                "OK (no duplicate keys)"
                if dup.empty
                else f"FAIL {len(dup)} duplicate keys -> aiter's merge would rewrite/raise"
            )
        )
    text = "\n".join(lines)
    print(text)
    if a.md:
        with open(a.md, "a") as fh:
            fh.write("```\n" + text + "\n```\n\n")


if __name__ == "__main__":
    sys.exit(main())
