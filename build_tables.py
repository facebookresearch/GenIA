#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Replicate the paper's quantitative tables on the project page.

Reads the **same** `csv/*.csv` files the paper bakes from and reuses
`regen_tables.py`'s table registry, column specs and ranking primitives
(`TABLES`, `parse_float`, `top_ranks`), so the numbers, the column selection and
the best/second-best marking are identical to the paper by construction — down
to the typefaces they are marked in, bold and underline. Only the rendering
layer differs: HTML instead of LaTeX.

    csv/*.csv ──┬── regen_tables.py ──> baked/tables-baked.tex   (paper)
                └── build_tables.py  ──> webpage/tables.js       (this page)

So the refresh path is unchanged: `python update_all_results.py` rewrites the
CSVs, then `python regen_tables.py` and this script re-emit from them.

    python paper/webpage/build_tables.py             # write tables.js + verify
    python paper/webpage/build_tables.py --no-verify

`--verify` (on by default) cross-checks every table against the baked LaTeX: the
multiset of numeric cells and of rank-marked cells must match. Since the
grouping/ranking loop here is a second implementation of `bake_table`'s, that
check is what keeps the two from drifting apart silently.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The project page lives BESIDE the paper, not inside it: paper/ is pushed to Overleaf,
# which has a project-wide file limit and no use for the page's sources.
PAPER = HERE.parent / "paper"
sys.path.insert(0, str(PAPER))

import regen_tables as rt  # noqa: E402  (needs PAPER on sys.path)
from _coverage import INCOMPLETE_MARK, MISSING_MARK  # noqa: E402

# --------------------------------------------------------------------------- #
# Page layout: which baked table goes where. Mirrors the paper's floats
# (tab:full_benchmarks / tab:full_ablations / tab:extra_backbones) and their
# subtable captions. A table in regen_tables.TABLES that is missing here still
# gets emitted, into a trailing section — a new table cannot silently vanish.
# --------------------------------------------------------------------------- #

# Each entry names the page section it mounts into — "results" (cross-method
# comparisons) or "ablations" (studies of our own components) — and optionally
# `collapsed` (folded away behind its header until clicked). Tables always
# stack vertically (the paper's 2x2 figure* float layout is not replicated
# here).
SECTIONS = [
    {"group": "results", "title": "Benchmarks",
     "blurb": "The main paper's <em>Table&nbsp;1</em>. Per column, the "
              "<span class='rank1'>best</span> entry is bold and the "
              "<span class='rank2'>second best</span> underlined, ranked within each "
              "input-view group; a dash marks a metric the method does not report. "
              "TRELLISv2 and Pixal3D predict lighting-disentangled materials, so they "
              "appear in the qualitative comparisons only.",
     "tables": [("tabStaticFull", "Static scenes"),
                ("tabDynamicFull", "Dynamic scenes")]},

    {"group": "ablations", "title": "Appearance components",
     "blurb": "<em>Attn</em> is the visibility attention bias, <em>RG</em> the "
              "rendering guidance, <em>TTR</em> the test-time refinement. Shape and "
              "pose are held fixed (ground-truth on GSO-30 and ActionBench, "
              "SAM3D-predicted on CO3D, ActionMesh on DAVIS), so a row differs from "
              "the one above it only in the component it adds. Held-out views report "
              "co-PSNR where the dataset has co-visibility masks, otherwise LPIPS and "
              "CLIP-I alone.",
     "tables": [("tabStaticAppearance", "Static scenes"),
                ("tabDynamicAppearance", "Dynamic scenes")]},

    {"group": "ablations", "title": "Pose prediction",
     "blurb": "Four cumulative steps, each adding one component of our pose stage to "
              "per-frame SAM3D (Base); each lane carries the metrics it has. Jit. is "
              "relative to the reference's own trajectory acceleration, so 1.0 means "
              "it moves like the reference. Rotation averaging is <em>inert by "
              "construction</em> on the static lanes (a single timestamp has nothing "
              "to average, so those groups repeat Base). Registration is most "
              "beneficial for static multi-view alignment, where it halves Chamfer "
              "distance; on DAVIS it improves depth but costs appearance.",
     # Split into a table per regime (bands = the four `pose_ladder_full_cols` lane
     # names, matching tab:ablation_poses_full in the supplementary) rather than one
     # wide four-lane table — the two never share a metric set, so the columns don't
     # line up as one table anyway. The un-split macro is still what gets verified
     # against the baked LaTeX (see `main`); this only re-partitions its columns for
     # display.
     "tables": [("tabAblationPosesFull", "Pose prediction — all four lanes",
                 [("Dynamic lanes", {"DAVIS", "ActionBench"}),
                  ("Static lanes", {"GSO-30", "GSO-30 MV"})])]},
]

# Baked tables the page deliberately does NOT show, with the reason. A macro
# listed here is neither mounted nor dumped into the "Other tables" catch-all —
# which exists so a NEW table cannot vanish silently, and would otherwise make
# every deliberate omission look like an oversight.
#
# Most of these share one rule: the page shows the tables the PAPER shows, so
# it cannot publish a metric the paper withholds. Each reaches no float in the
# paper (checked: no `\tab...` use outside its own tabs/*.tex, and no `\input`
# of that file). tabPseudoTracking is the one exception — the paper DOES print
# it (sec/supplementary.tex), but its δavg/Err columns are byte-identical to
# what tabDynamicFull (in Benchmarks, above) already reports for DAVIS; only
# its AJ and Anch. columns are unique, and dropping the section drops those
# two along with the duplicate pair.
EXCLUDED = {
    "tabGsoFull": "per-dataset cut of Table 1; reports SSIM, which the paper hides",
    "tabCoThreeDFull": "per-dataset cut of Table 1; reports SSIM, which the paper hides",
    "tabOABFull": "per-dataset cut of Table 1; reports SSIM, which the paper hides",
    "tabDAVISFull": "per-dataset cut of Table 1; reports SSIM, which the paper hides",
    "tabGsoAppearance": "per-dataset cut of the appearance ladder; superseded by the "
                        "merged tabStaticAppearance (2026-08-29)",
    "tabCoThreeDAppearance": "per-dataset cut of the appearance ladder; superseded by "
                             "the merged tabStaticAppearance (2026-08-29)",
    "tabOABAppearance": "per-dataset cut of the appearance ladder; superseded by the "
                        "merged tabDynamicAppearance (2026-08-29)",
    "tabDAVISAppearance": "per-dataset cut of the appearance ladder; superseded by the "
                          "merged tabDynamicAppearance (2026-08-29)",
    "tabAblationPoses": "DAVIS-only cut of the pose ladder; the paper shows "
                        "tabAblationPosesFull instead",
    "tabPseudoTracking": "its δavg/Err duplicate tabDynamicFull's DAVIS columns; "
                         "AJ/Anch. (its only unique columns) are not carried elsewhere",
}

# --------------------------------------------------------------------------- #
# LaTeX -> HTML for the column titles and formatted cells.
# --------------------------------------------------------------------------- #

TEX_HTML = [
    (r"$(t,s)$", "(<em>t</em>,<em>s</em>)"),
    (r"$\times 10^{-3}$", "&times;10<sup>&minus;3</sup>"),
    (r"$\delta_{\mathrm{avg}}$", "&delta;<sub>avg</sub>"),
    (r"$\rightarrow\!1$", "&rarr;&thinsp;1"),
    (r"$\uparrow$", "&uarr;"),
    (r"$\downarrow$", "&darr;"),
    (r"$\times$", "&times;"),
    (r"\#", "#"),
    ("~", "&nbsp;"),
]

# Ablation/TTR glyphs. The paper uses \cmark / \xmark / -- ; keep the same
# three-way distinction (on / off / no such stage for this method).
GLYPH_HTML = {
    r"\cmark": '<span class="mark-on">&#10003;</span>',
    r"\xmark": '<span class="mark-off">&times;</span>',
    "--": "&ndash;",
}


def tex_to_html(s: str) -> str:
    for tex, html in TEX_HTML:
        s = s.replace(tex, html)
    if "\\" in s or "$" in s:
        raise ValueError(
            f"unconverted LaTeX in {s!r} — add it to TEX_HTML in build_tables.py"
        )
    return s


def method_html(label: str, ours_label: str) -> str:
    """Mirror `regen_tables.method_label`'s two renderings of an "Ours (...)" row:
    `source` -> `Ours<sub>SAM3D</sub>` (the TTR variant drives its own column),
    `ttr` -> plain `Ours w/ TTR` (the backbone is named in the prose, no TTR column).
    Citations are dropped: this page is not the bibliography."""
    m = rt.OURS_LABEL.match(label)
    if not m:
        return tex_to_html(label)
    if ours_label == "source":
        return f"Ours<sub>{m.group(1)}</sub>"
    return f"Ours {m.group(2)}" if m.group(2) else "Ours"


def cell_html(raw: str, fmt) -> tuple[str, bool]:
    """Render a metric cell. Returns (html, is_missing).

    A ❌-marked cell renders as a red × with NO value, exactly as the paper's
    `render_cell` empties it (`MISSING_CELL`): a stale number is a number nothing
    maintains. A `--` cell is emitted as an en-dash rather than passed through:
    LaTeX renders `--` as one, HTML would show two hyphens.
    """
    if MISSING_MARK in raw:
        return '<span class="mark-fail">&times;</span>', True
    clean = raw.replace(INCOMPLETE_MARK, "")
    if clean.strip() in rt.MISSING:
        return "&ndash;", True
    return tex_to_html(fmt(clean)), False


# --------------------------------------------------------------------------- #
# Compute one table (mirrors regen_tables.bake_table, emitting data not LaTeX)
# --------------------------------------------------------------------------- #

def _band_cells(cols, level: str) -> list[dict] | None:
    """One header band row for attribute `level` ("band" or "superband"), or None
    when no column carries it. Merges runs exactly as `regen_tables._band_row`
    does: a band only joins its neighbour when the level above it agrees too, so
    two datasets' identically-named "Train" bands stay separate cells."""
    if not any(getattr(c, level) is not None for c in cols):
        return None
    key = (lambda c: (c.superband, c.band)) if level == "band" else (lambda c: c.superband)
    cells, i = [], 0
    while i < len(cols):
        if getattr(cols[i], level) is None:
            # one cell per unbanded column, as regen_tables.build_header does;
            # grouping on equality would merge them all (None == None)
            cells.append({"label": "", "span": 1, "rule": False})
            i += 1
            continue
        j = i
        while j < len(cols) and getattr(cols[j], level) is not None and key(cols[j]) == key(cols[i]):
            j += 1
        cells.append({"label": tex_to_html(getattr(cols[i], level)), "span": j - i,
                      "rule": True})
        i = j
    return cells


def compute_table(spec: rt.Table, col_filter=None) -> dict:
    # Rows + resolved columns from regen_tables, so this page and the baked LaTeX agree by
    # construction rather than by two copies of the same prelude staying in step.
    # `col_filter`, when given, keeps only the columns it accepts (used to split one
    # baked table's column bands into several page tables -- the row data and every
    # per-cell computation below are untouched, so this cannot introduce a value the
    # unfiltered table wasn't already verified against).
    rows, cols = rt.table_rows_and_cols(spec)
    if col_filter is not None:
        cols = [c for c in cols if col_filter(c)]
    rows = [r for r in rows
            if not any(all(r.get(k) == v for k, v in h.items()) for h in spec.hidden_rows)]
    metrics = [c for c in cols if c.kind == "metric"]

    if spec.group_col is None:
        groups: list[tuple[str | None, list[int]]] = [(None, list(range(len(rows))))]
    else:
        bucket: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            bucket.setdefault(r[spec.group_col], []).append(i)
        groups = list(bucket.items())

    rank_map: dict[tuple[str, int], int] = {}
    for gval, idxs in (groups if spec.rank_cells else ()):
        if gval in spec.groups_unranked:
            continue
        for m in metrics:
            vals = [(v, i) for i in idxs if (v := rt.parse_float(rows[i][m.col])) is not None]
            if len(vals) < 2:
                continue
            for i, rank in rt.top_ranks(vals, m.direction).items():
                rank_map[(m.col, i)] = rank

    # header: optional superband + band rows (colspan groups) + the title row
    superbands = _band_cells(cols, "superband")
    bands = _band_cells(cols, "band")
    head = [{"label": tex_to_html(c.title), "align": c.align} for c in cols]

    body = []
    for gi, (gval, idxs) in enumerate(groups):
        for ri, i in enumerate(idxs):
            r = rows[i]
            bold = spec.method_col is not None and (
                r[spec.method_col] in spec.bold_methods
                or (spec.bold_prefix is not None and r[spec.method_col].startswith(spec.bold_prefix))
            )
            cells = []
            cell_of_col = {}  # col index -> cells index (a multirow group cell is skipped)
            for ci, c in enumerate(cols):
                cell: dict = {} if c.align == "c" else {"align": c.align}
                if c.kind == "method":
                    cell["html"] = method_html(r[spec.method_col], spec.ours_label)
                elif c.kind == "flag":
                    cell["html"] = GLYPH_HTML.get(rt.flag_cell(r[c.col]), r[c.col])
                elif c.kind == "ttt":
                    cell["html"] = GLYPH_HTML.get(rt.ttt_cell(r[spec.method_col]), "&ndash;")
                elif c.kind == "group":
                    if spec.group_render == "multirow":
                        if ri:
                            continue          # covered by the first row's rowspan
                        cell["rowspan"] = len(idxs)
                    cell["html"] = tex_to_html(gval)
                else:
                    cell["html"], missing = cell_html(r[c.col], c.fmt)
                    rank = rank_map.get((c.col, i))
                    if rank:
                        cell["rank"] = rank
                    if missing:
                        cell["muted"] = True
                    if bold and spec.bold_cells:
                        cell["bold"] = True
                if bold and c.kind == "method":
                    cell["bold"] = True
                cell_of_col[ci] = len(cells)
                cells.append(cell)
            # One ⚠ per partial-result row segment, on the same cell the paper puts it:
            # the last cell of each source's block on a merged table, else the row's last.
            for pref, last in rt.warn_slots(spec, cols):
                if any(INCOMPLETE_MARK in v for k, v in r.items() if k.startswith(pref)):
                    cells[cell_of_col[last]]["html"] += \
                        ' <span class="mark-warn" title="averaged over a subset of scenes">&#9651;</span>'
            body.append({"cells": cells, "newGroup": ri == 0 and gi > 0})

    return {"csv": spec.csv, "superbands": superbands, "bands": bands,
            "head": head, "rows": body}


# --------------------------------------------------------------------------- #
# Verification against the baked LaTeX
# --------------------------------------------------------------------------- #

NUM = re.compile(r"-?\d+\.?\d*")
# The paper renders ranks as typefaces (sty/preamble.tex): \best{} bold,
# \secondbest{} underlined. Inverted from regen_tables' own map, so a macro
# rename there propagates here for free — this check went stale once already,
# when the paper moved off \cellcolor{first|second|third}.
RANK_OF = {rt.macro_for_rank(r): r for r in (1, 2, 3)}


def baked_macros(path: Path) -> dict[str, str]:
    """macro name -> its body, from baked/tables-baked.tex."""
    text = path.read_text()
    out = {}
    for m in re.finditer(r"\\newcommand\{\\(\w+)\}\{%\n(.*?)\n\}\n", text, re.S):
        out[m.group(1)] = m.group(2)
    return out


def _nums_html(html: str) -> list[str]:
    """Numbers a reader actually sees — entities (&#10003;) are glyphs, not data."""
    return NUM.findall(re.sub(r"<[^>]+>|&#?\w+;", " ", html))


def _nums_tex(cell: str) -> list[str]:
    """Same, for a baked LaTeX cell: citation keys carry years and \\multirow
    carries a row count, neither of which is a value in the table. Every other
    command token is dropped with its argument text KEPT (`\\best{21.5}` -> 21.5;
    `\\ours{SAM3D}` keeps the 3 its HTML twin `Ours<sub>SAM3D</sub>` also shows)."""
    cell = re.sub(r"\\cite\{[^}]*\}", " ", cell)
    cell = re.sub(r"\\multirow\{\d+\}\{\*\}", " ", cell)
    cell = re.sub(r"\\[a-zA-Z]+", " ", cell)
    return NUM.findall(cell)


def verify(table: dict, baked: str) -> list[str]:
    """Every number and every rank mark must match the paper's baked table.

    Compared cell-by-cell rather than by scraping the whole block, so a bolded
    ranked cell (`\\best{\\textbf{12.3}}`) is read the same on both sides.
    """
    body = baked.split(r"\midrule", 1)[-1].split(r"\bottomrule")[0]
    theirs: list[tuple[str, int | None]] = []
    for line in body.splitlines():
        line = line.strip().removesuffix(r"\\").strip()
        if not line or line.startswith("\\midrule"):
            continue
        for cell in line.split("&"):
            rank = next((r for c, r in RANK_OF.items() if f"\\{c}{{" in cell), None)
            theirs += [(n, rank) for n in _nums_tex(cell)]

    mine = [(n, c.get("rank")) for row in table["rows"] for c in row["cells"]
            for n in _nums_html(c["html"])]

    problems = []
    if sorted(n for n, _ in mine) != sorted(n for n, _ in theirs):
        only_mine = sorted(set(n for n, _ in mine) - set(n for n, _ in theirs))
        only_theirs = sorted(set(n for n, _ in theirs) - set(n for n, _ in mine))
        problems.append(f"numbers differ (only here: {only_mine[:6]}, only in paper: {only_theirs[:6]})")
    for rank in (1, 2, 3):
        m = sorted(n for n, r in mine if r == rank)
        t = sorted(n for n, r in theirs if r == rank)
        if m != t:
            problems.append(f"rank-{rank}: {len(m)} cells here vs {len(t)} in paper "
                            f"(only here: {sorted(set(m) - set(t))[:4]})")
    return problems


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=HERE / "tables.js")
    ap.add_argument("--no-verify", action="store_true",
                    help="skip the cross-check against baked/tables-baked.tex")
    args = ap.parse_args()

    by_macro = {t.macro: t for t in rt.TABLES}
    placed = {entry[0] for sec in SECTIONS for entry in sec["tables"]} | set(EXCLUDED)
    sections = [dict(sec) for sec in SECTIONS]
    leftover = [(m, m) for m in by_macro if m not in placed]
    if leftover:
        sections.append({"group": "results", "title": "Other tables",
                         "blurb": "Not yet placed in a section of this page — add them to "
                                  "<code>SECTIONS</code> in <code>build_tables.py</code>.",
                         "tables": leftover})

    baked_path = PAPER / "baked" / "tables-baked.tex"
    baked = baked_macros(baked_path) if not args.no_verify and baked_path.is_file() else {}

    out_sections, n_tables, n_bad = [], 0, 0
    for sec in sections:
        entries = []
        for macro, label, *rest in sec["tables"]:
            split = rest[0] if rest else None
            spec = by_macro.get(macro)
            if spec is None:
                print(f"! {macro} is in SECTIONS but not in regen_tables.TABLES — skipped")
                continue
            # Verification always runs against the full, un-split table: splitting only
            # re-partitions columns already computed above, so a table verified whole
            # cannot disagree with the baked LaTeX split into halves.
            table = compute_table(spec)
            table.update(id=macro, title=label)
            if macro in baked:
                for p in verify(table, baked[macro]):
                    print(f"! {macro} ({spec.csv}): {p}")
                    n_bad += 1
            if split is None:
                entries.append(table)
                n_tables += 1
            else:
                for i, (sub_label, bands) in enumerate(split):
                    sub = compute_table(
                        spec, col_filter=lambda c, bands=bands: c.band is None or c.band in bands)
                    sub.update(id=f"{macro}__{i}", title=sub_label)
                    entries.append(sub)
                    n_tables += 1
        if entries:
            block = {"group": sec["group"], "title": sec["title"],
                     "blurb": sec["blurb"], "tables": entries}
            if sec.get("collapsed"):
                block["collapsed"] = True
            out_sections.append(block)

    banner = ("/* Copyright (c) Meta Platforms, Inc. and affiliates. */\n\n"
              "/* GENERATED by paper/webpage/build_tables.py from the paper's csv/ —\n"
              " * do not hand-edit. Refresh: update_all_results.py, then this script. */\n")
    args.out.write_text(banner + "window.PAGE_TABLES = "
                        + json.dumps(out_sections, ensure_ascii=False,
                                     separators=(",", ":")) + ";\n")

    print(f"wrote {args.out.name}: {n_tables} tables in {len(out_sections)} sections")
    if baked:
        print("verified against baked/tables-baked.tex: "
              + ("all match" if not n_bad else f"{n_bad} MISMATCH(ES) above"))
    else:
        print("! not verified (no baked/tables-baked.tex, or --no-verify)")
    if n_bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
