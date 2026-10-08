#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Collect orbit renders into the project page, resolving runs via mapping.json.

Every method shown on the page — ours and baselines alike — is a **label from
`mapping.json`'s benchmark registry** (`benchmark.baselines` / `davis_sources`),
the same entries `sync_benchs.py` reads to fill the paper's result CSVs. The page
therefore shows the runs the tables score, by construction: `DATASETS` below
declares WHICH labels each dataset offers (and at which view counts), the
registry says WHERE each one's runs live, and the only discovery left is per
declared run dir — which scenes have an orbit render
(`scripts/visualization/render_final_results.py -r orbit`).

Videos and stills are both **re-encoded for the web in the copy step**, mirroring
how `update_qualitative.py` converts the paper's figure renders to JPEG on copy:
H.264 / yuv420p / Main profile / `+faststart` for video, `--still-format` for
stills. The supplementary ZIP has a size budget the renderer's output does not
respect. README.md carries the measured size/quality tables for both.

    python paper/webpage/collect_assets.py --dry-run     # what would be written
    python paper/webpage/collect_assets.py               # encode + write data.js
    python paper/webpage/collect_assets.py --crf 23      # bigger, near-transparent
    python paper/webpage/collect_assets.py --no-transcode
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
# The project page lives BESIDE the paper, not inside it: paper/ is pushed to Overleaf,
# which has a project-wide file limit and no use for the page's sources.
PAPER = HERE.parent / "paper"
sys.path.insert(0, str(PAPER))
import _paths  # noqa: E402  (the mapping.json loader the paper scripts share)
import regen_tables  # noqa: E402  (OURS_LABEL: the label grammar's one parser)
# The tables' "Ours (<backbone>) w/ TTR" rows come from the ablation ladder's
# full-config cell; reuse the exact constants sync_benchs builds them with.
from sync_benchs import FULL_CONFIG, _ours_label  # noqa: E402

# --------------------------------------------------------------------------- #
# Hand-edited page metadata. The title and venue are NOT here — index.html owns
# them, as hand-written prose.
# --------------------------------------------------------------------------- #

# The public project page (https://autonomousvision.github.io/genia). Every
# button is a greyed-out placeholder until its target is available: set its
# `href` (and drop `disabled`) then. `icon` names an entry of
# LINK_ICONS in js/app.js.
PAGE_LINKS = [
    {"label": "PDF", "icon": "pdf", "href": "paper.pdf"},
    {"label": "arXiv", "icon": "arxiv", "href": "https://arxiv.org/abs/2610.12388"},
    {"label": "Demo", "icon": "huggingface", "href": None, "disabled": True},
    {"label": "Code", "icon": "github", "href": "https://github.com/facebookresearch/GenIA"},
    # {"label": "Data", "icon": "data", "href": None, "disabled": True},  # hidden for now
]

# The main page's 3D gallery (#gallery3d-sec): one scene per entry, its files
# staged under assets/gallery3d/<scene>/ by prepare_gallery3d.py, which prints
# the entry to paste here. Passed through to data.js as-is.
#   title    tile badge, and the item's name in the viewer's Scene panel
#   views    input-view thumbnails: the scene's tile in the strip
#   assets   one or more .ply / .splat / .spz / .ksplat / .glb paths
#   temporal true: several assets are the frames of one 4D item (a timeline)
#   tracks   optional 3D point tracks (.npz), shown as their own item
#   colmap   optional COLMAP export (<dir>/sparse/000 + <dir>/images), shown as
#            its camera frustums and images (its points are hidden; the script
#            crops them, since the viewer still frames the view from them)
#   view     optional starting viewpoint {position, target}, in the COLMAP frame
GALLERY3D = [
    # single image
    {"title": "lab_duo", "assets": ["assets/gallery3d/lab_duo/world.compressed.ply"],
     "colmap": "assets/gallery3d/lab_duo/colmap",
     "views": ["assets/gallery3d/lab_duo/views/train_000.webp"],
     "view": {"position": [0.009, -0.7506, -0.9772], "target": [-0.0229, 0.0441, 2.4447]}},
    {"title": "office_buddies", "assets": ["assets/gallery3d/office_buddies/world.compressed.ply"],
     "colmap": "assets/gallery3d/office_buddies/colmap",
     "views": ["assets/gallery3d/office_buddies/views/train_000.webp"],
     "view": {"position": [-0.0071, -0.25, -0.3282], "target": [0.0174, 0.0081, 0.8233]}},
    {"title": "sloth", "assets": ["assets/gallery3d/sloth/world.compressed.ply"],
     "colmap": "assets/gallery3d/sloth/colmap",
     "views": ["assets/gallery3d/sloth/views/train_000.webp"],
     "view": {"position": [-0.0018, -0.2766, -0.382], "target": [0.0047, -0.0285, 0.9608]}},
    {"title": "lab_turtle", "assets": ["assets/gallery3d/lab_turtle/world.compressed.ply"],
     "colmap": "assets/gallery3d/lab_turtle/colmap",
     "views": ["assets/gallery3d/lab_turtle/views/train_000.webp"],
     "view": {"position": [0.0139, -0.5497, -0.7241], "target": [-0.0348, 0.0151, 1.8134]}},
    {"title": "stutty", "assets": ["assets/gallery3d/stutty/world.compressed.ply"],
     "colmap": "assets/gallery3d/stutty/colmap",
     "views": ["assets/gallery3d/stutty/views/train_000.webp"],
     "view": {"position": [-0.0095, -0.5211, -0.767], "target": [0.0241, -0.1416, 1.9237]}},
    # multi-view
    {"title": "lab_duck", "assets": ["assets/gallery3d/lab_duck/world.compressed.ply"],
     "colmap": "assets/gallery3d/lab_duck/colmap",
     "views": ["assets/gallery3d/lab_duck/views/train_v00_000.webp",
               "assets/gallery3d/lab_duck/views/train_v01_000.webp"],
     "view": {"position": [0.0, -0.6851, -0.8321], "target": [-0.0005, 0.1451, 2.0892]}},
    {"title": "pablo", "assets": ["assets/gallery3d/pablo/world.compressed.ply"],
     "colmap": "assets/gallery3d/pablo/colmap",
     "views": ["assets/gallery3d/pablo/views/train_v00_000.webp",
               "assets/gallery3d/pablo/views/train_v01_000.webp"],
     "view": {"position": [0.0262, -0.407, -0.5925], "target": [-0.0654, -0.1002, 1.4874]}},
    {"title": "stuffed_toy", "assets": ["assets/gallery3d/stuffed_toy/world.compressed.ply"],
     "colmap": "assets/gallery3d/stuffed_toy/colmap",
     "views": ["assets/gallery3d/stuffed_toy/views/train_v00_000.webp",
               "assets/gallery3d/stuffed_toy/views/train_v01_000.webp",
               "assets/gallery3d/stuffed_toy/views/train_v02_000.webp",
               "assets/gallery3d/stuffed_toy/views/train_v03_000.webp",
               "assets/gallery3d/stuffed_toy/views/train_v04_000.webp"],
     "view": {"position": [0.0124, -0.2925, -0.3904], "target": [-0.03, -0.0018, 0.9767]}},
    # dynamic: first, middle and last frame
    {"title": "camel", "assets": ["assets/gallery3d/camel/world/000.compressed.ply",
                                  "assets/gallery3d/camel/world/008.compressed.ply",
                                  "assets/gallery3d/camel/world/015.compressed.ply"],
     "temporal": True, "tracks": "assets/gallery3d/camel/tracks_3d.npz",
     "colmap": "assets/gallery3d/camel/colmap",
     "views": ["assets/gallery3d/camel/views/train_000.webp",
               "assets/gallery3d/camel/views/train_008.webp",
               "assets/gallery3d/camel/views/train_015.webp"],
     "view": {"position": [-0.2276, -2.3717, -3.1412], "target": [0.5882, 0.0588, 7.8018]}},
]

# Where the published page loads its media from: the public URL of the Google
# Cloud Storage folder holding assets/, i.e.
# https://storage.googleapis.com/<bucket>/<prefix>. Written into data.js as
# `assetBase`; js/app.js prefixes every `assets/...` path with it, and
# upload_assets.py reads it back to know where to upload. None = serve the
# local assets/ folder (the page then only works where that folder exists).
ASSET_BASE_URL = "https://storage.googleapis.com/geniadata"

# Order + presentation of the per-dataset comparison blocks, and the selectors
# above each one. Methods are named by their **registry label** — the exact
# string in mapping.json's `benchmark.baselines` / `davis_sources`, which is
# also the paper's table-row label — so what a chip offers and what the tables
# score cannot drift apart. A declared label with no registry entry (or no run
# at that view count, or no orbit render yet) is offered greyed out.
#
#   `variants` — the "Show" row: each entry maps view count -> registry label
#                plus the short chip text; the full label is drawn in the pill.
#   `baselines`— the "vs" row, per view count, in chip order.
#
# The view axis is derived: the counts a block offers are the keys of these two
# dicts, and a block spanning more than one count gets the Views control (and
# view-qualified pill labels).
# Table 1's dynamic rows (TripoSplat and V2M4 were dropped from the paper's
# dynamic comparison; Any4D is tracks-only, so its chip stays greyed out).
DYNAMIC_BASELINES = ["SAM3D", "Lift4D", "HiMoR", "Any4D"]


def _ttr_variants(labels: dict[int, str]) -> list[dict]:
    """Both TTR settings of one "Ours" configuration, as Table 1 reports them.
    `labels` holds the w/-TTR label per view count; its w/o-TTR twin is the same
    row with the suffix swapped (mapping.json carries it as its own entry,
    pointing at the `_no_finetune` run of the family)."""
    return [
        {"labels": labels, "display": "Ours w/ TTR"},
        {"labels": {n: l.replace("w/ TTR", "w/o TTR") for n, l in labels.items()},
         "display": "Ours w/o TTR"},
    ]


TTR_CHIPS = _ttr_variants({1: "Ours (ActionMesh) w/ TTR"})


def _static_meta(counts: tuple[int, ...]) -> dict:
    """The shared shape of the two static multi-view benchmarks: the two "Ours"
    chips per count — `_ours_label` picks the backbone the tables name at that
    count — against the per-count baseline lists (Table 1's rows per view group;
    ReconViaGen is reported at its own view count only, hence top group only)."""
    per_count = ["SAM3D", "TripoSplat", "RecGen", "CUPID"], \
                ["STREAM3D", "MV-SAM3D", "RecGen"], \
                ["MV-SAM3D", "Depth-3DGS", "ReconViaGen", "STREAM3D"]
    return {
        "variants": _ttr_variants({n: _ours_label("SAM3D", str(n)) for n in counts}),
        "baselines": dict(zip(counts, per_count)),
    }


# Scenes hidden from the page's interactive comparisons/galleries -- a
# page-local exclusion, NOT mapping.json's scene_allowlists (which the
# paper's own tables and evaluators read too; excluding a scene there would
# silently change reported numbers rather than just what this page shows).
EXCLUDED_SCENES = {
    "gso": {"cream", "mario", "school_bus1"},
}

# Scenes pulled OUT of the interactive comparisons/galleries and INTO the
# "Limitations and failure cases" section instead (same media, same chip
# selection -- just a different scene list on the same per-dataset block
# shape, built alongside `comparisons` in `main`).
LIMITATION_SCENES = {
    "davis_actionmesh": {"bear"},
    "oursactionbench": {"000-007_84496dfddf6046ce9ec9fddace6f5ed3"},
}

# Scenes pinned to the FRONT of a dataset's interactive comparisons, in the
# order listed here; every other scene still follows in its usual
# alphabetical order behind them.
FEATURED_SCENES = {
    "gso": ["school_bus2"],
    "davis_actionmesh": ["horsejump-low", "rhino", "cows", "horsejump-high",
                         "camel", "blackswan", "car-roundabout", "hike",
                         "mallard-water"],
}


def scene_sort_key(dataset: str, scene: str):
    featured = FEATURED_SCENES.get(dataset, [])
    return (featured.index(scene), scene) if scene in featured else (len(featured), scene)

DATASETS = {
    "davis_actionmesh": {
        "title": "DAVIS",
        "order": 0,
        "type": "dynamic",
        "variants": TTR_CHIPS,
        "baselines": {1: DYNAMIC_BASELINES},
    },
    "co3d": {
        "title": "CO3D",
        "order": 1,
        "type": "static",
        "default_views": 1,
        **_static_meta((1, 2, 4)),
    },
    "oursactionbench": {
        "title": "ActionBench",
        "order": 2,
        "type": "dynamic",
        "variants": TTR_CHIPS,
        "baselines": {1: DYNAMIC_BASELINES},
    },
    "gso": {
        "title": "GSO-30",
        "order": 3,
        "type": "static",
        "default_views": 1,
        **_static_meta((1, 2, 5)),
    },
}

# Methods the page shows whose table rows are still hand-maintained, so they
# have no mapping.json entry yet — same schema as a registry entry. Move each
# into `benchmark.davis_sources` once its row is synced (note that lets
# sync_benchs OVERWRITE the hand row from the run's eval dirs). Empty today:
# Lift4D and HiMoR, which replaced DG-Mesh and OriGS here, are registry entries.
PAGE_RUNS = []

# --------------------------------------------------------------------------- #
# Registry resolution — the label -> run-dir logic mirrors sync_benchs'
# `_baseline_rows` over the same three entry shapes.
# --------------------------------------------------------------------------- #

# page dataset id (the results subdir) -> mapping.json dataset key
REG_KEY = {_paths.dataset_subdir(k): k for k in ("gso", "oab", "co3d", "davis")}

_ORBIT_CACHE: dict[tuple[str, str], dict[str, Path]] = {}


def scene_orbits(run_dir: str, subdir: str) -> dict[str, Path]:
    """{scene: newest orbit render} inside one declared run dir, memoized.

    The page's ONE definition of "the orbit for this (run, scene)" — the
    comparison chips and the completion viewer both resolve through it, so they
    cannot disagree about which timestamp is newest.
    """
    key = (run_dir, subdir)
    if key not in _ORBIT_CACHE:
        best: dict[str, tuple[str, Path]] = {}
        for mp4 in _paths.results_path(run_dir, subdir).glob("*/*/final/viz/orbit/orbit.mp4"):
            scene, ts = mp4.parts[-6], mp4.parts[-5]
            if scene not in best or ts > best[scene][0]:
                best[scene] = (ts, mp4)
        _ORBIT_CACHE[key] = {s: p for s, (_, p) in best.items()}
    return _ORBIT_CACHE[key]


def registry_entries(reg_key: str) -> dict[str, dict]:
    """label -> entry for one dataset. DAVIS rows live in `davis_sources`,
    which overrides `benchmark.baselines` for labels present in both (e.g.
    "Ours (ActionMesh) w/ TTR" is a different run on DAVIS than on ActionBench).
    PAGE_RUNS fills labels the registry does not carry yet; the registry wins.
    """
    entries: dict[str, dict] = {e["label"]: e for e in PAGE_RUNS}
    for b in _paths.baselines():
        entries[b["label"]] = b
    if reg_key == "davis":
        for d in _paths.davis_sources():
            entries[d["label"]] = d
    return entries


def media_key(label: str, views: int) -> str:
    """Stable asset/manifest key for one (registry label, view count)."""
    return re.sub(r"[^a-z0-9]+", "_", label.lower()).strip("_") + f"_{views}v"


def pill_label(label: str) -> str:
    """The viewer's overlay pill for one chip. An "Ours (<backbone>) ..."
    label always reads as plain "Ours" here — SAM3D/MV-SAM3D/ActionMesh name
    the geometry source, an implementation detail the reader does not need in
    order to compare configurations, and it made every dynamic-dataset pill
    read "Ours (ActionMesh)" regardless of which step of the ladder was
    selected. The view count is never spelled out either — the Views chip
    above the panel already says which count is selected. A non-Ours label
    (no registry match) passes through unchanged."""
    m = regen_tables.OURS_LABEL.match(label)   # the grammar's one parser
    if not m:
        return label
    parts = ["Ours"]
    if m.group(2):
        parts.append(m.group(2))
    return ", ".join(parts)


def build_chips(meta: dict, reg_key: str, subdir: str) -> tuple[list[dict], list[dict]]:
    """Resolve one dataset's declared selectors against the registry + disk.

    Returns (references, baselines): chip dicts carrying the manifest fields
    (`id`, `views`, `label`, `display`, `available`) plus internal ones the
    report and the encode step read (`orbits` scene->src path, `reason`).
    """
    entries = registry_entries(reg_key)
    counts = sorted({n for spec in meta.get("variants", []) for n in spec["labels"]} |
                    set(meta.get("baselines", {})))

    def orbits_under(run_dir: str) -> dict[str, Path]:
        """This dataset's orbits under one run dir, restricted to the scenes the
        paper reports (mapping.json `scene_allowlists`, e.g. ActionBench's 6 of
        9). Filtered HERE rather than downstream so a chip's `available` flag,
        its greyed reason, the report's coverage counts and the encode set all
        describe the same scene set."""
        return {s: p for s, p in scene_orbits(run_dir, subdir).items()
                if _paths.scene_allowed(reg_key, s)}

    def ladder_run(label: str, n: int) -> str | None:
        """The tables' second "Ours" source: the ablation ladder's full-config
        cell (what sync_benchs' `_ours_rows` reads), for the label that ladder
        fills — `_ours_label(ablation_backbone, views)`. Covers the (label,
        view count) pairs `benchmark.baselines` does not, e.g. CO3D at 4 views.
        """
        backbone = _paths.ablation_backbone().get(reg_key)
        if backbone is None or label != _ours_label(backbone, str(n)):
            return None
        return _paths.ours_run(reg_key, FULL_CONFIG, str(n))

    def make_chip(label: str, n: int) -> dict:
        # Both sources may name a dir; like the tables' upsert order, the one
        # that actually has data (an orbit) wins, then the registry entry.
        entry = entries.get(label)
        candidates = [d for d in (_paths.run_dirs(entry, reg_key).get(n) if entry else None,
                                  ladder_run(label, n)) if d]
        run_dir = next((d for d in candidates if orbits_under(d)), None) \
            or (candidates[0] if candidates else None)
        if run_dir is None:
            reason = ("not in mapping.json" if entry is None
                      else f"no registered run at {n} view(s)")
            orbits: dict[str, Path] = {}
        else:
            orbits = orbits_under(run_dir)
            reason = None if orbits else f"no orbit under {run_dir}/{subdir}"
        return {"id": media_key(label, n), "views": n, "label": label,
                "orbits": orbits, "reason": reason, "available": bool(orbits)}

    refs, bases = [], []
    for n in counts:
        for spec in meta.get("variants", []):
            if n not in spec["labels"]:
                continue
            label = spec["labels"][n]
            chip = make_chip(label, n)
            chip["label"] = pill_label(label)
            chip["display"] = spec["display"]
            refs.append(chip)
        for label in meta.get("baselines", {}).get(n, []):
            bases.append(make_chip(label, n))

    ids = [c["id"] for c in refs + bases]
    clashes = {i for i in ids if ids.count(i) > 1}
    if clashes:   # two labels slugging to one id would silently merge their chips
        raise SystemExit(f"media_key collision in {subdir}: {sorted(clashes)}")
    return refs, bases


# --------------------------------------------------------------------------- #
# The paper's own headline figures, shown here the way the paper shows them
# (paths under _paths.FIGURES, mapping.json's figure root).
# `figures/teaser.tex` is an Ours-over-Baseline grid of five scene plates in
# three equal-width setting columns — the two static settings hold two
# half-width plates each, the dynamic one a single full-width plate — and
# `figures/pipeline.tex` is a narrow/wide + wide/narrow arrangement of four
# diagrams. Both are mirrored below (widths as grid-fraction weights, in that
# reading order).
#
# One entry per teaser plate: the scene dir under figures/teaser/imgs/, the
# header it sits under (None = shares the previous plate's, so a setting's
# header centres over its pair), the baseline shown under Ours — its badge text
# and its file stem in the scene dir — the plate's share of the grid, and the
# strip of INPUT thumbnails beneath it: the observations that plate was
# reconstructed from (one image, two views, or a frame range), as
# `\tinputsmono/mv/dyn` list them. `None` inside a strip is the paper's
# `\tellipsis` — the frames between the two it shows.
TEASER_COLUMNS = [
    ("office_buddies", "Monocular-Static",  "SAM3D",    "sam3d",   16.7, ("0000",)),
    ("sloth",          None,                "CUPID",    "cupid",   16.7, ("0000",)),
    ("pablo_small",    "Multiview-Static",  "MV-SAM3D", "mvsam3d", 16.7, ("0000", "0001")),
    ("lab_duck",       None,                "RecGen",   "recgen",  16.7, ("0000", "0001")),
    ("horsejump-high", "Monocular-Dynamic", "Lift4D",   "lift4d",  33.2,
     ("0000", None, "0015")),
]
TEASER_ROWS = ["Ours", "Baseline"]
# The stage titles are drawn inside the re-exported diagram PDFs; these page-side
# captions repeat sec/method.tex's subsection names. Widths follow the panels'
# cropped native aspect (0.40/0.60 + 0.60/0.40 of a row).
PIPELINE_PANELS = [
    ("stage-1", "1. <em>Optional</em> Shape Injection", 39),
    ("stage-2", "2. Pose Prediction", 60),
    ("stage-3", "3. Appearance Prediction", 60),
    ("stage-4", "4. <em>Optional</em> Test-time Refinement", 39),
]
# tabs/qualitative_completion.tex's row, and figures/time_vs_quality.tex's plot.
COMPLETION_PANELS = [("0000", "Input"), ("sam3d", "SAM3D"),
                     ("triposplat", "TripoSplat"), ("cupid", "CUPID"),
                     ("ours", "Ours w/ TTR")]
# The interactive twin of that figure: one orbit render per method on the same
# scene, mounted as a method-tab viewer the reader rotates by dragging (the
# still row is the fallback while a run is missing). Each entry names the
# results run dir its orbit comes from — the single-image qualitative lane
# (`dataset=image`, launchers run_mono_{ours,sam3d}_images.py and
# run_baseline_{cupid,triposplat}_image.py).
# The appearance ladder (fig:gso_appearance_qual_main): one GSO scene, the
# cumulative components across the columns, a train and a held-out view down
# the rows. Same shape as the teaser — headed columns, labelled rows — so it
# is emitted in that schema and drawn by the same builder.
#
# The paper's own tiles have the PSNR/co-PSNR badge burned into the pixels
# (update_qualitative.py's _draw_label); this page instead renders the RAW
# reconstruction and draws the number as an HTML overlay, sourced from the
# SAME eval JSON the paper's csv/gso_appearance.csv updater reads
# (`_paths.ours_run` / `ours_eval_dir`, `update_gso_appearance.py`'s own
# resolvers) — never OCR'd off a rendered badge, and free to update the moment
# a re-run changes the metric without anyone re-rendering a tile.
APPEARANCE_SCENE = "chicken"
APPEARANCE_TRAIN_FRAME = "000"
APPEARANCE_TEST_FRAME = "013"
# (label, filename slug, (init, attn, rg, ttt)) — the same 4-tuple
# `_paths.ours_run`/`ours_eval_dir` key ablation cells by.
APPEARANCE_COLUMNS = [
    ("SAM3D", "base",   ("on", "off", "off", "off")),
    ("+Attn", "attn",   ("on", "on",  "off", "off")),
    ("+RG",   "attnrg", ("on", "on",  "on",  "off")),
    ("+TTR",  "full",   ("on", "on",  "on",  "on")),
]
# (row label, metrics2D_{split} split key, metric field) — the caption's own
# choice: PSNR on the training view, co-PSNR (the covisible subset) on test.
APPEARANCE_ROWS = [("Train", "train", "psnr"), ("Test", "test", "co_psnr")]

COMPLETION_SCENE = "lab_turtle"
# Lanes are named plainly: `_paths.results_path` accepts an archived
# `<name>_legacy` dir for a lane named without it (and prefers the plain name
# when both exist), so a re-run does not need an edit here.
COMPLETION_VIEWER = [
    ("sam3d",       "SAM3D",       "mono_sam3d/base"),
    ("triposplat",  "TripoSplat",  "method_triposplat"),
    ("cupid",       "CUPID",       "method_cupid"),
    ("ours",        "Ours w/ TTR", "mono_ours/mono_static"),
]


# --------------------------------------------------------------------------- #
# The supplementary's eight qualitative figures (sec/supplementary.tex), on
# their own page. Each is transcribed from its tabs/qualitative_*.tex: the same
# scenes in the same order, the same method columns in the same order, and the
# same tiles -- `update_qualitative.py` already syncs those into
# figures/<dataset>/<scene>/<sub>/, badge burned in and (on oab/davis) cropped
# per row, so the page consumes THEM rather than re-deriving anything.
#
# The paper's header is two-level: a view-count band over the method names.
# `buildTeaserGrid` draws one header row, so a figure whose methods split by
# input-view count becomes one GRID PER GROUP, the band as the grid's caption --
# which also keeps any single grid under ~9 columns instead of 19.
#
# A group is (caption, [(header, cell-stem), ...]); `None` as a header merges
# the column into the previous one's span, which is how a method that owns
# several frames (oab's three timestamps) reproduces the paper's \mhead.
_OURS_N = "Ours w/o TTR"          # \ourslabelnew
_OURS_T = "Ours w/ TTR"           # \ourslabel

_GSO_GROUPS = [
    ("1 view (0)", [("SAM3D", "sam3d_1v"), ("TripoSplat", "triposplat_1v"),
                    ("TRELLISv2", "trellis_1v"), ("Pixal3D", "pixal3d_1v"),
                    ("RecGen", "recgen_1v"), ("CUPID", "cupid_1v"),
                    (_OURS_N, "ours_sam3d_1v_nottr"), (_OURS_T, "ours_sam3d_1v")]),
    ("2 views (0–1)", [("MV-SAM3D", "mvsam3d_2v"), ("RecGen", "recgen_2v"),
                       (_OURS_N, "ours_mvsam3d_2v_nottr"), (_OURS_T, "ours_mvsam3d_2v")]),
    ("5 views (0–4)", [("MV-SAM3D", "mvsam3d_5v"), ("Depth-3DGS", "depth3dgs_5v"),
                       ("STREAM3D", "stream3d_5v"), ("ReconViaGen", "reconviagen_5v"),
                       (_OURS_N, "ours_mvsam3d_5v_nottr"), (_OURS_T, "ours_mvsam3d_5v")]),
]
# CO3D runs the same ladder at 4 views instead of 5 (LaRa ships 4 input views).
_CO3D_GROUPS = [
    (_GSO_GROUPS[0][0], _GSO_GROUPS[0][1]),
    (_GSO_GROUPS[1][0], _GSO_GROUPS[1][1]),
    ("4 views (0–3)", [("MV-SAM3D", "mvsam3d_4v"), ("Depth-3DGS", "depth3dgs_4v"),
                       ("STREAM3D", "stream3d_4v"), ("ReconViaGen", "reconviagen_4v"),
                       (_OURS_N, "ours_mvsam3d_4v_nottr"), (_OURS_T, "ours_mvsam3d_4v")]),
]
_LADDER = [("Base", "base"), ("+Attn", "attn"), ("+RG", "attnrg"), ("+TTR", "full")]


def _frames(steps, frames):
    """One column per (step, frame), the header on the first of each block so
    `buildTeaserGrid` spans it across the rest -- the paper's \\mhead."""
    return [(label if i == 0 else None, f"{stem}_{f}")
            for label, stem in steps for i, f in enumerate(frames)]


# (id, page, mapping.json spec, title, dataset dir, scenes, groups, gt, pool, blurb).
# `title` is the benchmark alone on both pages: which KIND of figure it is comes
# from the page it is on, so "GSO-30 - appearance ladder" only repeated the
# heading above it.
# `page` picks which HTML file shows it -- the four cross-method comparisons on
# qualitative.html, the four appearance ladders on ablations.html, mirroring the
# supplementary's own "Additional Results" / "Extended ablations" split. Each
# page's mount div names the group it wants (`data-figures`).
# `gt` is (train-stem, test-stem); a None test stem is the paper's blank tile
# (DAVIS has no ground truth for a synthesized view). `pool` is (subdir, stems)
# for the figure's input-view column, drawn as its own small grid.
QUALITATIVE_FIGURES = [
    {"id": "qual-davis", "page": "comparisons", "spec": "davis", "title": "DAVIS", "dataset": "davis",
     "scenes": [("Horsejump-High", "horsejump-high"), ("Hike", "hike"),
                ("Horsejump-Low", "horsejump-low")],
     "groups": [("Frame 15", [("SAM3D", "sam3d_015"), ("Lift4D", "lift4d_015"),
                        ("HiMoR", "himor_015"), ("Any4D", "any4d_015"),
                        (_OURS_N, "ours_nottr_015"), (_OURS_T, "ours_015")])],
     "gt": ("00015_crop", None),
     "pool": ("pool", ["00000", "00005", "00010", "00015"]),
     "blurb": "DAVIS is monocular, so the second row is a synthesized novel view rather "
              "than a held-out camera, and no ground truth exists for it. Train view is badged with PSNR, test view with CLIP-I."},
    {"id": "qual-co3d", "page": "comparisons", "spec": "co3d", "title": "CO3D", "dataset": "co3d",
     "scenes": [("Scene 1", "123_14363_28981"), ("Scene 2", "34_1479_4753"),
                ("Scene 3", "374_42274_84517"), ("Scene 4", "391_47032_93657")],
     "groups": _CO3D_GROUPS, "gt": ("train", "test"),
     "pool": ("pool", ["000", "001", "002", "003"]),
     "blurb": "Badged with PSNR on the train view and CLIP-I on the held-out test view."},
    {"id": "qual-oab", "page": "comparisons", "spec": "oab", "title": "ActionBench", "dataset": "oursactionbench",
     "scenes": [("Scene 1", "000-007_17ed60fbe8af47b69cbf45b97730e0ab"),
                ("Scene 2", "000-007_84496dfddf6046ce9ec9fddace6f5ed3"),
                ("Scene 3", "000-001_095641b042f049c3aa86792836f8993d"),
                ("Scene 4", "000-008_95878b202a624ae88a021fe5ca99b335"),
                ("Scene 5", "000-004_28e4d4e402a64d5697e7eeae5212bd32")],
     "groups": [("Timestamps 0, 8, 13", _frames([("SAM3D", "sam3d"), ("Lift4D", "lift4d"),
                                ("HiMoR", "himor"),
                                (_OURS_N, "ours_actionmesh_nottr"),
                                (_OURS_T, "ours_actionmesh")], ["00", "08", "13"]))],
     "gt": ("{f}_train_crop", "{f}_crop"), "gt_frames": ["00", "08", "13"],
     "pool": ("imgs", ["00", "04", "07", "11", "15"]),
     "blurb": "Three timestamps per method, badged with PSNR on the train view and "
              "CLIP-I on the held-out one."},
    {"id": "qual-gso", "page": "comparisons", "spec": "gso", "title": "GSO-30", "dataset": "gso",
     "scenes": [("Backpack", "backpack"), ("Turtle", "turtle"),
                ("School Bus 2", "school_bus2"), ("Chicken", "chicken"),
                ("Sorter", "sorter")],
     "groups": _GSO_GROUPS, "gt": ("train", "test"),
     "pool": ("gt", ["000", "001", "002", "003", "004"]),
     "blurb": "Each render is badged with its per-scene PSNR on the train view and "
              "CLIP-I on the held-out test view."},

    {"id": "qual-davis-app", "page": "ablations", "spec": "davis_app", "title": "DAVIS", "dataset": "davis",
     "scenes": [("Horsejump-High", "horsejump-high"), ("Hike", "hike"),
                ("Horsejump-Low", "horsejump-low")],
     "groups": [("Frame 8", [(l, f"{s}_008") for l, s in _LADDER])],
     "gt": ("00008", None),
     "pool": ("pool", ["00000", "00005", "00010", "00015"]),
     "blurb": "DAVIS is monocular, so the second row is a synthesized novel view rather "
              "than a held-out camera, and no ground truth exists for it. Train view is badged with PSNR, test view with CLIP-I."},
    {"id": "qual-co3d-app", "page": "ablations", "spec": "co3d_app", "title": "CO3D", "dataset": "co3d",
     "scenes": [("Scene 1", "123_14363_28981"), ("Scene 2", "247_26469_51778"),
                ("Scene 3", "34_1479_4753"), ("Scene 4", "374_42274_84517")],
     "groups": [("1 view (0)", [(l, f"{s}_1v") for l, s in _LADDER]),
                ("4 views (0–3)", [(l, f"{s}_4v") for l, s in _LADDER])],
     "gt": ("train", "test"), "pool": ("pool", ["000", "001", "002", "003"]),
     "blurb": "Badged with PSNR on the train view and CLIP-I on the held-out one."},
    {"id": "qual-oab-app", "page": "ablations", "spec": "oab_app", "title": "ActionBench",
     "dataset": "oursactionbench",
     "scenes": [("Scene 1", "000-004_59599287510e43649360b850d56119d5"),
                ("Scene 2", "000-007_84496dfddf6046ce9ec9fddace6f5ed3"),
                ("Scene 3", "000-001_095641b042f049c3aa86792836f8993d"),
                ("Scene 4", "000-008_95878b202a624ae88a021fe5ca99b335"),
                ("Scene 5", "000-004_28e4d4e402a64d5697e7eeae5212bd32")],
     "groups": [("Timestamps 0, 8", _frames(_LADDER, ["00", "08"]))],
     "gt": ("{f}_train", "{f}"), "gt_frames": ["00", "08"],
     "pool": ("imgs", ["00", "04", "07", "11", "15"]),
     "blurb": "Two timestamps per step. Badged with PSNR on the train view and co-PSNR "
              "on the held-out one."},
    {"id": "qual-gso-app", "page": "ablations", "spec": "gso_app", "title": "GSO-30", "dataset": "gso",
     "scenes": [("Backpack", "backpack"), ("Turtle", "turtle"),
                ("School Bus 2", "school_bus2"), ("Chicken", "chicken"),
                ("Sorter", "sorter")],
     "groups": [("1 view (0)", [(l, f"{s}_1v") for l, s in _LADDER]),
                ("5 views (0–4)", [(l, f"{s}_5v") for l, s in _LADDER])],
     "gt": ("train", "test"), "pool": ("gt", ["000", "001", "002", "003", "004"]),
     "blurb": "Badged with PSNR on the train view and co-PSNR on the held-out one."},
]

# Input-view thumbnails for the interactive comparisons on index.html. Sourced
# from each dataset's own raw frames under data/ -- NOT paper/figures/, which
# only holds the handful of scenes QUALITATIVE_FIGURES hand-picks (5/30 on
# GSO, 4/10 on DAVIS) and left every other scene's tile reading "no input
# views". ActionBench is the one exception: QUALITATIVE_FIGURES already covers
# all 6 of its scenes via figures/, so it keeps reading from there rather than
# guessing at a raw layout nothing else here needs.
def comparison_pool(dataset: str, scene: str) -> list[tuple[str, Path]]:
    """(stem, src) pairs of one scene's input-view thumbnails, newest-source
    first. A missing file just shortens the list, same as a missing orbit
    drops a chip from `media`."""
    if dataset == "davis_actionmesh":
        meta_path = _paths.DATA / "davis_actionmesh" / scene / "metadata.json"
        if not meta_path.is_file():
            return []
        meta = json.loads(meta_path.read_text())
        frames = meta["selected_frames"]
        images_dir = _paths.REPO / meta["images_dir"]
        n = len(frames)
        idxs = sorted({0, n // 3, 2 * n // 3, n - 1})
        return [(Path(frames[i]).stem, images_dir / frames[i]) for i in idxs]
    if dataset == "co3d":
        root = _paths.DATA / "co3d_lara" / scene
        return [(f"{i:03d}", root / f"{i:03d}.png") for i in range(4)]
    if dataset == "gso":
        root = _paths.DATA / "gso30" / scene / "render_mvs_25" / "model"
        return [(f"{i:03d}", root / f"{i:03d}.png") for i in range(5)]
    if dataset == "oursactionbench":
        root = _paths.FIGURES / "oursactionbench" / scene / "imgs"
        return [(stem, root / f"{stem}.png")
                for stem in ("00", "04", "07", "11", "15")]
    return []


def rasterize_pdf(src: Path, dst: Path, dpi: int) -> bool:
    """The pipeline diagrams are vector; the page needs pixels. False without gs."""
    gs = shutil.which("gs")
    if gs is None:
        return False
    subprocess.run([gs, "-q", "-sDEVICE=pngalpha", f"-r{dpi}", "-dNOPAUSE", "-dBATCH",
                    "-dUseCropBox", "-o", str(dst), str(src)], check=True)
    return True


def collect_static_figures(out: Path, *, dry_run: bool, force: bool,
                           dpi: int = 200) -> tuple[dict, int]:
    """Teaser, pipeline, completion + runtime panels, straight from the
    registry's figure root."""
    static: dict = {}
    n = 0

    def harvest_still(src: Path, name: str) -> str | None:
        """Encode one still into assets/static/, returning its manifest path."""
        nonlocal n
        if not src.is_file():
            print(f"! no figure panel at {src}")
            return None
        rel = still_dest(Path("assets/static") / name)
        if write_asset(src, out / rel, dry_run=dry_run, force=force):
            n += 1
        return rel.as_posix()

    # Teaser: the ten plates plus the structure that arranges them, so the page can
    # rebuild the paper's grid (headers, rotated row labels, badges) in the DOM rather
    # than shipping a flattened screenshot that no longer exists.
    cells: dict[str, list] = {}
    for row_label in TEASER_ROWS:
        row_cells: list = []
        for scene, _, _, base_stem, _, _ in TEASER_COLUMNS:
            stem = "ours" if row_label == "Ours" else base_stem
            row_cells.append(harvest_still(_paths.FIGURES / "teaser" / "imgs" / scene / f"{stem}.png",
                                           f"teaser_{scene}_{row_label.lower()}.png"))
        cells[row_label] = row_cells

    def input_strip(scene: str, stems: tuple) -> list:
        """One column's input thumbnails, `None` kept where the paper draws its
        ellipsis (so the page shows the same gap rather than implying the two
        frames are adjacent)."""
        return [None if s is None else
                harvest_still(_paths.FIGURES / "teaser" / "imgs" / scene / f"{s}.png",
                              f"teaser_{scene}_input_{s}.png")
                for s in stems]

    if any(any(c) for c in cells.values()):
        static["teaser"] = {
            "columns": [{"header": header, "badge": badge, "width": width,
                         "inputs": input_strip(scene, stems)}
                        for scene, header, badge, _, width, stems in TEASER_COLUMNS],
            "rows": [{"label": label, "cells": cells[label]} for label in TEASER_ROWS],
        }

    # The completion figure's five tiles (fig:completion), same schema as the
    # pipeline panels so the page mounts them with one builder.
    completion = []
    for stem, title in COMPLETION_PANELS:
        rel = harvest_still(_paths.FIGURES / "completion" / COMPLETION_SCENE / f"{stem}.png",
                            f"completion_{stem}.png")
        if rel:
            completion.append({"title": title, "src": rel, "width": 19})
    if completion:
        static["completion"] = completion

    # The runtime-vs-quality plot (fig:time_vs_quality) ships as a pre-rendered PNG.
    runtime = harvest_still(_paths.FIGURES / "diagrams" / "time_vs_quality.png",
                            "time_vs_quality.png")
    if runtime:
        static["runtime"] = runtime

    def appearance_metric(settings: tuple, split: str, field: str) -> str | None:
        """This ladder cell's number, straight from its eval JSON — the exact
        value `update_gso_appearance.py` writes into csv/gso_appearance.csv."""
        eval_dir = _paths.ours_eval_dir("gso", settings, "1")
        if eval_dir is None:
            return None
        path = eval_dir / f"metrics2D_{split}.json"
        if not path.is_file():
            return None
        with open(path) as f:
            val = json.load(f).get("scenes", {}).get(APPEARANCE_SCENE, {}).get(field)
        return None if val is None else f"{val:.2f}"

    def appearance_render(settings: tuple, split: str, frame: str) -> Path | None:
        """The newest RAW (unbadged) render for one ladder cell."""
        run = _paths.ours_run("gso", settings, "1")
        if run is None:
            return None
        sub = "renders_train" if split == "train" else "renders_test"
        hits = sorted(_paths.results_path(run, "gso").glob(
            f"{APPEARANCE_SCENE}/*/final/{sub}/{frame}.png"))
        return hits[-1] if hits else None

    # The appearance ladder, in the teaser's grid schema. `zoom: "all"` scopes
    # the hover magnifier across the WHOLE grid rather than down one column: the
    # point of a ladder is to compare the same detail between its steps. A cell
    # is `{src, badge}` rather than a bare path — the metric renders as an HTML
    # overlay, decoupled from whatever the render itself shows.
    gt_dir = _paths.FIGURES / "gso" / APPEARANCE_SCENE / "gt"
    app_rows = []
    for row_label, split, field in APPEARANCE_ROWS:
        frame = APPEARANCE_TRAIN_FRAME if split == "train" else APPEARANCE_TEST_FRAME
        cells = []
        for label, slug, settings in APPEARANCE_COLUMNS:
            src = appearance_render(settings, split, frame)
            if src is None:
                print(f"! no appearance render for {label} ({split})")
                cells.append(None)
                continue
            rel = harvest_still(src, f"appearance_{slug}_{row_label.lower()}.png")
            badge = appearance_metric(settings, split, field)
            cells.append({"src": rel, "badge": badge} if rel else None)
        cells.append(harvest_still(gt_dir / f"{row_label.lower()}.jpg",
                                   f"appearance_gt_{row_label.lower()}.png"))
        app_rows.append({"label": row_label, "cells": cells})
    if any(any(r["cells"]) for r in app_rows):
        static["appearance"] = {
            "columns": [{"header": label, "width": 20}
                        for label, _, _ in APPEARANCE_COLUMNS] + [{"header": "GT", "width": 20}],
            "rows": app_rows,
            "zoom": "all",
            "plateHeight": 1,     # square renders; nothing to crop away
        }

    panels = []
    for stem, title, width in PIPELINE_PANELS:
        pdf = _paths.FIGURES / "diagrams" / f"{stem}.pdf"
        if not pdf.is_file():
            print(f"! no pipeline diagram at {pdf}")
            continue
        rel = still_dest(Path("assets/static") / f"pipeline_{stem}.png")
        dst = out / rel
        if is_stale(pdf, dst, force) and not dry_run:
            with tempfile.TemporaryDirectory() as td:
                page = Path(td) / "page.png"
                if not rasterize_pdf(pdf, page, dpi):
                    print("! no ghostscript — cannot rasterize the pipeline diagrams")
                    break
                dst.parent.mkdir(parents=True, exist_ok=True)
                if STILL_SAVE is None or not encode_still(page, dst):
                    shutil.copy2(page, dst)
            n += 1
        elif is_stale(pdf, dst, force):
            n += 1
        panels.append({"title": title, "src": rel.as_posix(), "width": width})
    if panels:
        static["pipeline"] = panels
    return static, n


def collect_qualitative_figures(out: Path, *, dry_run: bool, force: bool) -> tuple[list, int]:
    """The supplementary's eight qualitative figures, for `qualitative.html`.

    Each `QUALITATIVE_FIGURES` entry becomes one page section holding a couple of
    `buildTeaserGrid` manifests: the input pool, then every method group side by
    side under one row axis.

    Tiles come from two places. GT and pool tiles carry no metric, so those are
    copied straight out of `figures/` (`harvest`). PREDICTION tiles are re-made
    unbadged (`pred_cell`) so the number can be an HTML overlay, as on the main
    page's appearance ladder — see the comment there. A cell that resolves to
    None draws as the dashed placeholder, the same gap `\\qtile@resolve` fills
    with placeholder-square on paper.
    """
    figures, n = [], 0
    # Files as they are ENCODED into assets/, so plate_height() below measures
    # the tile the page actually shows -- a cropped one differs in shape from
    # the render it was cut out of. Each grid takes the slice it appended.
    made: list[Path] = []

    def harvest(base: Path, name: str, max_px: int | None = None) -> str | None:
        """One tile, by extension-less base path (as the .tex references it)."""
        nonlocal n
        src = next((p for ext in (".jpg", ".png", ".jpeg")
                    if (p := base.with_suffix(ext)).is_file()), None)
        if src is None:
            return None
        made.append(src)
        rel = still_dest(Path("assets/qualitative") / name)
        if write_asset(src, out / rel, dry_run=dry_run, force=force, max_px=max_px):
            n += 1
        return rel.as_posix()

    # The paper burns the metric into the tile (update_qualitative's _draw_label).
    # The page draws it as an HTML overlay instead -- the same trade the appearance
    # ladder on the main page makes, for the same two reasons: the number then
    # tracks the run's eval JSON rather than a pixel nobody re-renders, and the
    # badge scales with the PAGE. The burned one is a fixed POINT size against the
    # figure's own tile width, so one badge came out at 19% of a GSO tile and 10%
    # of a DAVIS one, on the same screen.
    #
    # Only the PREDICTION tiles are re-made; GT and pool tiles carry no badge, so
    # those are still copied straight out of figures/. And "re-made" reuses
    # update_qualitative's own machinery -- its render resolver, its per-row crop
    # solver, its track-overlay preference and its white composite are imported
    # and called with `label=None`, never reimplemented, so a page tile differs
    # from the paper's by exactly the badge.
    import update_qualitative as uq
    specs = _paths.qualitative()
    raw_dir = out / "assets" / ".qual"        # unbadged intermediates, cached by mtime

    def pred_cell(fig: dict, scene: str, stem: str, row: str) -> tuple[str | None, str | None]:
        """(manifest path, badge text) for one prediction tile."""
        nonlocal n
        spec = specs[fig["spec"]]
        cols, frame = spec.get("columns"), None
        track_on = spec.get("tracks_sub")
        if cols:
            run = cols.get(stem)
            view = (spec["train_views"] if row == "Train" else spec["scenes"])[scene]
            sub = "renders_train" if row == "Train" else spec["render_sub"]
            names = (view,)
        else:
            # `<method>_<frame>`; split from the RIGHT, method names carry underscores
            method, _, frame = stem.rpartition("_")
            run = spec.get("methods", {}).get(method)
            if row == "Train":
                # `also_train` means the figure emits a separate train subrow;
                # without it (DAVIS) the primary render IS the train view.
                sub = "renders_train" if spec.get("also_train") else spec["render_sub"]
            else:
                sub = spec.get("nvs_sub") or spec["render_sub"]
            names = uq._cell_stems(sub, frame)
        # A {"run": ...} entry is a method the paper shows but does not score
        # (TRELLISv2, Pixal3D: lighting-disentangled materials, no comparable
        # photometric number), so it gets a tile and no badge.
        badge_ok = not isinstance(run, dict)
        run = run.get("run") if isinstance(run, dict) else run
        if run is None:
            return None, None

        src, _ = uq._latest_render(run, scene, uq._track_sub(sub, track_on),
                                   tuple(f"{v}.png" for v in names))
        if src is None:
            return None, None
        crop = (uq._track_boxes(fig["spec"], spec, scene).get(
                    "train" if sub == "renders_train" else
                    "nvs" if sub == spec.get("nvs_sub") else "primary")
                if track_on else None)

        raw = raw_dir / f"{fig['id']}_{scene}_{stem}_{row.lower()}.png"
        rel = still_dest(Path("assets/qualitative") / raw.name)
        # Under --dry-run the intermediate is never written, so there is nothing
        # for write_asset to stat; the manifest it would print is unaffected.
        if not dry_run:
            raw.parent.mkdir(parents=True, exist_ok=True)
            uq._write_white(src, raw, label=None, crop=crop)   # the badge is the point
            made.append(raw)
            if write_asset(raw, out / rel, dry_run=dry_run, force=force):
                n += 1

        text = None
        if badge_ok:
            tile_w = spec.get("tile_width") or 0.105
            if sub == spec.get("nvs_sub"):
                b = uq._nvs_cell_label(run, scene, tile_w, frame)
            else:
                is_train_row = sub == "renders_train"
                # Which eval the number comes from: the train row always reads
                # metrics2D_train, and so does the primary row of a figure whose
                # primary render IS a train view (DAVIS). update_qualitative
                # calls this `prim_train`.
                from_train = is_train_row or spec["render_sub"] == "renders_train"
                metric = "psnr" if is_train_row else spec["label_metric"]
                b = uq._cell_label(run, scene, metric, from_train, tile_w, frame)
            text = b.text if b else None
        return rel.as_posix(), text

    def plate_height(paths: list[Path]) -> float:
        """The plate box's aspect, taken from the tiles that go IN it, so the
        box does not crop them: `buildTeaserGrid` reads this as height over
        width, and the cover-fit inside it then has nothing to trim. One value
        per grid — a figure renders at one size (DAVIS 512x291, every other
        512 square), and the ~1% between a scene's crop and its pool frame is
        not worth a per-row box.
        """
        from statistics import median
        try:
            from PIL import Image
        except ImportError:
            return 1.0
        vals = []
        for p in paths:
            try:
                with Image.open(p) as im:
                    vals.append(im.size[1] / im.size[0])
            except OSError:
                pass
        return round(median(vals), 3) if vals else 1.0

    for fig in QUALITATIVE_FIGURES:
        root = _paths.FIGURES / fig["dataset"]
        gt_train, gt_test = fig["gt"]
        gt_frames = fig.get("gt_frames")
        grids, missing = [], 0

        # The input pool: the paper's P column, which has no counterpart in a
        # builder whose rows are flat — so it becomes its own small grid, one
        # row per scene, one column per input view, above the comparison.
        pool_sub, pool_stems = fig["pool"]
        pool_rows = []
        mark = len(made)
        for label, scene in fig["scenes"]:
            cells = [harvest(root / scene / pool_sub / stem,
                             f"{fig['id']}_{scene}_pool_{stem}.png", max_px=320)
                     for stem in pool_stems]
            if any(cells):
                pool_rows.append({"label": label, "cells": cells})
        if pool_rows:
            grids.append({
                "caption": "Input views",
                # Marks this as the pool grid: it is a handful of thumbnails,
                # not a method comparison, so CSS caps its width rather than
                # letting it stretch to the figure's.
                "kind": "inputs",
                "columns": [{"header": s.lstrip("0") or "0", "width": 1} for s in pool_stems],
                "rows": pool_rows, "plateHeight": plate_height(made[mark:]),
                "zoom": "row",
            })

        # ONE grid holding every group side by side, as the paper's own single
        # tabular does: the groups share one row axis, so stacking them repeated
        # the scene labels three times and put "1 view" and "5 views" of the
        # same scene a screen apart. The band becomes the column's `superheader`
        # (drawn as a spanning row above the method names), carried on the first
        # column of each group with `None` after it to continue the span.
        columns, stems = [], []
        for caption, group in fig["groups"]:
            for i, (header, stem) in enumerate(group):
                columns.append({"header": header, "width": 1,
                                "superheader": caption if i == 0 else None})
                stems.append(stem)
        # GT closes the row, outside every band: "" breaks the span run without
        # drawing a second label over it.
        gt_frame_list = gt_frames or [None]
        for i in range(len(gt_frame_list)):
            columns.append({"header": "GT" if i == 0 else None, "width": 1,
                            "superheader": "" if i == 0 else None})

        rows = []
        mark = len(made)
        for label, scene in fig["scenes"]:
            for row in ("Train", "Test"):
                cells = []
                for stem in stems:
                    src, badge = pred_cell(fig, scene, stem, row)
                    cells.append({"src": src, "badge": badge} if src and badge else src)
                pat = gt_train if row == "Train" else gt_test
                for f in gt_frame_list:
                    cells.append(None if pat is None else
                                 harvest(root / scene / "gt" / pat.format(f=f),
                                         f"{fig['id']}_{scene}_gt_{row.lower()}"
                                         f"{'_' + f if f else ''}.png"))
                missing += sum(c is None for c in cells)
                rows.append({"label": f"{label} · {row}", "cells": cells})
        grids.append({
            "columns": columns, "rows": rows,
            "plateHeight": plate_height(made[mark:]),
            # Square renders; the magnifier syncs across the ROW -- rows are
            # scenes here, so "all" would magnify the same point on a
            # different object, and the comparable set is one scene on one
            # view across the methods.
            "zoom": "row",
        })

        cells_total = sum(len(r["cells"]) for g in grids for r in g["rows"])
        print(f"  {fig['id']:16} {len(grids)} grids, {cells_total - missing}/{cells_total} tiles"
              + (f"  ({missing} not rendered yet)" if missing else ""))
        figures.append({"id": fig["id"], "page": fig["page"], "title": fig["title"],
                        "blurb": fig["blurb"], "grids": grids})
    return figures, n


def scene_labels(dataset: str, scenes: list[str]) -> dict[str, str]:
    """Chip/caption labels for one dataset's scenes.

    ActionBench scene dirs are `000-003_<uuid>`; the numeric prefix reads far
    better on a button, but it is NOT unique (`000-004` names two distinct
    scenes on disk), so the uuid head is kept wherever the prefix collides.
    Every other dataset keeps its scene name verbatim — CO3D ids like
    `123_14363_28981` are underscore-delimited but every segment is meaningful.
    """
    if dataset != "oursactionbench":
        return {s: s for s in scenes}
    prefix = {s: s.split("_", 1)[0] for s in scenes}
    clashes = {p for p, n in Counter(prefix.values()).items() if n > 1}
    return {
        s: f"{p} ({s.split('_', 1)[1][:4]})" if p in clashes and "_" in s else p
        for s, p in prefix.items()
    }


def find_ffmpeg() -> str | None:
    """A system ffmpeg, else the one imageio ships (which wrote these mp4s)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
    except ImportError:
        return None
    return imageio_ffmpeg.get_ffmpeg_exe()


# --------------------------------------------------------------------------- #
# Still-image encoding. Measured on this page's render tiles (vs the PNG the
# renderer wrote, and the mean/max channel error at hard foreground/white edges):
#
#   png              1.0x   0.00 / 0     what the renderer emits
#   webp-lossless    1.6x   0.00 / 0     bit-exact
#   jpeg q90 4:4:4   2.0x   5.11 / 41    the convention update_qualitative.py uses
#   webp q90         5.9x   6.76 / 103   default here
#
# webp q90 is the default by choice, not by accident: libwebp has no 4:4:4 lossy
# mode, so it is always chroma-subsampled and bleeds colour across the hard
# foreground/background edge more than JPEG-4:4:4 does. Switch with
# --still-format if that shows up in a figure someone zooms into.
# --------------------------------------------------------------------------- #

STILL_FORMATS = {
    "webp":          (".webp", {"format": "WEBP", "method": 6}),
    "webp-lossless": (".webp", {"format": "WEBP", "method": 6, "lossless": True}),
    "jpeg":          (".jpg",  {"format": "JPEG", "subsampling": 0}),  # 4:4:4, see above
    "png":           (".png",  None),                                  # copy through
}

# Set once in main(); mirrors update_qualitative.py's module-level SYNC_EXT.
STILL_EXT = ".webp"
STILL_SAVE: dict | None = dict(STILL_FORMATS["webp"][1])


def still_dest(path: Path) -> Path:
    """Destination name for a still, under the configured format."""
    return path.with_suffix(STILL_EXT)


def encode_still(src: Path, dst: Path, max_px: int | None = None) -> bool:
    """Re-encode a still into `dst`'s format. False if PIL is unavailable.

    `max_px` caps the longest side. The figure tiles are already 512px, but the
    input-view pools are the raw frames (up to 1138px, 294 KB each) drawn at
    thumbnail size — shipping those at source resolution costs more than every
    tile beside them.
    """
    try:
        from PIL import Image
    except ImportError:
        return False
    img = Image.open(src)
    if max_px and max(img.size) > max_px:
        img.thumbnail((max_px, max_px), Image.LANCZOS)
    params = dict(STILL_SAVE)
    fmt = params.pop("format")
    if fmt == "JPEG" and img.mode in ("RGBA", "LA", "P"):
        # JPEG has no alpha; composite on white, as the paper's figure sync does
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    elif img.mode == "P":
        img = img.convert("RGBA")
    img.save(dst, format=fmt, **params)
    return True


def is_stale(src: Path, dst: Path, force: bool) -> bool:
    """Whether `dst` needs (re)writing: forced, absent, or older than `src`.
    Existence alone is not enough — a re-exported figure or re-rendered orbit
    keeps its filename, so it would shadow its successor forever."""
    return force or not dst.exists() or dst.stat().st_mtime < src.stat().st_mtime


def write_asset(src: Path, dst: Path, *, dry_run: bool, force: bool,
                ffmpeg: str | None = None, crf: int = 28,
                keyframe_every: int | None = None, max_px: int | None = None) -> bool:
    """Copy `src` to `dst`, re-encoding for the web on the way.

    Video: yuv420p + Main profile is what every browser decodes, and `+faststart`
    moves the moov atom to the front so playback starts before the file is fully
    read. `keyframe_every` caps the GOP length — the drag-to-rotate viewer
    scrubs by setting `currentTime`, and a browser seeks by decoding from the
    previous keyframe, so a scrubbed clip needs them dense (default x264 GOP is
    longer than these 60-frame orbits, i.e. one keyframe per file). Stills: per
    `--still-format` (see STILL_FORMATS).
    """
    if not is_stale(src, dst, force):
        return False
    if dry_run:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.suffix != ".mp4":
        # one file per asset: drop a sibling left behind by a different --still-format
        for old in dst.parent.glob(dst.stem + ".*"):
            if old != dst and old.suffix.lower() in {e for e, _ in STILL_FORMATS.values()}:
                old.unlink()
        if STILL_SAVE is None or not encode_still(src, dst, max_px):
            shutil.copy2(src, dst)
        return True
    if ffmpeg is None:                 # --no-transcode, or no ffmpeg found
        shutil.copy2(src, dst)
        return True
    gop = [] if keyframe_every is None else ["-g", str(keyframe_every)]
    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
         "-c:v", "libx264", "-profile:v", "main", "-pix_fmt", "yuv420p",
         "-crf", str(crf), "-preset", "slow", *gop,
         "-movflags", "+faststart", "-an",
         # yuv420p needs even dimensions; renders are square but not guaranteed
         "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2", str(dst)],
        check=True,
    )
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE, help="webpage directory")
    ap.add_argument("--datasets", nargs="*", default=None, help="restrict to these datasets")
    ap.add_argument("--max-scenes", type=int, default=0, help="cap scenes per dataset (0 = all)")
    ap.add_argument("--figure-dpi", type=int, default=200,
                    help="rasterization dpi for the vector pipeline diagrams (default 200)")
    ap.add_argument("--crf", type=int, default=28,
                    help="H.264 quality, lower is bigger (default 28; see README.md for measurements)")
    ap.add_argument("--still-format", choices=sorted(STILL_FORMATS), default="webp",
                    help="still-image codec (default webp: ~5.9x smaller than png; "
                         "webp-lossless is bit-exact at ~1.6x, jpeg is 4:4:4 at ~2x)")
    ap.add_argument("--still-quality", type=int, default=90,
                    help="quality for webp/jpeg stills (default 90; ignored by png/webp-lossless)")
    ap.add_argument("--no-transcode", action="store_true",
                    help="copy the renderer's mp4s verbatim instead of re-encoding for the web")
    ap.add_argument("--force", action="store_true", help="rewrite assets that already exist")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    ffmpeg = None if args.no_transcode else find_ffmpeg()
    if ffmpeg is None and not args.no_transcode:
        print("! no ffmpeg found (pip install imageio-ffmpeg) — copying videos verbatim")

    global STILL_EXT, STILL_SAVE
    STILL_EXT, save = STILL_FORMATS[args.still_format]
    STILL_SAVE = None if save is None else {**save, "quality": args.still_quality}
    if args.no_transcode:
        STILL_EXT, STILL_SAVE = ".png", None

    # ----- resolve every declared chip, build the manifest -------------------
    comparisons: list[dict] = []
    limitations: list[dict] = []
    galleries: list[dict] = []
    to_encode: dict[tuple[str, str, str], Path] = {}   # (ds, scene, key) -> src orbit
    to_encode_pool: dict[tuple[str, str, str], Path] = {}  # (ds, scene, stem) -> src still
    report: list[str] = []
    # `orbits` and `reason` are the chips' internal fields; everything else
    # goes into data.js.
    internal_fields = ("orbits", "reason")

    def coverage(rid: str, pool: list[dict]) -> int:
        return sum(rid in s["media"] for s in pool)

    datasets = [d for d in sorted(DATASETS, key=lambda d: DATASETS[d]["order"])
                if not args.datasets or d in args.datasets]
    for dataset in datasets:
        meta = DATASETS[dataset]
        refs, bases = build_chips(meta, REG_KEY[dataset], dataset)
        ref_ids = {c["id"] for c in refs}

        # scenes = union of orbit coverage over the declared chips (already
        # restricted to the paper's allowlist by build_chips, so the page shows
        # exactly the scenes the tables score)
        excluded = EXCLUDED_SCENES.get(dataset, set())
        limitation_ids = LIMITATION_SCENES.get(dataset, set())
        media: dict[str, dict[str, str]] = defaultdict(dict)
        srcs: dict[tuple[str, str], Path] = {}
        for chip in refs + bases:
            for scene, src in chip["orbits"].items():
                rel = Path("assets") / dataset / scene / f"{chip['id']}.mp4"
                media[scene][chip["id"]] = rel.as_posix()
                srcs[(scene, chip["id"])] = src

        def scene_pool(scene: str) -> list[str]:
            """Input-view thumbnail paths for one scene, deferred to `to_encode_pool`
            (mirrors how orbit videos are resolved above then encoded in one batch
            below) -- a missing source file just shortens the list, same as a
            missing orbit drops a chip from `media`."""
            rels = []
            for stem, src in comparison_pool(dataset, scene):
                if not src.is_file():
                    continue
                rel = still_dest(Path("assets") / dataset / scene / f"pool_{stem}.png")
                to_encode_pool[(dataset, scene, stem)] = src
                rels.append(rel.as_posix())
            return rels

        ordered_scenes = sorted(media, key=lambda s: scene_sort_key(dataset, s))
        labels = scene_labels(dataset, ordered_scenes)
        scenes_paired, scenes_solo, scenes_limitation = [], [], []
        for scene in ordered_scenes:
            if not any(k in ref_ids for k in media[scene]):
                continue  # nothing to put on the left
            entry = {"id": scene, "label": labels[scene], "media": media[scene],
                     "pool": scene_pool(scene)}
            if scene in limitation_ids:
                scenes_limitation.append(entry)
            elif scene in excluded:
                continue
            elif any(k not in ref_ids for k in media[scene]):
                scenes_paired.append(entry)
            else:
                scenes_solo.append(entry)
        if args.max_scenes:
            scenes_paired = scenes_paired[: args.max_scenes]
            scenes_solo = scenes_solo[: args.max_scenes]

        def make_block(entries: list[dict]) -> dict | None:
            """One `comparisons`-shaped block over `entries` -- shared by the
            interactive comparisons and the limitations section, which differ
            only in WHICH scenes populate `scenes` (LIMITATION_SCENES is
            carved out of `scenes_paired` above, not a separate query)."""
            if not entries:
                return None
            # open on the rendered configuration covering the most scenes, at the
            # highest view count that ties; nothing rendered -> the first declared
            # one, greyed, and the block says so. `default_views`, when the
            # dataset declares one, restricts the pool to that count first
            # (falling back to every count if nothing is available there).
            live = [r for r in refs if r["available"]]
            pref_views = meta.get("default_views")
            pool = [r for r in live if r["views"] == pref_views] if pref_views else live
            pool = pool or live
            best = max(pool, key=lambda r: (coverage(r["id"], entries), r["views"])) \
                if pool else refs[0]
            default_ref = best["id"]
            at_view = [b for b in bases if b["views"] == best["views"]]
            for scene in entries:
                for key in scene["media"]:
                    to_encode[(dataset, scene["id"], key)] = srcs[(scene["id"], key)]
            return {
                "id": dataset,
                "title": meta["title"],
                "blurb": meta.get("blurb", ""),
                "references": [{k: v for k, v in c.items() if k not in internal_fields}
                               for c in refs],
                "defaultReference": default_ref,
                "baselines": [{k: v for k, v in c.items() if k not in internal_fields}
                              for c in bases],
                "defaultBaseline": next((b["id"] for b in at_view if b["available"]), None),
                "scenes": entries,
            }

        block = make_block(scenes_paired)
        default_ref = block["defaultReference"] if block else None
        if block:
            comparisons.append(block)
        lim_block = make_block(scenes_limitation)
        if lim_block:
            limitations.append(lim_block)

        if scenes_solo:
            picked = {
                s["id"]: max((k for k in s["media"] if k in ref_ids),
                             key=lambda r: coverage(r, scenes_solo))
                for s in scenes_solo
            }
            for s in scenes_solo:
                to_encode[(dataset, s["id"], picked[s["id"]])] = srcs[(s["id"], picked[s["id"]])]
            galleries.append({
                "id": f"{dataset}_orbits",
                "title": f"{meta['title']} — more orbit renders",
                "blurb": "Our reconstruction, 360&deg; orbit. No baseline orbit rendered for these scenes yet.",
                "columns": 4,
                "items": [
                    {"src": s["media"][picked[s["id"]]], "caption": s["label"]}
                    for s in scenes_solo
                ],
            })

        # ----- per-dataset report lines --------------------------------------
        report.append(f"  comparison {dataset:<16} {len(scenes_paired)} scenes")
        if scenes_limitation:
            names = ", ".join(s["id"] for s in scenes_limitation)
            report.append(f"  limitations {dataset:<15} {names}")
        for chip in refs:
            mark = " *" if chip["id"] == default_ref else "  "
            state = f"{coverage(chip['id'], scenes_paired)} scenes" if chip["available"] \
                else chip["reason"]
            vs = ", ".join(
                b["label"] + ("" if b["available"] else " (no orbit)")
                for b in bases if b["views"] == chip["views"]
            )
            # chip['label'] no longer names the view count (the page's own
            # pill doesn't either — the Views chip already says which is
            # selected), so print it here explicitly or the rows of a
            # multi-view dataset become indistinguishable on the console.
            ref_label = f"{chip['label']} ({chip['views']}v)"
            report.append(f"      {mark} {ref_label:<42} {state:<34} vs {vs or '-'}")
        for s in scenes_paired:
            gaps = [b["label"] for b in bases
                    if b["available"] and b["id"] not in s["media"]]
            if gaps:
                report.append(f"      no orbit for scene {s['label']}: {', '.join(gaps)}")

    # ----- encode ------------------------------------------------------------
    # `to_encode` was filled from the finished blocks, so only media a chip can
    # actually reach is encoded (a scene with no ours orbit, or past
    # --max-scenes, costs nothing and ships nothing).
    n_written = 0
    src_bytes = dst_bytes = 0
    for (dataset, scene, key), src in sorted(to_encode.items()):
        dst = args.out / "assets" / dataset / scene / f"{key}.mp4"
        if write_asset(src, dst, dry_run=args.dry_run, force=args.force,
                       ffmpeg=ffmpeg, crf=args.crf):
            n_written += 1
            if not args.dry_run:
                src_bytes += src.stat().st_size
                dst_bytes += dst.stat().st_size

    # Sweep videos the manifest no longer references (renamed labels, dropped
    # scenes) — orphans would otherwise ship in the supplementary ZIP forever.
    if not args.dry_run and not args.datasets and not args.max_scenes:
        expected = {args.out / "assets" / d / s / f"{k}.mp4" for (d, s, k) in to_encode}
        for old in (args.out / "assets").glob("*/*/*.mp4"):
            if old not in expected:
                old.unlink()
                print(f"  pruned stale asset {old.relative_to(args.out)}")

    # Input-view thumbnails for the interactive comparisons, same batching as
    # the orbit videos above.
    for (dataset, scene, stem), src in sorted(to_encode_pool.items()):
        rel = still_dest(Path("assets") / dataset / scene / f"pool_{stem}.png")
        if write_asset(src, args.out / rel, dry_run=args.dry_run, force=args.force,
                       max_px=320):
            n_written += 1

    if not args.dry_run and not args.datasets and not args.max_scenes:
        expected_pool = {args.out / still_dest(Path("assets") / d / s / f"pool_{stem}.png")
                         for (d, s, stem) in to_encode_pool}
        for old in (args.out / "assets").glob(f"*/*/pool_*{STILL_EXT}"):
            if old not in expected_pool:
                old.unlink()
                print(f"  pruned stale asset {old.relative_to(args.out)}")

    static, n = collect_static_figures(args.out, dry_run=args.dry_run,
                                       force=args.force, dpi=args.figure_dpi)
    n_written += n

    print("qualitative figures (qualitative.html)")
    qualitative, n = collect_qualitative_figures(args.out, dry_run=args.dry_run,
                                                 force=args.force)
    n_written += n

    # ----- completion viewer: one orbit per method on the completion scene ----
    viewer_methods = []
    for key, label, run_dir in COMPLETION_VIEWER:
        src = scene_orbits(run_dir, "image").get(COMPLETION_SCENE)
        if src is None:
            print(f"! no completion orbit for {label} under {run_dir}/image/{COMPLETION_SCENE}")
            continue
        rel = Path("assets") / "completion" / f"{key}.mp4"
        if write_asset(src, args.out / rel, dry_run=args.dry_run,
                       force=args.force, ffmpeg=ffmpeg, crf=args.crf,
                       keyframe_every=8):
            n_written += 1
        viewer_methods.append({"id": key, "label": label, "src": rel.as_posix()})
    # The input tile is the still row's own "Input" panel, not a second
    # derivation of its name — so a rename cannot leave the viewer 404ing, and a
    # missing panel drops the viewer rather than pointing it at nothing.
    viewer_input = next((p["src"] for p in static.get("completion", [])
                         if p["title"] == "Input"), None)
    if viewer_methods and viewer_input:
        static["completion_viewer"] = {"input": viewer_input, "methods": viewer_methods}

    # Sweep statics the manifest no longer references (renamed panels,
    # restructured figures). The encode step skips existing files, so without
    # this a stale render keeps shadowing its successor forever — the pipeline
    # panels did exactly that when the diagrams were re-exported under
    # unchanged names (delete the file to force its re-render).
    if not args.dry_run:
        expected_static = set(re.findall(
            r'"(assets/(?:static|completion|qualitative)/[^"]+)"',
            json.dumps([static, qualitative])))
        for sub in ("static", "completion", "qualitative"):
            for old in (args.out / "assets" / sub).glob("*"):
                if old.relative_to(args.out).as_posix() not in expected_static:
                    old.unlink()
                    print(f"  pruned stale asset {old.relative_to(args.out)}")

    # Every name on the page lives on the block that shows it (`references[]` /
    # `baselines[]` carry their labels).
    manifest = {
        "links": PAGE_LINKS,
        "assetBase": ASSET_BASE_URL,
        "gallery3d": GALLERY3D,
        "static": static,
        "comparisons": comparisons,
        "limitations": limitations,
        "galleries": galleries,
        # Read only by qualitative.html; index.html ignores it (both pages load
        # this one manifest, and each mounts the keys its own DOM has hosts for).
        "qualitative": qualitative,
    }

    banner = (
        "/* Copyright (c) Meta Platforms, Inc. and affiliates. */\n\n"
        "/* GENERATED by paper/webpage/collect_assets.py — do not hand-edit;\n"
        " * change DATASETS in that script (methods are mapping.json labels)\n"
        " * and re-run instead. */\n"
    )
    data_js = banner + "window.PAGE_DATA = " + json.dumps(manifest, indent=2, ensure_ascii=False) + ";\n"

    if args.dry_run:
        print(data_js)
    else:
        (args.out / "data.js").write_text(data_js)

    # ----- report ------------------------------------------------------------
    tag = "[dry-run] " if args.dry_run else ""
    print(f"{tag}assets written: {n_written}")
    if dst_bytes:
        print(f"  video {src_bytes / 1e6:.1f} MB -> {dst_bytes / 1e6:.1f} MB "
              f"({src_bytes / dst_bytes:.1f}x smaller, crf {args.crf})")
    print("\n".join(report))
    for block in galleries:
        print(f"  gallery    {block['id']:<16} {len(block['items'])} items")
    if not args.dry_run:
        print(f"wrote {args.out / 'data.js'}")


if __name__ == "__main__":
    main()
