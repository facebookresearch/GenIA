#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.

"""Re-subset Latin Modern Roman and splice it into css/app.css's @font-face blocks.

The page renders in Latin Modern Roman — the OTF re-release of Computer Modern,
and what the paper itself typesets in (`main.tex` loads no font package; LaTeX's
`article` class default stands). Matching it without a network request means
embedding it: this subsets the four weights to the page's own character set and
inlines them as `data:font/woff2;base64,...` URIs, so re-running it after adding
a new symbol somewhere on the page is one command rather than a manual
copy-paste of base64 text.

    python webpage/subset_font.py

Needs `fontTools[woff]` (for WOFF2 + brotli) and a Latin Modern install — this
script reads the OTFs straight out of the running machine's texlive, at
`/usr/share/texlive/texmf-dist/fonts/opentype/public/lm/`; point `LM_DIR` at
another `texmf-dist/fonts/opentype/public/lm/` if the fonts live elsewhere.
"""
import base64
import re
import sys
from pathlib import Path

try:
    from fontTools import subset
    from fontTools.ttLib import TTFont
except ImportError:
    sys.exit("needs fontTools: pip install fontTools[woff]")

HERE = Path(__file__).resolve().parent
CSS_PATH = HERE / "css" / "app.css"
LM_DIR = Path("/usr/share/texlive/texmf-dist/fonts/opentype/public/lm")

# (css weight/style, source file, css-block-order — bolditalic before bold,
# since "bold" is a substring of "bolditalic" and a naive splice would corrupt
# the bolditalic block if it ran second).
WEIGHTS = [
    ("bolditalic", "lmroman10-bolditalic.otf"),
    ("regular", "lmroman10-regular.otf"),
    ("italic", "lmroman10-italic.otf"),
    ("bold", "lmroman10-bold.otf"),
]

# Basic Latin + Latin-1 (prose, digits, ×, nbsp) + the specific General
# Punctuation / Arrows / Greek codepoints the page's HTML entities and literal
# strings resolve to (checked via grep over index.html, tables.js, data.js,
# js/app.js): – — ‘ ’ “ ” … ‹ › (guillemets, pager arrows) ← ↑ → ↓ (rank/metric
# arrows), plus uppercase Greek (present in the source face; see NOTE below).
#
# NOTE: Latin Modern Roman's TEXT face carries no lowercase Greek and no
# checkmark — δ (the tracking header) and ✓ (the ablation flag) come from a
# separate math/symbol font in the paper too, so the browser falls back to the
# next font in the CSS stack for those two glyphs alone. Not a bug here;
# widening UNICODES cannot add glyphs the source font does not have.
UNICODES = (
    list(range(0x0020, 0x007F)) +
    list(range(0x00A0, 0x0100)) +
    list(range(0x2010, 0x2028)) +
    list(range(0x2030, 0x203B)) +
    list(range(0x2190, 0x219A)) +
    list(range(0x0391, 0x03CA)) +
    [0x2713]
)


def subset_weight(src: Path) -> bytes:
    font = TTFont(src)
    options = subset.Options()
    options.flavor = "woff2"
    options.desubroutinize = True
    options.name_IDs = ["*"]
    options.notdef_outline = True
    options.recalc_bounds = True
    subsetter = subset.Subsetter(options=options)
    subsetter.populate(unicodes=UNICODES)
    subsetter.subset(font)
    import io
    buf = io.BytesIO()
    font.save(buf)
    return buf.getvalue()


def main() -> None:
    if not LM_DIR.is_dir():
        sys.exit(f"no Latin Modern OTFs at {LM_DIR} — install texlive-fonts-extra "
                 "or point LM_DIR at another texmf-dist/fonts/opentype/public/lm/")
    css = CSS_PATH.read_text()
    for weight, fname in WEIGHTS:
        woff2 = subset_weight(LM_DIR / fname)
        b64 = base64.b64encode(woff2).decode("ascii")
        # Locate the @font-face block by its (weight, style) pair rather than
        # by the placeholder text it once held, so a re-run is idempotent.
        is_bold = "bold" in weight
        is_italic = "italic" in weight
        block_re = re.compile(
            r"@font-face\s*\{[^}]*?font-weight:\s*" + ("700" if is_bold else "400") +
            r";[^}]*?font-style:\s*" + ("italic" if is_italic else "normal") +
            r";[^}]*?src:\s*url\(data:font/woff2;base64,([A-Za-z0-9+/=]+)\)[^}]*?\}",
            re.S,
        )
        m = block_re.search(css)
        if not m:
            sys.exit(f"no @font-face block found for weight={weight} in {CSS_PATH}")
        css = css[:m.start(1)] + b64 + css[m.end(1):]
        print(f"{weight}: {len(woff2)} bytes -> spliced")
    CSS_PATH.write_text(css)
    print(f"wrote {CSS_PATH}")


if __name__ == "__main__":
    main()
