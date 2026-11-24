# Copyright (c) Meta Platforms, Inc. and affiliates.

"""The app's look: 3DView's, so the page and the 3D view embedded in it read as one surface.

Every value here is taken from 3DView's own dark theme (``submodules/3DView/core/src/webview/
ui/styles.ts``, the theme the embedded viewer opens in): its viewport and panel colours, its
cyan accent, the frosted "liquid glass" rim of its panels (the ``--glass-*`` variables, copied
verbatim), its pill toggles, its segmented control and its system font. Change them there
first; this file follows.

Two parts: ``TOKENS``, Gradio theme tokens (where Gradio itself resolves a component's colours,
so they reach components this file never names), and ``CSS``, for what has no token (the
panel rim, the header, the toggles, the nav). ``_dark`` keys because the app is dark-only.
"""

#: 3DView's dark theme (``body[data-viewer-theme="dark"]``).
VIEWPORT = "#1b1b1d"            # --vscode-editor-background: the 3D view's ground
PANEL = "rgba(36, 36, 39, 0.8)"  # --vscode-editorWidget-background #242427 at 80 %
TEXT = "#e1e1e1"                # --vscode-foreground
SUBDUED = "#9aa0aa"             # --vscode-descriptionForeground
ACCENT = "#03d5ff"              # --vscode-focusBorder / --vscode-button-background
ACCENT_HOVER = "#54e2ff"        # --vscode-button-hoverBackground
ACCENT_TEXT = "#06283b"         # --vscode-button-foreground
SECONDARY = "rgba(58, 61, 65, 0.7)"   # --vscode-button-secondaryBackground #3a3d41 at 70 %
SECONDARY_HOVER = "#4a4e54"     # --vscode-button-secondaryHoverBackground
INPUT = "rgba(255, 255, 255, 0.14)"   # --vscode-input-background
HOVER = "rgba(255, 255, 255, 0.13)"   # --vscode-toolbar-hoverBackground
RIM = "rgba(255, 255, 255, 0.55)"     # panel border: --vscode-widget-border at 55 %
MENU = "#242427"                # --vscode-menu-background (opaque: it opens over content)
MENU_BORDER = "rgba(255, 255, 255, 0.16)"
MENU_SELECTION = "rgba(255, 255, 255, 0.09)"
#: An inner surface on a panel (a nested block, a file list): 3DView's transform-field inset,
#: the input colour at 45 %.
SURFACE = "rgba(255, 255, 255, 0.063)"

#: One text scale, after 3DView's compact one: 12 px labels and help text, 13 px body text,
#: controls and buttons, 15 px titles. Every Gradio component sizes from these steps.
TEXT_SIZE = dict(xxs="11px", xs="11px", sm="12px", md="13px", lg="13px", xl="15px", xxl="15px")

FONT = ("-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "Arial", "sans-serif")
FONT_MONO = ("ui-monospace", "SFMono-Regular", "Menlo", "Consolas", "monospace")

TOKENS = {
    "body_background_fill_dark": VIEWPORT,
    "background_fill_primary_dark": VIEWPORT,
    "background_fill_secondary_dark": SURFACE,
    "block_background_fill_dark": PANEL,
    "panel_background_fill_dark": SURFACE,
    "input_background_fill_dark": INPUT,
    "block_border_width_dark": "1px",
    "block_radius": "12px",
    "border_color_primary_dark": MENU_BORDER,
    "border_color_accent_dark": ACCENT,
    "block_border_color_dark": RIM,
    "input_border_color_dark": "transparent",
    "input_border_color_focus_dark": ACCENT,
    "input_radius": "6px",
    "input_placeholder_color_dark": SUBDUED,
    "body_text_color_dark": TEXT,
    "body_text_color_subdued_dark": SUBDUED,
    # A 3DView panel colour, so a label floating over an image stays readable.
    "block_label_background_fill_dark": PANEL,
    "block_label_border_width_dark": "0px",
    "block_label_text_color_dark": SUBDUED,
    "block_title_text_color_dark": TEXT,
    "block_info_text_color_dark": SUBDUED,
    "color_accent": ACCENT,
    "color_accent_soft_dark": "rgba(3, 213, 255, 0.16)",
    "button_large_radius": "6px",
    "button_large_text_weight": "500",
    "button_small_text_weight": "500",
    "button_small_radius": "6px",
    "button_primary_background_fill_dark": ACCENT,
    "button_primary_background_fill_hover_dark": ACCENT_HOVER,
    "button_primary_text_color_dark": ACCENT_TEXT,
    "button_primary_border_color_dark": "transparent",
    "button_primary_border_color_hover_dark": "transparent",
    "button_secondary_background_fill_dark": SECONDARY,
    "button_secondary_background_fill_hover_dark": SECONDARY_HOVER,
    "button_secondary_text_color_dark": TEXT,
    "button_secondary_border_color_dark": "transparent",
    "button_secondary_border_color_hover_dark": "transparent",
    "button_cancel_border_color_hover_dark": "transparent",
    "checkbox_background_color_dark": INPUT,
    "checkbox_background_color_hover_dark": INPUT,
    "checkbox_background_color_focus_dark": INPUT,
    "checkbox_background_color_selected_dark": ACCENT,
    "checkbox_border_color_dark": "transparent",
    "checkbox_border_color_hover_dark": "transparent",
    "checkbox_border_color_focus_dark": ACCENT,
    "checkbox_border_color_selected_dark": ACCENT,
    "checkbox_label_background_fill_dark": "transparent",
    "checkbox_label_background_fill_hover_dark": "transparent",
    "checkbox_label_background_fill_selected_dark": "transparent",
    "checkbox_label_border_color_dark": "transparent",
    "slider_color_dark": ACCENT,
    "stat_background_fill_dark": ACCENT,
    "code_background_fill_dark": SURFACE,
    "table_even_background_fill_dark": "transparent",
    "table_odd_background_fill_dark": SURFACE,
    "table_border_color_dark": MENU_BORDER,
    "link_text_color_dark": ACCENT,
    "link_text_color_hover_dark": ACCENT_HOVER,
    "link_text_color_active_dark": ACCENT_HOVER,
    "link_text_color_visited_dark": ACCENT,
}

CSS = """
.dark {
  /* 3DView's glass rim, verbatim (styles.ts :root, with the dark theme's reflex knobs). */
  --glass-rim-light: #fff; --glass-rim-dark: #000;
  --glass-reflex-light: .5; --glass-reflex-dark: 2;
  --glass-shadow:
    inset 0 0 0 1px color-mix(in srgb,var(--glass-rim-light) calc(var(--glass-reflex-light)*10%),transparent),
    inset 1.8px 3px 0 -2px color-mix(in srgb,var(--glass-rim-light) calc(var(--glass-reflex-light)*90%),transparent),
    inset -2px -2px 0 -2px color-mix(in srgb,var(--glass-rim-light) calc(var(--glass-reflex-light)*80%),transparent),
    inset -3px -8px 1px -6px color-mix(in srgb,var(--glass-rim-light) calc(var(--glass-reflex-light)*60%),transparent),
    inset -0.3px -1px 4px 0 color-mix(in srgb,var(--glass-rim-dark) calc(var(--glass-reflex-dark)*12%),transparent),
    inset -1.5px 2.5px 0 -2px color-mix(in srgb,var(--glass-rim-dark) calc(var(--glass-reflex-dark)*20%),transparent),
    inset 0 3px 4px -2px color-mix(in srgb,var(--glass-rim-dark) calc(var(--glass-reflex-dark)*20%),transparent),
    inset 2px -6.5px 1px -4px color-mix(in srgb,var(--glass-rim-dark) calc(var(--glass-reflex-dark)*10%),transparent),
    0 1px 5px 0 color-mix(in srgb,var(--glass-rim-dark) calc(var(--glass-reflex-dark)*18%),transparent),
    0 8px 22px 0 color-mix(in srgb,var(--glass-rim-dark) calc(var(--glass-reflex-dark)*26%),transparent);
  --glass-shadow-sm:
    inset 0 0 0 1px color-mix(in srgb,var(--glass-rim-light) calc(var(--glass-reflex-light)*14%),transparent),
    inset 1px 1.5px 0 -1px color-mix(in srgb,var(--glass-rim-light) calc(var(--glass-reflex-light)*85%),transparent),
    inset -1px -1.5px 0 -1px color-mix(in srgb,var(--glass-rim-light) calc(var(--glass-reflex-light)*55%),transparent),
    inset 0 2px 3px -2px color-mix(in srgb,var(--glass-rim-dark) calc(var(--glass-reflex-dark)*22%),transparent),
    0 1px 2px 0 color-mix(in srgb,var(--glass-rim-dark) calc(var(--glass-reflex-dark)*14%),transparent);
}

/* Panels: the outer blocks get 3DView's glass; a block nested in one is an inset
   surface on it, not a second rimmed panel. */
.gradio-container .block:not(.block .block):not(.hide-container) { box-shadow: var(--glass-shadow); }
.gradio-container .block .block, .gradio-container .form .block,
.gradio-container .controls-tile .block {
  border: none; box-shadow: none; background: transparent; border-radius: 8px;
}
.gradio-container div.form { background: transparent; border: none; }
.gradio-container .table-wrap, .gradio-container .file-preview {
  background: $SURFACE; border-radius: 8px;
}
.gradio-container .grid-wrap { background: transparent; }

/* The run settings, one panel like the accordion beside them. */
.gradio-container .controls-tile {
  background: var(--block-background-fill); border: 1px solid var(--block-border-color);
  border-radius: var(--block-radius); box-shadow: var(--glass-shadow);
}

/* Buttons as 3DView's: the accent one lifts on hover, the quiet ones are glass. */
.gradio-container button.primary { font-weight: 500;
  transition: background .15s ease, transform .1s ease, box-shadow .15s ease; }
.gradio-container button.primary:hover { transform: translateY(-1px);
  box-shadow: 0 3px 12px rgba(0, 0, 0, .3); }
.gradio-container button.secondary { box-shadow: var(--glass-shadow-sm); }

/* Checkboxes as 3DView's pill toggles. */
.gradio-container input[type=checkbox] {
  appearance: none; -webkit-appearance: none; margin: 0; position: relative; flex: none;
  width: 26px; height: 15px; border-radius: 8px; border: none;
  background: $INPUT; box-shadow: var(--glass-shadow-sm); cursor: pointer;
  transition: background .18s ease;
}
.gradio-container input[type=checkbox]::before {
  content: ""; position: absolute; top: 2px; left: 2px; width: 11px; height: 11px;
  border-radius: 50%; background: $TEXT; transition: translate .2s cubic-bezier(.5,0,0,1);
}
.gradio-container input[type=checkbox]:checked { background: $ACCENT; background-image: none !important; }
.gradio-container input[type=checkbox]:checked::before { translate: 11px 0; background: #fff; }

/* The dropdown's list opens over the text below it, so it is opaque, like 3DView's menus. */
.gradio-container ul.options {
  background: $MENU; border: 1px solid $MENU_BORDER; border-radius: 8px; padding: 4px;
  box-shadow: 0 8px 24px rgba(0, 0, 0, .45);
}
.gradio-container ul.options .item { border-radius: 6px; }
.gradio-container ul.options .item:hover, .gradio-container ul.options .item.active {
  background: $MENU_SELECTION;
}

.run-status, .run-status p { text-align: center; }

/* The running block's loops (ODE steps, iterations), one bar each, after 3DView's sliders. */
.task-list { display: flex; flex-direction: column; gap: 8px; max-width: 560px;
  margin: 2px auto 0; }
.task-row { display: grid; grid-template-columns: 1fr auto; row-gap: 4px; font-size: 12px;
  color: $SUBDUED; }
.task-count { font-variant-numeric: tabular-nums; }
.task-track { grid-column: 1 / -1; height: 4px; border-radius: 99em; background: $INPUT;
  box-shadow: var(--glass-shadow-sm); overflow: hidden; }
.task-fill { height: 100%; border-radius: 99em; background: $ACCENT; transition: width .3s ease; }

/* A section title as 3DView writes one: small, uppercase, in the secondary text colour. */
.section-title { margin: 22px 0 2px; font-size: 11px; font-weight: 600; letter-spacing: .7px;
  text-transform: uppercase; color: $SUBDUED; }
.section-title span { font-weight: 400; letter-spacing: .2px; text-transform: none; }
/* The selected example: the accent rim 3DView gives a focused control. */
.gradio-container .scene-card.selected { border-color: $ACCENT !important;
  box-shadow: 0 0 0 1px $ACCENT, var(--glass-shadow) !important; }

/* The header: one of 3DView's glass panels, sticky over what scrolls under it. */
.app-header {
  position: sticky; top: 8px; z-index: 5;
  display: flex; align-items: center; gap: 12px; margin: 4px 0 18px; padding: 9px 12px 9px 16px;
  border-radius: 12px; border: 1px solid $RIM;
  background: $PANEL; backdrop-filter: blur(12px) saturate(1.3);
  -webkit-backdrop-filter: blur(12px) saturate(1.3); box-shadow: var(--glass-shadow);
}
.app-header-title {
  font-size: 15px; font-weight: 600; letter-spacing: .2px; color: $TEXT;
  white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.app-header-brand { display: flex; align-items: center; gap: 10px; min-width: 0; overflow: hidden; }
/* Header layout: Gradio stretches every block in a row (`.block {width: 100%}` and
   `.row > * {flex-grow: 1 !important}`), so both are undone for the header's halves. */
.gradio-container .app-header { flex-wrap: nowrap !important; }
.gradio-container .app-header > *, .gradio-container .app-header .app-header-nav,
.gradio-container .app-header-nav > * { width: auto !important; min-width: 0 !important; }
.gradio-container .app-header > * { flex: 0 1 auto !important; }
.gradio-container .app-header-nav > * { flex: 0 0 auto !important; }
/* The modes as 3DView's segmented control: a track, the selected one a raised thumb in
   the accent colour. */
.gradio-container .app-header .app-header-nav {
  margin-left: auto; display: flex; gap: 0; flex-wrap: nowrap; padding: 3px;
  border-radius: 99em; background: rgba(255, 255, 255, .077); box-shadow: var(--glass-shadow-sm);
}
.gradio-container .app-header-nav button {
  white-space: nowrap; border-radius: 99em; padding: 5px 16px; box-shadow: none;
  background: transparent; color: $TEXT; font-weight: 500;
}
.gradio-container .app-header-nav button:hover { color: $ACCENT; background: transparent;
  transform: none; box-shadow: none; }
.gradio-container .app-header-nav button.primary {
  background: rgba(225, 225, 225, .16); color: $ACCENT; box-shadow: var(--glass-shadow-sm);
}

/* Scrollbars as 3DView's: a thin pill toned with the text colour, on no track. */
html { scrollbar-width: thin; scrollbar-color: rgba(225, 225, 225, .22) transparent; }
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track, ::-webkit-scrollbar-corner { background: transparent; }
::-webkit-scrollbar-thumb { border: 2px solid transparent; background: rgba(225, 225, 225, .22);
  background-clip: content-box; border-radius: 99em; }
::-webkit-scrollbar-thumb:hover { background: rgba(225, 225, 225, .38); background-clip: content-box; }

/* Gradio's footer: "Use via API" is not for visitors, and its Settings would switch the page
   to a light mode this design does not have. */
footer { display: none !important; }

.app-credits, .app-credits p { font-size: 11px; color: $SUBDUED; text-align: center;
  margin-top: 18px; }
.app-credits a { color: $SUBDUED; text-decoration: underline; text-underline-offset: 2px; }
.app-credits a:hover { color: $TEXT; }
"""
for _name in ("PANEL", "TEXT", "SUBDUED", "ACCENT", "INPUT", "RIM", "MENU_BORDER",
              "MENU_SELECTION", "MENU", "SURFACE"):
    CSS = CSS.replace(f"${_name}", globals()[_name])
