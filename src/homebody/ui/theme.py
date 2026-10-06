"""Shared console palette and presentation. 2D sizes are pixels of a 640-pixel-wide pane, scaled to each image, and 3D sizes are metres."""
from html import escape

MASK = (177, 4, 14)
PLAN = (140, 21, 21)
ROUTE = (177, 4, 14)
GOAL = (130, 0, 0)
FACING = (83, 86, 90)
STANCE = (184, 58, 75)
ACCENT = (177, 4, 14)
RELEASE = (130, 0, 0)
ROBOT = (46, 45, 41)
BRAND = (140, 21, 21)
SCENE_BACKGROUND = (248, 247, 243)
LINE, RING, DOT = 4, 11, 6
LINE_3D, DOT_3D, HEADING = .025, .022, .4
LIGHTING = {
    "ambient": {"color": (255, 255, 255), "intensity": .45},
    "key": {"color": (255, 248, 237), "intensity": 2., "position": (4., -6., 9.),
            "cast_shadow": True},
    "fill": {"color": (245, 242, 236), "intensity": .8, "position": (-5., -3., 6.),
             "cast_shadow": False},
}
MATTE_ROUGHNESS = .8
PANEL_WIDTH = 460

STYLE = f"<style>:root{{--hb-accent:rgb{BRAND};}}</style>" + """
<style>
/* Viser renders all generated widgets as siblings inside this one Box. */
div:has(> div > .hb-header) {padding-inline:16px;}
/* The header's live dot replaces viser's own connection row. */
[data-testid="floating-panel-handle"]:has(.tabler-icon-cloud-check, .tabler-icon-player-pause)
 {display:none !important;}
.hb-header {display:flex; align-items:center; justify-content:space-between;
 padding:5px 0 12px; border-bottom:1px solid #e7e2dc; color:#2e2d29;}
.hb-wordmark {font-size:19px; font-weight:700; letter-spacing:-.75px;}
.hb-wordmark span {color:var(--hb-accent);}
.hb-meta {display:flex; align-items:center; gap:12px; font-size:11px; color:#6d6c69;}
.hb-live {display:inline-flex; align-items:center; gap:6px; color:#2e2d29; font-weight:600;}
.hb-live::before {content:""; width:9px; height:9px; border-radius:50%; background:#e0201b;
 animation:hb-pulse 1.6s ease-in-out infinite;}
.hb-live .hb-off {display:none;}
body:not(:has(.tabler-icon-cloud-check)) .hb-live::before {background:#979694; animation:none;}
body:not(:has(.tabler-icon-cloud-check)) .hb-live .hb-on {display:none;}
body:not(:has(.tabler-icon-cloud-check)) .hb-live .hb-off {display:inline;}
.hb-room {text-align:right; line-height:1.35;}
.hb-section {margin:14px 0 7px; color:#2e2d29; font-size:11px;
 font-weight:650; letter-spacing:.02em;}
.hb-bubble {margin:8px 0 18px; padding:16px 18px; min-height:150px; box-sizing:border-box;
 border:1px solid #e7e2dc; border-radius:11px;
 background:#fbf9f6; color:#2e2d29; animation:hb-appear .2s ease-out;}
.hb-title {display:flex; align-items:center; gap:6px; font-size:13px;
 color:var(--hb-accent); font-weight:650; letter-spacing:.03em; margin-bottom:5px;}
.hb-message {font-size:17px; line-height:1.55; white-space:pre-wrap; overflow-wrap:anywhere;}
.hb-note.is-done::before {content:"✓ "; color:var(--hb-accent); font-weight:700;}
.hb-stage {margin-bottom:18px; padding:10px 13px; border-left:2px solid var(--hb-accent);
 border-radius:0 7px 7px 0; background:#fbf9f6; color:#2e2d29;
 font-size:12px; line-height:1.45; overflow-wrap:anywhere;}
.hb-result {border-left-color:#979694; background:#fbf9f6;}
.hb-stage-head {display:flex; justify-content:space-between; align-items:baseline; gap:8px;}
.hb-stage-count {color:#6d6c69; font-size:10px;}
.hb-steps {display:flex; flex-wrap:wrap; gap:6px; margin:9px 0 0; padding:0;
 list-style:none;}
.hb-steps li {padding:4px 7px; border:1px solid #e7e2dc; border-radius:6px;
 color:#64625f; background:#fff; font-size:11px;}
.hb-steps li.is-current {border-color:var(--hb-accent); color:var(--hb-accent);
 background:#fff5f3; font-weight:650;}
.hb-steps li.is-failed {border-color:var(--hb-accent); color:var(--hb-accent);
 background:#fff5f3;}
.hb-steps li.is-done::before {content:"✓ "; color:#77736e;}
.hb-step-time {margin-left:7px; color:#6d6c69; font-variant-numeric:tabular-nums;}
.hb-note {margin:2px 0 18px; color:#585754; font-size:12px; line-height:1.45;
 white-space:pre-wrap; overflow-wrap:anywhere;}
/* Viser's default input label takes a left column; the instruction is a full-width card. */
.mantine-Flex-root:has(.mantine-Textarea-root) {flex-direction:column; align-items:stretch;}
.mantine-Flex-root:has(.mantine-Textarea-root) > :first-child
 {width:100% !important; box-sizing:border-box; padding:0 0 6px !important;}
.mantine-Flex-root:has(.mantine-Textarea-root) > :last-child
 {width:100% !important; min-width:0; flex:none;}
.mantine-Textarea-root,.mantine-Textarea-wrapper,.mantine-Textarea-input
 {width:100% !important; box-sizing:border-box;}
.mantine-Input-input {border-radius:8px; font-size:13px; line-height:1.45;}
.mantine-Textarea-input {height:78px !important; min-height:78px !important;
 max-height:78px !important; resize:none !important; overflow-y:auto !important;
 padding:8px 10px !important;}
.mantine-Button-root {border-radius:8px; font-weight:600;}
@keyframes hb-pulse {0%,100% {opacity:1; box-shadow:0 0 0 0 rgba(224,32,27,.5)}
 50% {opacity:.55; box-shadow:0 0 0 4px rgba(224,32,27,0)}}
@keyframes hb-appear {from {opacity:.25; transform:translateY(4px)}
 to {opacity:1; transform:translateY(0)}}
@media (prefers-reduced-motion:reduce) {
 .hb-bubble,.hb-live::before {animation:none}}
</style>
"""


def header(scene_name):
    return ('<div class="hb-header"><div class="hb-wordmark">Home<span>Body</span></div>'
            '<div class="hb-meta"><span class="hb-live" role="status">'
            '<span class="hb-on">Connected</span><span class="hb-off">Disconnected</span></span>'
            f'<span class="hb-room">{escape(scene_name)} (sim)</span></div></div>')


def section(title):
    return f'<div class="hb-section">{escape(title)}</div>'


def note(text, *, done=False):
    """Model or provider text as escaped HTML, never markdown."""
    kind = "hb-note is-done" if done else "hb-note"
    return f'<div class="{kind}">{escape(text)}</div>' if text else ""
