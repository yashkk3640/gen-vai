"""The storyboard as a page to review: one card per shot, and the rhythm at a glance.

A storyboard is data, and data is a poor thing to sign off. This lays it out the way a
storyboard is read on paper - a frame, and beside it what the camera does, what is said,
how it is cut and why the shot is there - plus a bar showing where the time goes.

Pure: a storyboard and the paths of its key frames in, one self-contained HTML string
out. No script, no external assets; it opens from disk and prints.
"""

from html import escape

from genvai.storyboard import Shot, Storyboard

_ROLE_COLOURS = {
    "hook": "#F4B400",
    "build": "#7BAAF7",
    "reveal": "#F06292",
    "offer": "#81C995",
    "proof": "#B39DDB",
    "cta": "#FF8A65",
}

_WORDS = {
    "push_in": "push in",
    "pull_out": "pull out",
    "pan_left": "pan left",
    "pan_right": "pan right",
    "eye_level": "eye level",
    "top_down": "top down",
    "slide_up": "slides up",
    "type": "types on",
    "words": "word by word",
    "pop": "pops",
    "fade": "fades in",
    "light_leak": "light leak",
}


READABLE = 4.5
"""The contrast ratio a caption must reach - WCAG's figure for body text."""


def page(
    board: Storyboard,
    frames: dict[str, str],
    checks: dict[str, tuple[tuple[str, str, float, str], ...]] | None = None,
) -> str:
    """The storybook for `board`. `frames` maps shot id to a key frame's relative path;
    `checks` maps it to each caption's measured legibility."""
    checks = checks or {}
    deep, light, accent = board.palette
    starts = _starts(board)
    cards = "\n".join(
        _card(board, shot, index, starts[index], frames.get(shot.id, ""), checks.get(shot.id, ()))
        for index, shot in enumerate(board.shots)
    )
    return _TEMPLATE.format(
        title=escape(board.title),
        logline=escape(board.logline),
        arc=escape(board.arc),
        bpm=f"{board.bpm:.0f}",
        duration=f"{board.duration:.1f}",
        shots=len(board.shots),
        version=board.version,
        deep=deep,
        light=light,
        accent=accent,
        rhythm=_rhythm(board),
        legend=_legend(),
        cards=cards,
    )


def _starts(board: Storyboard) -> list[float]:
    starts, at = [], 0.0
    for shot in board.shots:
        starts.append(at)
        at += board.seconds(shot)
    return starts


def _rhythm(board: Storyboard) -> str:
    total = sum(s.beats for s in board.shots) or 1.0
    return (
        "".join(
            f'<div class="beat" style="flex:{shot.beats};background:{_ROLE_COLOURS[shot.role]}" '
            f'title="{escape(shot.id)} {shot.role} - {shot.beats:g} beats">'
            f"<span>{escape(shot.id.upper())}</span></div>"
            for shot in board.shots
        )
        + f'<div class="total">{total:g} beats</div>'
    )


def _legend() -> str:
    return "".join(
        f'<span class="key"><i style="background:{colour}"></i>{role}</span>'
        for role, colour in _ROLE_COLOURS.items()
    )


def _card(
    board: Storyboard,
    shot: Shot,
    index: int,
    start: float,
    frame: str,
    checks: tuple[tuple[str, str, float, str], ...] = (),
) -> str:
    seconds = board.seconds(shot)
    image = (
        f'<img src="{escape(frame)}" alt="Key frame of {escape(shot.id)}">'
        if frame
        else '<div class="noframe">no frame</div>'
    )
    measured = {(text, role): (ratio, fix) for text, role, ratio, fix in checks}
    captions = (
        "".join(
            f"<li><b>{escape(c.text)}</b> <em>{c.role} · {_word(c.entrance)}"
            f"{f' at beat {c.at:g}' if c.at else ''}</em>"
            f"{_readability(*measured[(c.text, c.role)]) if (c.text, c.role) in measured else ''}"
            "</li>"
            for c in shot.captions
        )
        or "<li><em>no text</em></li>"
    )
    effects = "".join(f'<span class="chip">{_word(e)}</span>' for e in shot.effects)
    picture = board.picture(shot.picture)
    source = (
        f"{picture.kind}{f' · {escape(picture.name)}' if picture.name else ''}"
        f" · poster {picture.poster + 1}"
        if picture
        else ("whole poster" if shot.layout == "poster" else "none")
    )
    return f"""
<article class="shot">
  <div class="frame">{image}<span class="num">{index + 1}</span></div>
  <div class="body">
    <header>
      <span class="role" style="background:{_ROLE_COLOURS[shot.role]}">{shot.role}</span>
      <span class="time">{start:.1f}s – {start + seconds:.1f}s · {shot.beats:g} beats</span>
    </header>
    <p class="subject">{escape(shot.subject)}</p>
    <dl>
      <dt>Layout</dt><dd>{shot.layout}</dd>
      <dt>Framing</dt><dd>{shot.framing}</dd>
      <dt>Angle</dt><dd>{_word(shot.angle)}</dd>
      <dt>Camera</dt><dd>{_word(shot.move)}</dd>
      <dt>Cut in</dt><dd>{_word(shot.cut)}</dd>
      <dt>Picture</dt><dd>{source}</dd>
    </dl>
    <ul class="captions">{captions}</ul>
    <div class="chips">{effects}</div>
    <p class="note">{escape(shot.note)}</p>
  </div>
</article>"""


def _readability(ratio: float, fix: str) -> str:
    good = ratio >= READABLE
    note = "" if fix == "as designed" else f" · {escape(fix)}"
    return (
        f' <span class="read {"ok" if good else "bad"}" '
        f'title="Contrast against the picture behind it">'
        f"{'✓' if good else '✗'} {ratio:.1f}:1{note}</span>"
    )


def _word(value: str) -> str:
    return _WORDS.get(value, value.replace("_", " "))


_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title} · storyboard</title>
<style>
:root {{ --deep: {deep}; --light: {light}; --accent: {accent};
  --bg: #121015; --panel: #1c1920; --line: #2d2833; --text: #efe9f1; --muted: #a79fb0; }}
* {{ box-sizing: border-box; }}
body {{ margin: 0; background: var(--bg); color: var(--text);
  font: 15px/1.5 "Segoe UI", system-ui, sans-serif; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 32px 16px 64px; }}
.top {{ border-left: 6px solid var(--accent); padding: 4px 0 4px 18px; margin-bottom: 28px; }}
h1 {{ margin: 0 0 6px; font-size: 28px; letter-spacing: .01em; }}
.logline {{ margin: 0 0 12px; font-size: 17px; color: var(--light); max-width: 70ch; }}
.meta {{ display: flex; flex-wrap: wrap; gap: 8px 18px; color: var(--muted); font-size: 13px; }}
.meta b {{ color: var(--text); font-weight: 600; }}
.swatches {{ display: inline-flex; gap: 4px; vertical-align: middle; }}
.swatches i {{ width: 16px; height: 16px; border-radius: 4px; border: 1px solid #0006; }}
h2 {{ font-size: 13px; text-transform: uppercase; letter-spacing: .12em; color: var(--muted);
  margin: 28px 0 10px; }}
.rhythm {{ display: flex; align-items: stretch; height: 42px; gap: 3px; }}
.beat {{ border-radius: 5px; display: flex; align-items: center; justify-content: center;
  min-width: 22px; color: #1a1420; font-weight: 700; font-size: 12px; }}
.total {{ align-self: center; color: var(--muted); font-size: 12px; padding-left: 8px;
  white-space: nowrap; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 10px; color: var(--muted);
  font-size: 12px; }}
.key i {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px;
  margin-right: 5px; vertical-align: -1px; }}
.shots {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(340px, 1fr));
  gap: 16px; }}
.shot {{ display: grid; grid-template-columns: 132px 1fr; gap: 14px; background: var(--panel);
  border: 1px solid var(--line); border-radius: 12px; padding: 12px; }}
.frame {{ position: relative; }}
.frame img, .noframe {{ width: 132px; aspect-ratio: 9 / 16; object-fit: cover;
  border-radius: 8px; display: block; background: #000; }}
.noframe {{ display: grid; place-items: center; color: var(--muted); font-size: 12px; }}
.num {{ position: absolute; top: 6px; left: 6px; background: #000b; color: #fff;
  border-radius: 6px; padding: 0 7px; font-weight: 700; font-size: 13px; }}
.body header {{ display: flex; justify-content: space-between; align-items: center;
  gap: 8px; margin-bottom: 6px; }}
.role {{ color: #1a1420; font-weight: 700; font-size: 12px; text-transform: uppercase;
  letter-spacing: .06em; border-radius: 999px; padding: 2px 10px; }}
.time {{ color: var(--muted); font-size: 12px; white-space: nowrap; }}
.subject {{ margin: 0 0 8px; font-weight: 600; }}
dl {{ display: grid; grid-template-columns: auto 1fr; gap: 1px 10px; margin: 0 0 8px;
  font-size: 13px; }}
dt {{ color: var(--muted); }}
dd {{ margin: 0; }}
.captions {{ list-style: none; margin: 0 0 8px; padding: 0; font-size: 13px; }}
.captions li {{ padding: 3px 0; border-top: 1px dashed var(--line); }}
.captions em {{ color: var(--muted); font-style: normal; font-size: 12px; }}
.chips {{ display: flex; flex-wrap: wrap; gap: 5px; margin-bottom: 8px; }}
.chip {{ font-size: 11px; background: #ffffff12; border: 1px solid var(--line);
  border-radius: 999px; padding: 1px 8px; color: var(--muted); }}
.read {{ font-size: 11px; border-radius: 4px; padding: 0 5px; white-space: nowrap; }}
.read.ok {{ background: #1e3b2a; color: #9be3b4; }}
.read.bad {{ background: #4a1c1c; color: #ffb3b3; }}
.note {{ margin: 0; font-size: 13px; color: var(--light); opacity: .85; font-style: italic; }}
@media (max-width: 520px) {{
  .shots {{ grid-template-columns: 1fr; }}
  .shot {{ grid-template-columns: 104px 1fr; }}
  .frame img, .noframe {{ width: 104px; }}
}}
@media print {{ body {{ background: #fff; color: #000; }} .shot {{ break-inside: avoid; }} }}
</style>
</head>
<body>
<main>
  <section class="top">
    <h1>{title}</h1>
    <p class="logline">{logline}</p>
    <div class="meta">
      <span>Arc <b>{arc}</b></span>
      <span><b>{shots}</b> shots</span>
      <span><b>{duration}s</b> at <b>{bpm}</b> bpm</span>
      <span>Storyboard <b>v{version}</b></span>
      <span class="swatches" title="deep, light, accent">
        <i style="background:{deep}"></i><i style="background:{light}"></i>
        <i style="background:{accent}"></i>
      </span>
    </div>
  </section>
  <h2>Rhythm</h2>
  <div class="rhythm">{rhythm}</div>
  <div class="legend">{legend}</div>
  <h2>Shots</h2>
  <div class="shots">
{cards}
  </div>
</main>
</body>
</html>
"""
