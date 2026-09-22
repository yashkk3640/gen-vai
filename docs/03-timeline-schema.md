# 03 - Timeline schema

The `Timeline` is the single source of truth. The LLM writes it, the user's edits
patch it, the renderer consumes it. Everything else is derived.

All models are **frozen** (immutable). Mutation is impossible; transforms return new values.

## Example

```json
{
  "schema_version": 1,
  "version": 2,
  "intent": "30s vertical explainer on compound interest, calm, soft piano",
  "canvas": {
    "width": 1080, "height": 1920, "fps": 30, "background": "#0B0E14",
    "safe_area": { "top": 0.10, "bottom": 0.20, "left": 0.05, "right": 0.18 }
  },
  "seed": 20260922,
  "style_suffix": "muted film grain, warm key light, shallow depth of field",

  "scenes": [
    {
      "id": "s1",
      "duration": 2.4,
      "role": "hook",
      "visual": {
        "kind": "generated",
        "prompt": "a single coin on a dark table, soft rim light, shallow depth of field",
        "negative_prompt": "text, watermark, people",
        "seed": 11021,
        "asset_id": "img_9f2a...",
        "approved": true
      },
      "motion": {
        "kind": "ken_burns",
        "start_rect": [0.05, 0.05, 0.90, 0.90],
        "end_rect":   [0.20, 0.18, 0.62, 0.62],
        "easing": "ease_in_out"
      },
      "narration": {
        "text": "It starts with a single coin.",
        "voice": "en_US-amy-medium",
        "asset_id": "aud_4c71..."
      },
      "overlays": [
        { "kind": "text", "content": "Day 1", "position": "bottom_center", "style_ref": "caption" }
      ],
      "transition_in": { "kind": "fade", "duration": 0.4 }
    }
  ],

  "music": {
    "state": "resolved",
    "query": { "mood": "calm", "genre": "solo piano", "bpm": [60, 80], "energy": "low" },
    "candidates": [
      { "id": "c1", "title": "Quiet Hours", "source_url": "https://...", "licence": "CC-BY-4.0", "duration": 128.0 }
    ],
    "selected_candidate_id": "c1",
    "asset_id": "mus_7b13...",
    "gain_db": -18.0,
    "duck_under_narration": true
  },

  "captions": { "enabled": true, "mode": "word", "style_ref": "caption", "source": "narration" },

  "export": {
    "audio_variants": ["full", "narration_only"],
    "seamless_loop": false
  },

  "styles": {
    "caption": {
      "font": "Inter-SemiBold", "size_pct": 5.5, "color": "#FFFFFF",
      "stroke_color": "#000000", "stroke_px": 6, "max_chars_per_line": 20,
      "highlight_color": "#FFD400", "highlight_mode": "color", "uppercase": false
    }
  },

  "assets": {
    "img_9f2a...": { "kind": "image", "path": "assets/images/9f2a....png", "sha256": "9f2a...",
                     "provenance": { "provider": "diffusers", "model": "sd-turbo", "seed": 11021 } },
    "mus_7b13...": { "kind": "audio", "path": "assets/music/7b13....mp3", "sha256": "7b13...",
                     "provenance": { "provider": "local", "source_url": "https://...", "licence": "CC-BY-4.0" } }
  }
}
```

## Clips versus photos

A scene's `visual` is a discriminated union. Camera-roll editing uses two members:

```json
{ "kind": "clip",  "asset_id": "vid_3a1f", "source_start": 6.2, "source_end": 8.6,
  "crop": [0.2, 0.0, 0.6, 1.0], "fit": "cover", "speed": 1.0, "mute": true }

{ "kind": "asset", "asset_id": "img_77c2", "fit": "blur_pad" }
```

`source_start` and `source_end` **are** the edit: a fifteen-second phone clip keeps the
two seconds worth keeping. `Scene.duration` must equal `(source_end - source_start) / speed`.

They are kept as separate fields rather than deriving one from the other because
"hold that shot longer" and "show more of that clip" are different requests, and the
edit vocabulary needs to express both.

`mute` defaults to true. Camera-roll audio is usually wind and chatter; the bed is
music unless a clip's own sound is the point.

## Field notes

**`version`** increments on every applied edit. `schema_version` changes only when the
format itself changes, and gates migration.

**`seed`** at the top level seeds everything not explicitly seeded, so a whole video is
reproducible from one number.

**`asset_id: null`** means *unresolved* - the resolve phase must fill it.
A non-null id means the asset exists on disk and must not be regenerated.

**`visual.approved`** protects user-blessed images. The resolver never replaces an
approved visual, even if the prompt changes; the LLM must explicitly unapprove it.

**Asset ids are content hashes.** Two scenes wanting the same image share one file.

**`role`** makes the opening beat explicit. Retention in short form is decided in the
first couple of seconds, so the planner is required to choose a `hook` rather than
beginning with scene one of an essay.

**`style_suffix`** is appended to every generated image prompt. Holding one phrase
constant is the cheapest defence against each shot looking like a different video.
It narrows drift; it does not eliminate it ('Character and scene consistency' in docs/07-backlog.md).

**`safe_area`** marks the fractions of the canvas covered by platform UI. Text placed
outside those bounds is not read, so the renderer treats it as a hard constraint on
overlay and caption placement.

## Music state machine

This encodes the rule that nothing is downloaded without consent.

```
  none  --LLM describes-->  suggested  --provider searches-->  candidates_ready
                                                                      |
                                                    user picks one     |
                                                                      v
                                                                  approved
                                                                      |
                                                       download        |
                                                                      v
                                                                 resolved
                                                                      |
                                              user declines / removes  |
                                                                      v
                                                                  declined
```

The renderer accepts `resolved` (mix the track) and `declined` / `none` (render silent).
Any other state is a hard error - it means the pipeline tried to render before the
user was asked. **There is no path from `suggested` to a downloaded file that skips
`approved`.**

## Export variants

A short-form feed weights reach toward *attached* trending audio. A track baked into
the file is not an attached sound: it forfeits that signal and risks Content-ID
muting. So the default emits two cuts from one video pass:

| Variant | Use |
| --- | --- |
| `full` | Narration and music mixed. Good for platforms without an in-app sound library. |
| `narration_only` | Upload this where a trending sound will be attached in the app. |
| `silent` | Video only, for adding a full audio bed elsewhere. |

The variants share the video render and differ only in the audio mux, so the extra
outputs are close to free.

## Edit operations

On a change request the LLM returns `{"ops": [...]}` against this closed vocabulary.
It never returns a whole timeline. Each op is a discriminated union member validated
by pydantic; an unknown `op` value is rejected outright.

| Op | Fields | Effect |
| --- | --- | --- |
| `set_scene_duration` | `scene_id`, `seconds` | Retime one scene |
| `scale_all_durations` | `factor` | "make the whole thing faster" |
| `set_visual_prompt` | `scene_id`, `prompt`, `negative_prompt?` | Clears `asset_id`, forcing regeneration |
| `set_clip_span` | `scene_id`, `source_start`, `source_end` | Retrim; adjusts scene duration to match |
| `nudge_clip_span` | `scene_id`, `seconds` | Shift the trim, keeping its length and the beat alignment |
| `set_clip_speed` | `scene_id`, `speed` | Slow-motion or speed-up |
| `set_crop` | `scene_id`, `crop?`, `fit?` | Reframe inside the canvas |
| `set_clip_muted` | `scene_id`, `mute` | Unmute a clip whose own audio matters |
| `replace_source` | `scene_id`, `asset_id` | Use a different take, keeping timing |
| `set_motion` | `scene_id`, `motion` | Change the camera move |
| `set_narration` | `scene_id`, `text` | Rewrite the line, clears its audio asset |
| `set_overlay_text` | `scene_id`, `index`, `content` | Edit on-screen text |
| `insert_scene` | `after_scene_id?`, `scene` | Add a beat |
| `remove_scene` | `scene_id` | Drop a beat |
| `reorder_scenes` | `scene_ids` | Full reordering; must be a permutation |
| `set_transition` | `scene_id`, `transition` | Change how it enters |
| `set_canvas` | `width`, `height`, `fps?` | Reframe, e.g. 9:16 -> 16:9 |
| `set_music_query` | `query` | Resets music state to `suggested` |
| `select_music_candidate` | `candidate_id` | Moves to `approved` (still requires the download confirmation) |
| `remove_music` | - | Moves to `declined` |
| `set_music_gain` | `gain_db` | Louder/quieter bed |
| `set_captions` | `enabled`, `mode?`, `style_ref?` | Toggle burn-in, or switch to word-level |
| `set_scene_role` | `scene_id`, `role` | Retarget a beat, e.g. promote one to `hook` |
| `set_style_suffix` | `style_suffix` | Restyle every generated visual at once |
| `set_export` | `export` | Change output variants or the loop flag |
| `set_style` | `style_ref`, `patch` | Font, size, colour |

### Validation rules

Applied **before** any op runs; the whole list is rejected atomically if any op fails.

1. Every referenced `scene_id` exists.
2. `reorder_scenes` is an exact permutation of existing ids - no adds, drops, or duplicates.
3. Durations land within `[0.3, 120.0]` seconds.
4. `insert_scene` supplies a complete, schema-valid scene.
5. Canvas dimensions are even (H.264 requirement) and within `[16, 4096]`.
6. `select_music_candidate` names a candidate actually present in `candidates`.

Atomicity matters: a half-applied op list would leave the project in a state the
user never asked for and cannot reason about.

### Fields the LLM may never write

`asset_id`, `sha256`, `path`, `provenance`, `version`, `assets`.

These are set by the resolver from real files on disk. Letting a language model
invent a content hash would silently corrupt asset integrity, so the schema used
for structured output excludes them entirely.
