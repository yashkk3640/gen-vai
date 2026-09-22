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
  "canvas": { "width": 1080, "height": 1920, "fps": 30, "background": "#0B0E14" },
  "seed": 20260922,

  "scenes": [
    {
      "id": "s1",
      "duration": 4.5,
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

  "captions": { "enabled": true, "style_ref": "caption", "source": "narration" },

  "styles": {
    "caption": {
      "font": "Inter-SemiBold", "size_pct": 4.2, "color": "#FFFFFF",
      "stroke_color": "#000000", "stroke_px": 3, "max_chars_per_line": 24
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

## Edit operations

On a change request the LLM returns `{"ops": [...]}` against this closed vocabulary.
It never returns a whole timeline. Each op is a discriminated union member validated
by pydantic; an unknown `op` value is rejected outright.

| Op | Fields | Effect |
| --- | --- | --- |
| `set_scene_duration` | `scene_id`, `seconds` | Retime one scene |
| `scale_all_durations` | `factor` | "make the whole thing faster" |
| `set_visual_prompt` | `scene_id`, `prompt`, `negative_prompt?` | Clears `asset_id`, forcing regeneration |
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
| `set_captions` | `enabled`, `style_ref?` | Toggle burn-in |
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
