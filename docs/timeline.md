# Timeline format

The single source of truth. The LLM writes it, edits patch it, the renderer reads it.
All models are frozen — transforms return new values, nothing mutates.

```json
{
  "version": 2,
  "intent": "30s reel from the trip",
  "canvas": { "width": 1080, "height": 1920, "fps": 30,
              "safe_area": { "top": 0.10, "bottom": 0.20, "left": 0.05, "right": 0.18 } },
  "seed": 20260922,
  "scenes": [
    { "id": "s1", "duration": 2.4, "role": "hook",
      "visual": { "kind": "clip", "asset_id": "vid_3a1f",
                  "source_start": 6.2, "source_end": 8.6, "fit": "cover", "mute": true },
      "motion": { "kind": "still" },
      "overlays": [ { "kind": "text", "content": "day one", "position": "bottom_center" } ],
      "transition_in": { "kind": "fade", "duration": 0.4 } }
  ],
  "music": { "state": "resolved", "asset_id": "mus_7b13", "gain_db": -18.0 },
  "captions": { "enabled": true, "mode": "word" },
  "export": { "audio_variants": ["full", "narration_only"], "snap_cuts_to_beat": true },
  "assets": { "vid_3a1f": { "kind": "video", "path": "assets/clips/...", "sha256": "..." } }
}
```

## The parts that matter

**`source_start` / `source_end` are the edit.** A fifteen-second phone clip keeps the two
seconds worth keeping. `duration` must equal `(source_end - source_start) / speed`. They
are separate fields because "hold that shot longer" and "show more of that clip" are
different requests.

**`role`** (`hook` / `body` / `payoff` / `cta`) forces the planner to choose an opening
deliberately. Retention is decided in the first two seconds.

**`safe_area`** marks the bands covered by platform UI. Text outside them is not read, so
the renderer treats it as a hard constraint.

**`asset_id: null`** means unresolved — the resolve phase must fill it. Non-null means the
file exists and must not be regenerated.

**Asset ids are content hashes.** The same photo twice costs one file.

## Export variants

Short-form feeds weight reach toward *attached* trending audio. A baked-in track is not
an attached sound: it forfeits that signal and risks Content-ID muting. So one video pass
yields several cuts.

| Variant | Use |
| --- | --- |
| `full` | Narration and music mixed |
| `narration_only` | Upload this when you will attach a sound in the app |
| `silent` | Video only |

## Music is a state machine

```
none → suggested → candidates_ready → approved → resolved
                                   ↘ declined
```

Nothing downloads before `approved`, and only an explicit user choice makes that
transition. The renderer accepts `none`, `declined` and `resolved`; any other state is a
hard error, because it means the user was never asked.

## Edit operations

On a change request the LLM returns `{"ops": [...]}` from a closed vocabulary — never a
replacement timeline. Each op is validated against the current timeline, and the whole
list is applied atomically or rejected.

**Clips**: `set_clip_span` · `nudge_clip_span` · `set_clip_speed` · `set_crop` ·
`set_clip_muted` · `replace_source`

**Structure**: `insert_scene` · `remove_scene` · `reorder_scenes` · `set_scene_role` ·
`set_transition`

**Timing**: `set_scene_duration` · `scale_all_durations`

**Look**: `set_canvas` · `set_captions` · `set_style` · `set_style_suffix` ·
`set_overlay_text` · `set_motion`

**Music**: `set_music_query` · `select_music_candidate` · `remove_music` · `set_music_gain`

**Idea mode**: `set_visual_prompt` · `set_narration`

The LLM may never write `asset_id`, `sha256`, `path`, `provenance`, `version` or
`assets` — those come from real files, and a hallucinated hash would corrupt the store.
The schema used for structured output excludes them.
