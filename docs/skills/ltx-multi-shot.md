# Multi-shot scouting, and building long takes from it

A workflow for LTX local generation: use one multi-shot render to discover a place from
several angles, then grow each angle into a long continuous take.

The split is not stylistic. The LTX prompting guide is explicit that image-to-video
should "prefer a single continuous take unless you intentionally describe a cut away
from that opening image" — so multi-shot is for *finding* a look, and continuous takes
are for *extending* it. Mixing them fights the model.

```
1. SCOUT     one multi-shot text-to-video, 2-4 angles of the same place
2. HARVEST   ltx_extract_shots -> one still per shot
3. GROW      each still seeds an image-to-video continuous take
4. CHAIN     ltx_extract_last_frame -> feed forward for more length
```

## 1. Writing the multi-shot prompt

Cuts are named in prose. Shot lists, numbered beats and screenplay sluglines do not
work unless the cut itself is also described:

- "A hard cut transitions to…"
- "The view cuts to a close-up of…"
- "A match cut connects…"

**Four obligations at every cut**, all four or the shot drifts into a different place:

1. Name the transition.
2. Re-establish the new shot — scale, angle, subjects, lighting.
3. Repeat the same visual identifiers for anything that recurs (a yellow raincoat, the
   same pale dawn light). The model has no other handle on identity.
4. State audio continuity — "the synth score continues across the cut", or that it drops.

Element order within a shot: establish the framing, set the scene, describe the action,
define characters by physical cues, name the camera movement, then audio (dialogue in
quotation marks).

Camera vocabulary the model responds to: *follows · tracks · pans across · circles
around · tilts upward · pushes in / pulls back · overhead view · handheld · over-the-
shoulder*.

**Budget the duration.** 2-4 shots is the useful range, and each needs room to read.
A measured example: three shots requested in a 6s render produced only one detected
cut — 2s per shot was not enough for the model to commit to three. Give roughly 3s per
shot, or drop to two shots.

### Worked example

> A wide establishing shot frames an abandoned greenhouse at dawn, broken panes
> filtering pale light onto overgrown beds, the camera pushing in slowly. A hard cut
> transitions to a low-angle shot from inside, looking up through the shattered roof at
> drifting clouds; the same pale dawn light, the same tangled vines. Another hard cut
> moves to an overhead view of the central aisle, moss splitting the tiles, the camera
> tracking forward.

Same place, three scales, identifiers repeated at each cut.

## 2. Harvesting the shots

```
ltx_extract_shots(video_path="…/ltx2_video_….mp4")
```

Returns one still per shot: the opening frame, then the first frame after each detected
cut. The opening frame is always included — nothing precedes it, so detection alone
would drop the first shot.

Only **hard cuts** are found. Dissolves score below the threshold and are skipped, which
is deliberate: a dissolve has no single representative frame. If a gentle camera move is
being split into false shots, raise `scene_threshold`; if a real cut is missed, lower it.

Fewer stills than shots you wrote usually means the model merged beats — lengthen the
render rather than lowering the threshold, which mostly buys false positives.

## 3. Growing a shot into a long take

Feed a still back in as the first frame, and describe **one continuous take** — no cuts:

```
ltx_generate_video(
    prompt="The camera pushes slowly down the mossy aisle, light shifting…",
    image_path="…_shot02.png",
    duration_seconds=10,
)
```

Keep the prompt consistent with what the still already shows. It fixes the framing,
lighting and subject; the prompt only has to say what *happens next*.

## 4. Chaining for more length

Local **LTX 2.5 has no Extend and no Retake** — verified in `ltx_capabilities.py`. Length
past one render is built by hand-off:

```
ltx_extract_last_frame(video_path="…") -> still
ltx_generate_video(image_path=that_still, …)
```

**Expect drift.** Each hand-off re-encodes through the model; colour and sharpness
degrade, usually visibly by the third or fourth link. Two practical answers:

- **Switch to LTX 2.3**, which has native Extend and Retake, if the seam quality matters
  more than 2.5's image quality. Both checkpoints may already be on disk.
- **Use multi-keyframe** (up to 10 anchors on both checkpoints) instead of chaining, when
  the sequence is known in advance. Anchors constrain the whole span at once rather than
  compounding error link by link.

## Constraints worth knowing before you plan

**Frame rate rounds down.** The VAE temporal grid needs `(frames - 1) % 8 == 0`, so a
duration is exact only when `fps × duration` is divisible by 8. 24 and 48 fps are always
exact; 30fps for 5s delivers 4.83s. Pick durations that divide cleanly, or accept the
trim.

**Aspect ratios snap to /64.** Delivered dimensions can differ from the requested ratio
by up to ~2.5%. 32:9 at 540p is 1344×384, a real ratio of 3.50 against 3.556.

**Cost grows with pixels × frames**, and nothing enforces a VRAM ceiling. On Windows the
driver spills to host RAM rather than raising OOM, so an over-budget render does not
fail — it runs ~100× slower. Scout at 540p, grow the shots you keep at 1080p.
