# Multicam: the same action from several cameras

Re-shoot one clip from other viewpoints so the angles can be intercut. Unlike chaining,
this does not advance time — every camera covers the same moment, which is what makes
the results usable as a cut.

The work is done by the **CrossView Prompt** IC-LoRA, a community adapter that acts as a
virtual second camera: it keeps the subject and staging and moves the viewpoint. Install
it from Settings → Browse IC-LoRAs if `ltx_crossview` reports it missing. Despite the
`LTX2.3` in its filename it lists both 2.3 and 2.5 as supported.

```
1. MASTER    generate the reference clip -- this is camera 1
2. ANGLES    ltx_crossview on it, once per additional camera
3. CUT       all angles share the reference's dimensions, so they intercut
```

## The vocabulary is closed

The adapter was trained on one exact sentence. Paraphrasing weakens it, so
`ltx_crossview` builds the prompt for you from three enumerated choices:

```
crossview. new camera angle: {azimuth}, {elevation}, {distance}.

azimuth    same angle | slightly to the left | slightly to the right
           | to the left | to the right | far to the left | far to the right
elevation  lower | same height | higher
distance   closer | same distance | further
```

## Strength is the difference between a move and no move

Measured on a market-stall reference at 1024×576:

| `lora_strength` | Result |
| --- | --- |
| 1.5 (catalogue default) | Scene perfectly preserved, but the viewpoint barely shifts — "far to the right, lower, closer" returned a frame nearly identical to camera 1 |
| 2.0 (backend maximum) | A genuinely different camera position: foreground traffic changed, the cart recentred, the street perspective visibly rotated |

**Start at 2.0 for multicam.** The default exists for adapters used to nudge a shot, not
to relocate a camera. Values above 2.0 are rejected by the backend outright.

## Match the dimensions, or you cannot cut

`ltx_crossview` sends `resolution_factor: 0`, which is a **sentinel meaning "source
dimensions"** — not "no scaling". This matters more than it looks.

The catalogue's own default of 1.5 takes a different branch that targets a 768 bucket:
a 1024×576 reference came back **576×256**, a smaller frame *and* a different aspect
ratio, useless next to the master in a timeline. With the sentinel, every angle lands at
exactly the reference's size.

## What it does and does not preserve

**Content identity is excellent.** Across angles the market kept the same blue cart, the
same pyramid of oranges, the same striped awning, the same shopfronts and cobblestones.
This is far more reliable than image-to-video chaining, where objects drift within a
single link.

**Temporal sync is approximate.** It re-renders the scene rather than replaying it from
another lens, so background action can differ — a scooter appeared in one angle that was
not in the master. Treat the angles as coverage of the same moment, not as frame-locked
camera feeds. Cut on action at your own risk; cut on beats is safer.

## Cost

`skip_stage_2` is on, so an angle is cheaper than a fresh generation: about 2 minutes for
a 5s 540p clip against roughly 2.5 for the master. Budget one pass per camera — three
angles is three renders.

## Choosing a set

Give the cameras real separation, or the cut will not read. A serviceable three-camera
set from one master:

| Camera | Angle |
| --- | --- |
| 2 | `far to the right`, `lower`, `closer` |
| 3 | `far to the left`, `higher`, `same distance` |
| 4 | `same angle`, `same height`, `closer` — a punch-in on the master |

Neighbouring choices (`slightly to the left` against `same angle`) produce angles too
close to cut between. Reach for the `far` variants first.
