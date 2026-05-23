# ghostly

A Rust port of the particle-face visuals in `~/src/avatar`, intended to
run server-side and feed an H.264 video stream (the same shape the
freeq Eliza tile already broadcasts over MoQ).

## Status

| Character | Status |
| --- | --- |
| **Oblivion** | full port — geometry with horns, fire eyes, ominous-sink transition, red voronoi mesh |
| **Narrator** | placeholder — clean blue ghost stub |
| **Utopia** | placeholder — gold palette + diamond geometry, full globe wire-band port pending |
| **Eliza** | placeholder — first-draft teal palette, freeq-side integration pending |

The placeholder modules' doc comments list what's left to port.

## Quick start

```bash
# Render a single frame.
cargo run --release -- oblivion --output out.png

# Animate the materialization — 90 frames at 30fps.
cargo run --release -- oblivion --frames 90 --output anim/

# Higher particle count + 720p.
cargo run --release -- oblivion --particles 30000 --size 1280x720 --output big.png

# List characters.
cargo run --release -- --list
```

## How it works

The renderer mirrors the avatar JS pipeline:

1. **Procedural depth map** (`src/face/geometry.rs`) — head ellipse +
   gaussian feature peaks (brow, nose, eye sockets, cheekbones, lips,
   chin) + optional horns. Each character tunes ~15 dimensionless
   parameters in [`Geometry`] to reshape the face.
2. **Mask + weighted sampling** (`src/face/generate.rs`) — bake a
   low-res mask and inverse-CDF sample ~12K particles, biased toward
   features.
3. **Particle field render** (`src/render.rs`) — additive premultiplied
   blend onto a `tiny_skia::Pixmap`. A trail fill carries between
   frames so motion smears like the canvas version.
4. **Contour overlay** — facial-feature polylines stroked as glowing
   outer + crisp inner lines, breathing with the character's pulse.
5. **Optional voronoi mesh** — faint hexagonal lattice (a stand-in for
   the JS spec's Voronoi cells; reads the same at distance).

## Mapping to the source

| ghostly | avatar |
| --- | --- |
| `src/character.rs` | `static/viz/face-character.js` schema header |
| `src/face/geometry.rs` | `face-gen.js` (`faceDepth`, `faceMask`, `colorPoint`, `hornDepth`) |
| `src/face/generate.rs` | `face-gen.js` (`generateFace`) |
| `src/characters/oblivion.rs` | `CHARACTERS.oblivion` + `GEOMETRY_PRESETS.oblivion` + `PALETTES.oblivion` + `oblivionContours()` |
| `src/render.rs` | a subset of `face-of-god-face.js` (the per-frame render) |

## TODO (in rough order)

- Port `narrator` + `utopia` displaces and contour functions.
- Wire `RenderConfig.fresnel_intensity` into the particle pass — soft
  rim glow on the field, not just on the contour.
- Real voronoi cells (Lloyd-relaxed, lower count) — the hex stand-in
  works but is regular.
- Audio-reactive particle drift — port the low/mid/high band
  modulation in `face-of-god-face.js:render` (the heart of the avatar
  reactiveness).
- Sentiment-driven palette morphing — port the JS `setEmotion` style
  blend.
- Eliza-specific geometry + integration into `freeq-eliza/src/video.rs`
  as a swappable render backend.

## License

MIT or Apache-2.0 at your option.
