//! `ghostly` CLI — render a single frame or an animated PNG sequence
//! for any character in the registry.
//!
//! Examples:
//!   ghostly oblivion --output out.png
//!   ghostly oblivion --frames 90 --output anim/  # writes anim/0000.png … anim/0089.png
//!   ghostly --list

use std::path::PathBuf;
use std::process::ExitCode;

use ghostly::{characters, apply_emotion, Emotion, FaceState, RenderSettings, Renderer};

const USAGE: &str = "Usage: ghostly <character> [--output PATH] [--frames N] [--particles N] [--size WxH] [--emotion NAME[:0.0-1.0]]\n       ghostly --list\n\nCharacters: oblivion, narrator, utopia, eliza\nEmotions:   joy, triumph, curiosity, passion, calm, awe, warmth, concern";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    }
    if args.iter().any(|a| a == "--list") {
        for name in characters::ALL {
            println!("{name}");
        }
        return ExitCode::SUCCESS;
    }

    // First positional arg is the character name; remaining are flags.
    let mut name: Option<String> = None;
    let mut output: PathBuf = PathBuf::from("out.png");
    let mut frames: usize = 1;
    let mut particles: usize = 12_000;
    let mut width: u32 = 640;
    let mut height: u32 = 360;
    let mut emotion: Option<(Emotion, f32)> = None;

    let mut i = 0;
    while i < args.len() {
        let arg = &args[i];
        if !arg.starts_with("--") && name.is_none() {
            name = Some(arg.clone());
            i += 1;
            continue;
        }
        match arg.as_str() {
            "--output" => {
                output = PathBuf::from(args.get(i + 1).cloned().unwrap_or_default());
                i += 2;
            }
            "--frames" => {
                frames = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(frames);
                i += 2;
            }
            "--particles" => {
                particles = args
                    .get(i + 1)
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(particles);
                i += 2;
            }
            "--size" => {
                if let Some(s) = args.get(i + 1) {
                    if let Some((w, h)) = s.split_once('x') {
                        if let (Ok(w), Ok(h)) = (w.parse(), h.parse()) {
                            width = w;
                            height = h;
                        }
                    }
                }
                i += 2;
            }
            "--emotion" => {
                // `joy` or `passion:0.75` — defaults to full intensity.
                if let Some(s) = args.get(i + 1) {
                    let (name, intensity) = match s.split_once(':') {
                        Some((n, t)) => (n, t.parse().unwrap_or(1.0_f32)),
                        None => (s.as_str(), 1.0_f32),
                    };
                    if let Some(e) = Emotion::parse(name) {
                        emotion = Some((e, intensity.clamp(0.0, 1.0)));
                    } else {
                        eprintln!("unknown emotion: {name:?}");
                        return ExitCode::from(2);
                    }
                }
                i += 2;
            }
            _ => {
                eprintln!("unknown arg: {arg}\n{USAGE}");
                return ExitCode::from(2);
            }
        }
    }

    let Some(name) = name else {
        eprintln!("{USAGE}");
        return ExitCode::from(2);
    };
    let Some(character) = characters::by_name(&name) else {
        eprintln!("unknown character: {name:?}\n{USAGE}");
        return ExitCode::from(2);
    };
    let character = match emotion {
        Some((e, i)) => apply_emotion(&character, e, i),
        None => character,
    };

    let state = FaceState::new(&character, particles, 2.8, 42);
    let settings = RenderSettings {
        width,
        height,
        ..RenderSettings::default()
    };
    let Some(mut renderer) = Renderer::new(settings) else {
        eprintln!("failed to allocate {width}x{height} pixmap");
        return ExitCode::FAILURE;
    };

    if frames == 1 {
        let pixmap = renderer.render(&character, &state, 0.0);
        if let Err(e) = pixmap.save_png(&output) {
            eprintln!("save_png failed: {e}");
            return ExitCode::FAILURE;
        }
        println!("wrote {}", output.display());
    } else {
        // PNG sequence — ./out/0000.png style.
        if let Err(e) = std::fs::create_dir_all(&output) {
            eprintln!("could not create {}: {e}", output.display());
            return ExitCode::FAILURE;
        }
        // First half scattered → materialized, then steady idle.
        let mut state = state;
        let mut renderer = renderer;
        state.scatter(&character, 0.0);
        let dt = 1.0 / 30.0;
        for f in 0..frames {
            let t = f as f32 * dt;
            // Materialize over the first second, then hold.
            let target = if t < 1.0 { t } else { 1.0 };
            state.step(&character, target, dt);
            state.step_gaze(t, dt);
            if let Some(cfg) = character.render_config.embers {
                state.step_embers(&cfg, dt, 5.0);
            }
            let pixmap = renderer.render(&character, &state, t);
            let path = output.join(format!("{f:04}.png"));
            if let Err(e) = pixmap.save_png(&path) {
                eprintln!("save_png {} failed: {e}", path.display());
                return ExitCode::FAILURE;
            }
        }
        println!("wrote {frames} frames to {}", output.display());
    }

    ExitCode::SUCCESS
}
