//! Character packs — the forkable, on-disk definition of a ghostly
//! character.
//!
//! A [`Character`] can't be serialized directly: its [`Transition`]
//! carries a boxed displace closure, and its `contour_paths` are
//! hand-authored polylines. So a pack works by **inheritance**: it
//! names a built-in `base` archetype (which supplies the structural
//! bits — contours + transition) and then overrides the tunable,
//! plain-data fields — geometry, palette, render effects — plus the
//! *entire* [`VoiceProfile`] (the audio identity). Everything a third
//! party actually wants to tweak to make "their own" character is data;
//! the bits that must stay code are referenced by name.
//!
//! This is also where lineage lives conceptually: a fork is just a pack
//! whose `base` (or, at the platform layer, a `forkedFrom` reference)
//! points at the character it descends from.
//!
//! ```no_run
//! use ghostly::pack::CharacterPack;
//! // Export a built-in as an editable starting point…
//! let pack = CharacterPack::from_character("oblivion").unwrap();
//! std::fs::write("oblivion.json", pack.to_json_string().unwrap()).unwrap();
//! // …then load a (possibly hand-edited) pack and render it.
//! let pack = CharacterPack::from_file("oblivion.json").unwrap();
//! let character = pack.to_character().unwrap();
//! ```

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::audio::profile::{self, VoiceProfile};
use crate::character::{Character, ContourBaseline, Geometry, Palette, RenderConfig};
use crate::characters;

/// Boxed, sendable error for pack I/O and parsing.
pub type PackError = Box<dyn std::error::Error + Send + Sync>;

/// A forkable character definition. Only `name` and `base` are
/// required; every other field is an optional override on top of the
/// `base` archetype, so a fork can be as small as "same as Oblivion but
/// with a blue palette."
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CharacterPack {
    /// Display name for this character (independent of `base`).
    pub name: String,
    /// Built-in archetype this pack inherits structure from — the
    /// contour polylines and the scatter/transition behaviour, which
    /// can't be expressed as data. Must be a known built-in
    /// ([`characters::ALL`]). Defaults to `"eliza"`.
    #[serde(default = "default_base")]
    pub base: String,
    /// Default resting emotion (e.g. `"calm"`, `"passion"`). Maps to a
    /// [`crate::Emotion`]; drives the idle palette/voice bias.
    #[serde(default = "default_emotion")]
    pub default_emotion: String,
    /// Contour stroke colour (0-255 RGB). Inherited from `base` if absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contour_color: Option<[u8; 3]>,
    /// Face proportions. Inherited from `base` if absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub geometry: Option<Geometry>,
    /// Colour palette. Inherited from `base` if absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub palette: Option<Palette>,
    /// Breathing/stroke baseline. Inherited from `base` if absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contour_baseline: Option<ContourBaseline>,
    /// Render effects (fresnel, embers, nebula, vignette, audio glow…).
    /// Inherited from `base` if absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub render_config: Option<RenderConfig>,
    /// The full voice DSP profile (formant, pitch, EQ, glitch, comb,
    /// shimmer, gain…). This is the character's *audio* identity.
    /// Inherited from `base`'s default profile if absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub voice: Option<VoiceProfile>,
}

fn default_base() -> String {
    "eliza".to_string()
}

fn default_emotion() -> String {
    "calm".to_string()
}

/// The default resting emotion a built-in character ships with — mirrors
/// [`profile::for_character`]'s character→emotion mapping.
fn emotion_for(name: &str) -> &'static str {
    match name {
        "utopia" => "joy",
        "oblivion" => "passion",
        "eliza" => "curiosity",
        _ => "calm",
    }
}

/// Leak an owned name into a `&'static str`. A character is built once
/// and lives for the session, so a single small leak per loaded pack is
/// acceptable (and avoids threading a lifetime through the renderer).
fn leak_name(s: &str) -> &'static str {
    Box::leak(s.to_string().into_boxed_str())
}

impl CharacterPack {
    /// Export a built-in character as an editable pack — the starting
    /// point for forking. `None` if the name isn't a known built-in.
    pub fn from_character(name: &str) -> Option<Self> {
        let c = characters::by_name(name)?;
        Some(Self {
            name: name.to_string(),
            base: name.to_string(),
            default_emotion: emotion_for(name).to_string(),
            contour_color: Some(c.contour_color),
            geometry: Some(c.geometry),
            palette: Some(c.palette),
            contour_baseline: Some(c.contour_baseline),
            render_config: Some(c.render_config.clone()),
            voice: Some(profile::for_character(name)),
        })
    }

    /// Materialize this pack into a renderable [`Character`]: start from
    /// the `base` archetype (for contours + transition) and apply every
    /// override present. `None` if `base` isn't a known built-in.
    pub fn to_character(&self) -> Option<Character> {
        let mut c = characters::by_name(&self.base)?;
        c.name = leak_name(&self.name);
        if let Some(v) = self.geometry {
            c.geometry = v;
        }
        if let Some(v) = self.palette {
            c.palette = v;
        }
        if let Some(v) = self.contour_color {
            c.contour_color = v;
        }
        if let Some(v) = self.contour_baseline {
            c.contour_baseline = v;
        }
        if let Some(v) = &self.render_config {
            c.render_config = v.clone();
        }
        Some(c)
    }

    /// The voice DSP profile for this pack — the explicit `voice`
    /// override if present, otherwise the `base` archetype's default.
    pub fn voice_profile(&self) -> VoiceProfile {
        self.voice.unwrap_or_else(|| profile::for_character(&self.base))
    }

    /// Whether `base` resolves to a known built-in archetype.
    pub fn base_is_known(&self) -> bool {
        characters::by_name(&self.base).is_some()
    }

    /// Parse a pack from a JSON string.
    pub fn from_json_str(json: &str) -> Result<Self, PackError> {
        Ok(serde_json::from_str(json)?)
    }

    /// Serialize this pack to pretty JSON.
    pub fn to_json_string(&self) -> Result<String, PackError> {
        Ok(serde_json::to_string_pretty(self)?)
    }

    /// Load a pack from a JSON file on disk.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self, PackError> {
        let bytes = std::fs::read_to_string(path)?;
        Self::from_json_str(&bytes)
    }

    /// Write this pack to a JSON file on disk.
    pub fn to_file(&self, path: impl AsRef<Path>) -> Result<(), PackError> {
        std::fs::write(path, self.to_json_string()?)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builtins_export_and_rebuild() {
        for name in characters::ALL {
            let pack = CharacterPack::from_character(name)
                .unwrap_or_else(|| panic!("{name} should export to a pack"));
            // Round-trips through JSON without losing the structure.
            let json = pack.to_json_string().unwrap();
            let back = CharacterPack::from_json_str(&json).unwrap();
            assert_eq!(back.name, *name);
            assert_eq!(back.base, *name);
            // Rebuilds into a renderable character.
            let character = back.to_character().expect("rebuilds");
            assert_eq!(character.name, *name);
        }
    }

    #[test]
    fn override_changes_only_what_it_names() {
        // A fork of Oblivion that only recolours the palette keeps
        // Oblivion's geometry (the structural identity) intact.
        let base = characters::by_name("oblivion").unwrap();
        let mut pack = CharacterPack::from_character("oblivion").unwrap();
        pack.name = "azure-oblivion".to_string();
        pack.palette = Some(Palette::GHOST);
        let c = pack.to_character().unwrap();
        assert_eq!(c.name, "azure-oblivion");
        assert_eq!(c.palette.base, Palette::GHOST.base);
        // Geometry untouched by the palette-only fork.
        assert_eq!(c.geometry.brow_ridge, base.geometry.brow_ridge);
    }

    #[test]
    fn minimal_pack_inherits_base() {
        // The smallest possible fork: just a name + base, everything
        // else inherited.
        let json = r#"{ "name": "mini", "base": "oblivion" }"#;
        let pack = CharacterPack::from_json_str(json).unwrap();
        assert!(pack.base_is_known());
        let c = pack.to_character().unwrap();
        assert_eq!(c.name, "mini");
        // Inherited Oblivion's palette since none was overridden.
        let oblivion = characters::by_name("oblivion").unwrap();
        assert_eq!(c.palette.base, oblivion.palette.base);
        // Voice falls back to the base archetype's default profile.
        assert_eq!(pack.voice_profile().formant_shift, profile::for_character("oblivion").formant_shift);
    }

    #[test]
    fn unknown_base_fails_cleanly() {
        let json = r#"{ "name": "x", "base": "nonesuch" }"#;
        let pack = CharacterPack::from_json_str(json).unwrap();
        assert!(!pack.base_is_known());
        assert!(pack.to_character().is_none());
    }
}
