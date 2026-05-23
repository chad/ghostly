//! The character registry.
//!
//! Each character file builds and returns a [`crate::character::Character`]
//! — a self-contained bundle the renderer reads. New characters slot
//! in here without changing the renderer.
//!
//! The current line-up matches the avatar reference plus our own:
//!
//! | Character | Status | Personality (qa SYSTEM analogue) |
//! | --- | --- | --- |
//! | [`oblivion`] | full port | predatory, menacing, fire eyes |
//! | [`narrator`] | placeholder | calm blue ghost (avatar default) |
//! | [`utopia`] | placeholder | gold orb, warm, hopeful |
//! | [`eliza`] | placeholder | freeq's cyberpunk presence |

pub mod eliza;
pub mod narrator;
pub mod oblivion;
pub mod utopia;

use crate::character::Character;

/// Build a character by short name. `None` if the name is unknown.
pub fn by_name(name: &str) -> Option<Character> {
    match name.to_ascii_lowercase().as_str() {
        "oblivion" => Some(oblivion::build()),
        "narrator" => Some(narrator::build()),
        "utopia" => Some(utopia::build()),
        "eliza" => Some(eliza::build()),
        _ => None,
    }
}

/// Every character we currently know about. Stable order for CLI
/// listings.
pub const ALL: &[&str] = &["oblivion", "narrator", "utopia", "eliza"];
