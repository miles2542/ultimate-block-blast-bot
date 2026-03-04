//! Input module for Block Blast Bot
//! 
//! Contains input controller for Android device via ADB.

// TODO: Implement input controller
// - controller.rs: ADB command execution
// - gesture.rs: Gesture primitives (tap, swipe, long press)

pub struct InputController;

impl InputController {
    pub fn new() -> Self {
        InputController
    }
}

impl Default for InputController {
    fn default() -> Self {
        Self::new()
    }
}