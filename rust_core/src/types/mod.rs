//! Types module for Block Blast Bot
//! 
//! Contains core type definitions for board, pieces, and moves.

// TODO: Implement core types
// - board.rs: Board representation (u64 bitmask)
// - piece.rs: Piece shape definitions

use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors that can occur in the Rust core
#[derive(Error, Debug)]
pub enum CoreError {
    #[error("Invalid board state")]
    InvalidBoardState,
    
    #[error("Invalid piece shape")]
    InvalidPieceShape,
    
    #[error("No valid moves found")]
    NoValidMoves,
    
    #[error("Solver timeout")]
    SolverTimeout,
}

/// Board representation using a 64-bit bitmask
/// Each bit represents a cell (1 = filled, 0 = empty)
#[derive(Clone, Copy, Debug, Default, Serialize, Deserialize)]
pub struct Board(pub u64);

impl Board {
    /// Create an empty board
    pub fn empty() -> Self {
        Board(0)
    }
    
    /// Check if a cell is filled
    pub fn is_filled(&self, row: usize, col: usize) -> bool {
        if row >= 8 || col >= 8 {
            return false;
        }
        let bit = 1u64 << (row * 8 + col);
        (self.0 & bit) != 0
    }
    
    /// Set a cell as filled
    pub fn set_filled(&mut self, row: usize, col: usize) {
        if row < 8 && col < 8 {
            let bit = 1u64 << (row * 8 + col);
            self.0 |= bit;
        }
    }
}

/// Piece shape defined by relative cell positions
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct PieceShape {
    pub cells: Vec<(i32, i32)>,
}

impl PieceShape {
    /// Create a new piece shape
    pub fn new(cells: Vec<(i32, i32)>) -> Self {
        PieceShape { cells }
    }
    
    /// Get the number of cells in this piece
    pub fn cell_count(&self) -> usize {
        self.cells.len()
    }
}

/// A single move placing a piece on the board
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Move {
    pub piece_index: usize,
    pub target_row: i32,
    pub target_col: i32,
}

/// A chain of moves for a 3-piece set
#[derive(Clone, Debug, Default)]
pub struct MoveChain {
    pub moves: Vec<Move>,
    pub final_score: f32,
    pub line_clears: u32,
}