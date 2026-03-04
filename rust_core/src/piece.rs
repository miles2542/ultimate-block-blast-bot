//! Block Blast Piece Representation
//!
//! Handles generating all valid placements (masks + offsets) for a given piece shape.

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Placement {
    pub mask: u64,
    pub target_row: i32,
    pub target_col: i32,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PieceMasks {
    /// A precalculated list of all valid placements (masks + offsets)
    pub valid_placements: Vec<Placement>,
    /// Number of blocks in this piece (used for scoring)
    pub cell_count: u32,
}

impl PieceMasks {
    /// Generates all valid placements for a given piece shape.
    /// `shape` is a list of (row, col) coordinates relative to any origin.
    pub fn new(shape: &[(i32, i32)]) -> Self {
        if shape.is_empty() {
            return Self {
                valid_placements: vec![],
                cell_count: 0,
            };
        }

        let mut placements = Vec::new();

        // Normalize the shape so the top-leftmost block is at (0, 0)
        let min_r = shape.iter().map(|&(r, _)| r).min().unwrap_or(0);
        let min_c = shape.iter().map(|&(_, c)| c).min().unwrap_or(0);
        let max_r = shape.iter().map(|&(r, _)| r).max().unwrap_or(0);
        let max_c = shape.iter().map(|&(_, c)| c).max().unwrap_or(0);

        let height = max_r - min_r + 1;
        let width = max_c - min_c + 1;

        // If the piece is somehow larger than the board, it has no valid placements.
        if height > 8 || width > 8 {
            return Self {
                valid_placements: vec![],
                cell_count: shape.len() as u32,
            };
        }

        // Iterate over all possible top-left anchors for this shape's bounding box on the 8x8 grid
        for dr in 0..=(8 - height) {
            for dc in 0..=(8 - width) {
                let mut mask: u64 = 0;
                for &(r, c) in shape {
                    let adjusted_r = r - min_r + dr;
                    let adjusted_c = c - min_c + dc;
                    // Bit index: (row * 8) + col
                    mask |= 1u64 << (adjusted_r * 8 + adjusted_c);
                }
                placements.push(Placement {
                    mask,
                    target_row: dr,
                    target_col: dc,
                });
            }
        }

        Self {
            valid_placements: placements,
            cell_count: shape.len() as u32,
        }
    }
}

pub mod common_pieces {
    use super::PieceMasks;

    // --- DOT ---
    pub fn dot() -> PieceMasks {
        PieceMasks::new(&[(0, 0)])
    }

    // --- HORIZONTAL / VERTICAL LINES ---
    pub fn h_line(len: i32) -> PieceMasks {
        let coords: Vec<(i32, i32)> = (0..len).map(|c| (0, c)).collect();
        PieceMasks::new(&coords)
    }

    pub fn v_line(len: i32) -> PieceMasks {
        let coords: Vec<(i32, i32)> = (0..len).map(|r| (r, 0)).collect();
        PieceMasks::new(&coords)
    }

    // --- SQUARES / RECTANGLES ---
    pub fn rect(height: i32, width: i32) -> PieceMasks {
        let mut coords = Vec::new();
        for r in 0..height {
            for c in 0..width {
                coords.push((r, c));
            }
        }
        PieceMasks::new(&coords)
    }

    pub fn square(size: i32) -> PieceMasks {
        rect(size, size)
    }

    // --- TETRIS SHAPES (T, L, J, S, Z) ---

    // T-Piece (4 rotations)
    pub fn t_up() -> PieceMasks {
        PieceMasks::new(&[(0, 0), (0, 1), (0, 2), (1, 1)])
    }
    pub fn t_down() -> PieceMasks {
        PieceMasks::new(&[(0, 1), (1, 0), (1, 1), (1, 2)])
    }
    pub fn t_left() -> PieceMasks {
        PieceMasks::new(&[(0, 1), (1, 0), (1, 1), (2, 1)])
    }
    pub fn t_right() -> PieceMasks {
        PieceMasks::new(&[(0, 0), (1, 0), (1, 1), (2, 0)])
    }

    // S-Piece (2 rotations)
    pub fn s_horizontal() -> PieceMasks {
        PieceMasks::new(&[(0, 1), (0, 2), (1, 0), (1, 1)])
    }
    pub fn s_vertical() -> PieceMasks {
        PieceMasks::new(&[(0, 0), (1, 0), (1, 1), (2, 1)])
    }

    // Z-Piece (2 rotations)
    pub fn z_horizontal() -> PieceMasks {
        PieceMasks::new(&[(0, 0), (0, 1), (1, 1), (1, 2)])
    }
    pub fn z_vertical() -> PieceMasks {
        PieceMasks::new(&[(0, 1), (1, 0), (1, 1), (2, 0)])
    }

    // J-Piece (4 rotations)
    pub fn j_up() -> PieceMasks {
        PieceMasks::new(&[(0, 0), (1, 0), (1, 1), (1, 2)])
    }
    pub fn j_down() -> PieceMasks {
        PieceMasks::new(&[(0, 0), (0, 1), (0, 2), (1, 2)])
    }
    pub fn j_left() -> PieceMasks {
        PieceMasks::new(&[(0, 1), (1, 1), (2, 0), (2, 1)])
    }
    pub fn j_right() -> PieceMasks {
        PieceMasks::new(&[(0, 0), (0, 1), (1, 0), (2, 0)])
    }

    // L-Piece (4 rotations)
    pub fn l_up() -> PieceMasks {
        PieceMasks::new(&[(0, 2), (1, 0), (1, 1), (1, 2)])
    }
    pub fn l_down() -> PieceMasks {
        PieceMasks::new(&[(0, 0), (0, 1), (0, 2), (1, 0)])
    }
    pub fn l_left() -> PieceMasks {
        PieceMasks::new(&[(0, 0), (0, 1), (1, 1), (2, 1)])
    }
    pub fn l_right() -> PieceMasks {
        PieceMasks::new(&[(0, 0), (1, 0), (2, 0), (2, 1)])
    }

    // --- CORNERS (3X3 L-shape / corner shape, 5 cells total) ---
    pub fn corner_top_left() -> PieceMasks {
        PieceMasks::new(&[(0, 0), (0, 1), (0, 2), (1, 0), (2, 0)])
    }
    pub fn corner_top_right() -> PieceMasks {
        PieceMasks::new(&[(0, 0), (0, 1), (0, 2), (1, 2), (2, 2)])
    }
    pub fn corner_bottom_left() -> PieceMasks {
        PieceMasks::new(&[(0, 0), (1, 0), (2, 0), (2, 1), (2, 2)])
    }
    pub fn corner_bottom_right() -> PieceMasks {
        PieceMasks::new(&[(0, 2), (1, 2), (2, 0), (2, 1), (2, 2)])
    }

    // --- SMALL CORNERS (2X2 L-shape, 3 cells total) ---
    pub fn small_corner_tl() -> PieceMasks {
        PieceMasks::new(&[(0, 0), (0, 1), (1, 0)])
    }
    pub fn small_corner_tr() -> PieceMasks {
        PieceMasks::new(&[(0, 0), (0, 1), (1, 1)])
    }
    pub fn small_corner_bl() -> PieceMasks {
        PieceMasks::new(&[(0, 0), (1, 0), (1, 1)])
    }
    pub fn small_corner_br() -> PieceMasks {
        PieceMasks::new(&[(0, 1), (1, 0), (1, 1)])
    }

    // --- DIAGONALS ---
    pub fn diag_2_left() -> PieceMasks {
        PieceMasks::new(&[(0, 0), (1, 1)])
    }
    pub fn diag_2_right() -> PieceMasks {
        PieceMasks::new(&[(0, 1), (1, 0)])
    }
    pub fn diag_3_left() -> PieceMasks {
        PieceMasks::new(&[(0, 0), (1, 1), (2, 2)])
    }
    pub fn diag_3_right() -> PieceMasks {
        PieceMasks::new(&[(0, 2), (1, 1), (2, 0)])
    }

    /// Get an array of all available piece variations logically clustered together
    /// Used for random piece generation in the GA simulation.
    pub fn get_all_variants() -> Vec<PieceMasks> {
        vec![
            dot(),
            // Lines 2-5 H and V
            h_line(2),
            h_line(3),
            h_line(4),
            h_line(5),
            v_line(2),
            v_line(3),
            v_line(4),
            v_line(5),
            // Squares and Rects
            square(2),
            square(3),
            rect(2, 3),
            rect(3, 2),
            // Tetris
            t_up(),
            t_down(),
            t_left(),
            t_right(),
            s_horizontal(),
            s_vertical(),
            z_horizontal(),
            z_vertical(),
            j_up(),
            j_down(),
            j_left(),
            j_right(),
            l_up(),
            l_down(),
            l_left(),
            l_right(),
            // Large Corbers
            corner_top_left(),
            corner_top_right(),
            corner_bottom_left(),
            corner_bottom_right(),
            // Small Corners
            small_corner_tl(),
            small_corner_tr(),
            small_corner_bl(),
            small_corner_br(),
            // Diagonals
            diag_2_left(),
            diag_2_right(),
            diag_3_left(),
            diag_3_right(),
        ]
    }
}
