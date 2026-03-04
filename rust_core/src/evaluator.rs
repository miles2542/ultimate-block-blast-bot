//! Board Evaluation Heuristics
//!
//! Evaluates the "goodness" of a board state using fast bitboard operations.

use crate::board::Bitboard;

/// Tunable weights for the heuristic evaluation function.
/// These will be optimized via a genetic algorithm.
#[derive(Clone, Debug, PartialEq)]
pub struct Weights {
    /// Penalty weight for 1x1 holes (empty cell surrounded by blocks/walls).
    /// Typically a large negative number.
    pub hole_penalty: f32,

    /// Penalty weight for having too many total blocks on the board.
    /// Acts as a survival instinct.
    pub density_penalty: f32,

    /// Reward weight for generating actual game points.
    /// Typically 1.0.
    pub score_reward: f32,

    /// Massive reward weight if the final sequence move cleared a line,
    /// keeping the combo multiplier alive for the next turn.
    pub combo_preservation_reward: f32,

    /// Penalty for jagged surface variance
    pub bumpiness_penalty: f32,

    /// Penalty for deep 1-cell wide trenches
    pub well_depth_penalty: f32,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Profile {
    SafeEndurance,
    Aggressive,
}

impl Weights {
    pub fn new(profile: Profile) -> Self {
        match profile {
            Profile::SafeEndurance => Self {
                hole_penalty: -88.0,
                density_penalty: -3.08,
                score_reward: 5.1,
                combo_preservation_reward: 32.7,
                bumpiness_penalty: -0.31,
                well_depth_penalty: -21.9,
            },
            Profile::Aggressive => Self {
                hole_penalty: -33.9,
                density_penalty: -10.04,
                score_reward: 4.6,
                combo_preservation_reward: 58.9,
                bumpiness_penalty: -6.31,
                well_depth_penalty: -10.8,
            },
        }
    }
}

impl Default for Weights {
    fn default() -> Self {
        Self::new(Profile::SafeEndurance)
    }
}

pub struct Evaluator;

impl Evaluator {
    /// Evaluates a board state and returns a heuristic score.
    /// Higher is better.
    #[inline(always)]
    pub fn evaluate(
        board: &Bitboard,
        lines_cleared_this_turn: usize,
        accumulated_score: u32,
        weights: &Weights,
    ) -> f32 {
        let mut score = 0.0;

        // 1. Reward the raw game points generated
        score += (accumulated_score as f32) * weights.score_reward;

        // 2. Combo Preservation Bonus
        // If the entire sequence managed to clear a line, the combo is kept alive.
        if lines_cleared_this_turn > 0 {
            score += weights.combo_preservation_reward;
        }

        // 3. Density Penalty (Survival)
        let count = board.occupied_cells();
        score += (count as f32) * weights.density_penalty;

        // 4. Hole Penalty (1x1 trapped spaces)
        let holes = Self::calculate_holes(board);
        score += (holes as f32) * weights.hole_penalty;

        // 5. Bumpiness Penalty
        let bumpiness = Self::calculate_bumpiness(board);
        score += (bumpiness as f32) * weights.bumpiness_penalty;

        // 6. Well Depth Penalty
        let wells = Self::calculate_wells(board);
        score += (wells as f32) * weights.well_depth_penalty;

        score
    }

    /// Fast bitboard operation to calculate 1x1 fully enclosed holes.
    /// A hole is an empty cell where all 4 adjacent cells are either
    /// occupied blocks or the edge of the board.
    #[inline(always)]
    pub fn calculate_holes(board: &Bitboard) -> u32 {
        let b = board.0;
        let empty = !b;

        // Check Up (Top edge acts as solid wall, so bitwise OR with 0xFF at row 0)
        let up = (b << 8) | 0x00000000000000FF;

        // Check Down (Bottom edge acts as solid wall, so bitwise OR with 0xFF... at row 7)
        let down = (b >> 8) | 0xFF00000000000000;

        // Check Left (Left edge acts as solid wall, prevent wrapping)
        let left_mask = 0xFEFEFEFEFEFEFEFE;
        let left = ((b << 1) & left_mask) | 0x0101010101010101;

        // Check Right (Right edge acts as solid wall, prevent wrapping)
        let right_mask = 0x7F7F7F7F7F7F7F7F;
        let right = ((b >> 1) & right_mask) | 0x8080808080808080;

        // A cell is a hole if it's empty AND bounded on all 4 sides!
        let holes_mask = empty & up & down & left & right;

        holes_mask.count_ones()
    }

    /// Calculates bumpiness by finding the height of each column and summing the absolute differences.
    #[inline(always)]
    pub fn calculate_bumpiness(board: &Bitboard) -> u32 {
        let b = board.0;
        let mut heights = [0_i32; 8];

        // For each column, find the highest populated bit
        for col in 0..8 {
            let col_mask = 0x0101010101010101 << col;
            let col_bits = b & col_mask;
            if col_bits == 0 {
                heights[col] = 0;
            } else {
                // Find highest set bit in this column
                // counting leading zeros from a u64 tells us how far down from the MSB it is.
                // Row 0 is at the top (LSB or MSB depending on endianness of our representation).
                // In our representation, row * 8 + col. Thus row 7 is bits 56-63 (MSB).
                let highest_bit = 63 - col_bits.leading_zeros();
                let highest_row = highest_bit / 8;
                heights[col] = (highest_row + 1) as i32; // Height from bottom
            }
        }

        let mut bumpiness = 0;
        for i in 0..7 {
            bumpiness += (heights[i] - heights[i + 1]).abs();
        }

        bumpiness as u32
    }

    /// Calculates deep wells (1-cell wide empty columns deeper than 2 blocks).
    #[inline(always)]
    pub fn calculate_wells(board: &Bitboard) -> u32 {
        let b = board.0;
        let mut well_depths_sum = 0;

        for col in 0..8 {
            let col_mask = 0x0101010101010101 << col;
            let empty_cells_in_col = (!b) & col_mask;

            // If the column has no empty cells, it's not a well
            if empty_cells_in_col == 0 {
                continue;
            }

            // Check if both adjacent columns are higher
            let left_wall_present = if col == 0 {
                true
            } else {
                let left_col_mask = 0x0101010101010101 << (col - 1);
                (b & left_col_mask) != 0 // Rough heuristic: adjacent left col is populated
            };

            let right_wall_present = if col == 7 {
                true
            } else {
                let right_col_mask = 0x0101010101010101 << (col + 1);
                (b & right_col_mask) != 0
            };

            if left_wall_present && right_wall_present {
                // Count consecutive empty cells downwards
                let empty_count = empty_cells_in_col.count_ones();
                if empty_count >= 3 {
                    well_depths_sum += empty_count;
                }
            }
        }

        well_depths_sum
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calculate_holes_empty_board() {
        let board = Bitboard::empty();
        assert_eq!(Evaluator::calculate_holes(&board), 0);
    }

    #[test]
    fn test_calculate_holes_center() {
        let mut board = Bitboard::empty();
        // Place a plus shape + wrapped around an empty center
        // Row 1 Col 2: 10
        // Row 2 Col 1: 17
        // Row 2 Col 3: 19
        // Row 3 Col 2: 26
        board.place(1 << 10 | 1 << 17 | 1 << 19 | 1 << 26);
        assert_eq!(Evaluator::calculate_holes(&board), 1);
    }

    #[test]
    fn test_calculate_holes_corner() {
        let mut board = Bitboard::empty();
        // Empty top-left corner (row 0, col 0) surrounded by down and right
        board.place(1 << 1 | 1 << 8); // r0c1 and r1c0
        assert_eq!(Evaluator::calculate_holes(&board), 1);
    }

    #[test]
    fn test_evaluate() {
        let board = Bitboard::empty(); // 0 density, 0 holes
        let weights = Weights::default();
        let score = Evaluator::evaluate(&board, 1, 100, &weights);

        // Expected: 100 * 5.1 + 32.7 (combo) = 542.7
        assert_eq!(score, 542.7);
    }
}
