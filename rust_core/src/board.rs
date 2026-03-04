use std::fmt;

/// 8x8 Block Blast Board represented as a single 64-bit integer.
///
/// Mapping:
/// Bit 0 (LSB) = Top-Left (Row 0, Col 0)
/// Bit 7 = Top-Right (Row 0, Col 7)
/// Bit 56 = Bottom-Left (Row 7, Col 0)
/// Bit 63 (MSB) = Bottom-Right (Row 7, Col 7)
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct Bitboard(pub u64);

impl Bitboard {
    /// Creates an empty board
    pub const fn empty() -> Self {
        Bitboard(0)
    }

    /// Checks if a piece mask collides with the board or goes out of bounds.
    /// Note: `piece_mask` should already be strictly bounded within the 8x8 grid.
    #[inline(always)]
    pub fn can_place(&self, piece_mask: u64) -> bool {
        (self.0 & piece_mask) == 0
    }

    /// Places a piece mask onto the board.
    #[inline(always)]
    pub fn place(&mut self, piece_mask: u64) {
        self.0 |= piece_mask;
    }

    /// Removes a piece mask from the board.
    #[inline(always)]
    pub fn remove(&mut self, piece_mask: u64) {
        self.0 &= !piece_mask;
    }

    /// Checks for and clears any full rows or columns.
    /// Returns a tuple of (lines_cleared, new_board).
    /// Highly optimized bit-level operations.
    pub fn clear_lines(&self) -> (usize, Bitboard) {
        let mut board = self.0;
        let mut clear_mask = 0u64;
        let mut lines = 0;

        // 1. Check Rows
        // A row is full if all 8 bits are 1. The mask for an 8-bit row is 0xFF.
        let mut row_mask: u64 = 0xFF;
        for _ in 0..8 {
            if (board & row_mask) == row_mask {
                clear_mask |= row_mask;
                lines += 1;
            }
            row_mask <<= 8;
        }

        // 2. Check Columns
        // A column is full if every 8th bit is 1. Mask for Col 0 is 0x0101010101010101.
        let mut col_mask: u64 = 0x0101010101010101;
        for _ in 0..8 {
            if (board & col_mask) == col_mask {
                clear_mask |= col_mask;
                lines += 1;
            }
            col_mask <<= 1;
        }

        // 3. Apply Clear
        board &= !clear_mask;

        (lines, Bitboard(board))
    }

    /// Returns the number of occupied cells
    #[inline(always)]
    pub fn occupied_cells(&self) -> u32 {
        self.0.count_ones()
    }
}

impl fmt::Debug for Bitboard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "Bitboard:")?;
        for r in 0..8 {
            for c in 0..8 {
                let bit = 1u64 << (r * 8 + c);
                if (self.0 & bit) != 0 {
                    write!(f, "[]")?;
                } else {
                    write!(f, " .")?;
                }
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_empty() {
        let board = Bitboard::empty();
        assert_eq!(board.0, 0);
        assert_eq!(board.occupied_cells(), 0);
    }

    #[test]
    fn test_place_and_collision() {
        let mut board = Bitboard::empty();
        let piece_mask = 0x0F; // 1x4 horizontal at top-left

        assert!(board.can_place(piece_mask));
        board.place(piece_mask);
        assert_eq!(board.0, 0x0F);
        assert_eq!(board.occupied_cells(), 4);

        // Cannot place overlapping piece
        assert!(!board.can_place(0x01));

        // Can place elsewhere
        let piece2_mask = 0x0101; // 2x1 vertical at col 0, starting row 1
        assert!(board.can_place(piece2_mask << 8));
    }

    #[test]
    fn test_row_clear() {
        let mut board = Bitboard::empty();
        // Fill row 0 and row 2 completely
        let row0 = 0xFF;
        let row2 = 0xFF_u64 << 16;
        board.place(row0 | row2 | 0x01_u64 << 8); // Add one stray bit in row 1

        let (lines, new_board) = board.clear_lines();
        assert_eq!(lines, 2);
        assert_eq!(new_board.0, 0x01_u64 << 8); // Only stray bit remains
    }

    #[test]
    fn test_col_clear() {
        let mut board = Bitboard::empty();
        // Fill col 0 completely
        let col0 = 0x0101010101010101;
        board.place(col0 | 0x02); // Add stray bit in col 1, row 0

        let (lines, new_board) = board.clear_lines();
        assert_eq!(lines, 1);
        assert_eq!(new_board.0, 0x02); // Only stray bit remains
    }

    #[test]
    fn test_intersecting_clear() {
        let mut board = Bitboard::empty();
        let row0: u64 = 0xFF;
        let col0: u64 = 0x0101010101010101;
        board.place(row0 | col0);

        // They intersect at (0, 0). Should count as 2 lines cleared.
        let (lines, new_board) = board.clear_lines();
        assert_eq!(lines, 2);
        assert_eq!(new_board.0, 0); // Everything cleared
    }
}
