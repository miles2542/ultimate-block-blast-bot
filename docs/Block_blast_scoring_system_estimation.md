# Block Blast Scoring Engine: Technical Reference

This document provides the mathematical specifications and (**rough initial ideas**) optimized implementation for calculating scores in the autonomous Block Blast engine.

---

### 1. Mathematical Formula

The score for a single move is defined as the sum of the points from the block's cell count and the bonus points from cleared lines.

$$S_{move} = N + \lfloor (26.4 \times M_L \times M_C) + 0.5 \rfloor$$

#### Notation:
*   **$N$**: Number of cells in the block placed (e.g., a 2x2 square is 4).
*   **$L$**: Number of lines cleared in the move ($0 \le L \le 6$).
*   **$C$**: Current combo count at the time of the move.

#### Line Multiplier ($M_L$):
Determines the exponential growth based on simultaneous line clears.
*   If $L = 0 \implies M_L = 0$
*   If $L = 1 \implies M_L = 1$
*   If $L > 1 \implies M_L = L \times (L - 1)$

#### Combo Multiplier ($M_C$):
Determines linear growth adjusted by "reward tiers" at milestones.
*   $M_C = (C + 1) \times \text{TierFactor}$
*   **TierFactor Table:**
    *   $1.0$ when $0 \le C < 5$
    *   $1.5$ when $5 \le C < 10$
    *   $2.0$ when $C \ge 10$

---

### 2. Special Rules & Logic

1.  **Combo Persistence**: 
    *   Clearing $\ge 1$ line in a move **increments** the combo.
    *   Placing a block that clears **0** lines **preserves** the current combo.
    *   The combo **resets to 0** only if an entire "turn" (a set of 3 blocks) is completed without clearing a single line.
2.  **Point Variance**: The game sometimes reward (small) additional points for some board shapes, or some other factors not yet understood. Usually the formula might be off by $\pm 1$ to $3$ points, which is acceptable. 
3.  **Board Clear Bonus**: A full board clear (no cells remaining) typically grants a flat $+300$ bonus. This is calculated after the move score and does not scale with multipliers.

---

### 3. Rust Implementation (Optimized LUT)

For a high-performance bitboard engine, the floating-point math should be avoided during the search. Instead, use a precomputed 2D **Lookup Table (LUT)** stored in the L1 cache.

```rust
/// Precomputed Scoring Lookup Table
/// Rows: Lines Cleared (0-6)
/// Cols: Combo Count (0-63)
pub struct ScoreEngine {
    lut: [[u32; 64]; 7],
}

impl ScoreEngine {
    pub fn new() -> Self {
        let mut lut = [[0u32; 64]; 7];
        for l in 1..7 {
            let m_l = if l == 1 { 1 } else { l * (l - 1) };
            for c in 0..64 {
                let tier_factor = if c < 5 { 1.0 } else if c < 10 { 1.5 } else { 2.0 };
                let m_c = (c as f64 + 1.0) * tier_factor;
                
                // 26.4 is the observed version constant
                lut[l as usize][c] = (26.4 * m_l as f64 * m_c + 0.5).floor() as u32;
            }
        }
        ScoreEngine { lut }
    }

    /// O(1) Score calculation for engine heuristics
    #[inline(always)]
    pub fn calculate_score(&self, lines: usize, combo: usize, cell_count: u32) -> u32 {
        if lines == 0 {
            return cell_count;
        }
        // Cap combo at 63 for table safety
        let safe_c = combo.min(63);
        let safe_l = lines.min(6);
        
        cell_count + self.lut[safe_l][safe_c]
    }
}
```

### 4. Logic for Bitboard Heuristics
When evaluating the "Best Move," the engine should prioritize moves that:
1.  **Maintain/Increase Combo**: Moves that keep the combo alive are worth significantly more than moves that clear more lines but reset the combo later.
2.  **Threshold Jumps**: Aim specifically to cross the $C=5$ and $C=10$ boundaries, as the TierFactor increases the value of every subsequent move by $50\%$ and $100\%$ respectively.
