//! Block Blast Search Engine
//!
//! Performs a recursive search through all piece permutations and placements
//! to find the mathematically optimal sequence for a turn.

use crate::board::Bitboard;
use crate::evaluator::{Evaluator, Weights};
use crate::piece::{PieceMasks, Placement};
use crate::scoring::ScoreEngine;
use rand::prelude::*;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct SearchResult {
    /// The heuristic score of this sequence (higher is better)
    pub evaluation_score: f32,
    /// Total actual game points gained during this turn
    pub points_generated: u32,
    /// The order pieces should be played (indices referencing the original array)
    pub piece_sequence: Vec<usize>,
    /// The specific placements cleanly chosen for each piece
    pub placements: Vec<Placement>,
    /// Final board state after all pieces drop and lines clear
    pub final_board: Bitboard,
    /// The final combo streak achieved by the end of this turn
    pub final_combo: usize,
    /// Number of nodes evaluated during this search
    pub nodes_evaluated: u64,
    /// Whether the search completed exhaustively or hit a time limit
    pub exhaustive: bool,
}

pub struct SearchEngine;

impl SearchEngine {
    /// Exhaustively searches for the best turn sequence for a given set of pieces.
    pub fn find_best_turn(
        board: Bitboard,
        pieces: &[&PieceMasks],
        start_combo: usize,
        score_engine: &ScoreEngine,
        weights: &Weights,
        time_limit_ms: u64,
        lookahead_depth: usize,
        lookahead_samples: usize,
    ) -> Option<SearchResult> {
        let mut top_results = Self::find_top_turns(
            board,
            pieces,
            start_combo,
            score_engine,
            weights,
            time_limit_ms,
            10,
        );

        if top_results.is_empty() {
            return None;
        }

        if lookahead_depth == 0 {
            return Some(top_results[0].clone());
        }

        // --- EXPECTIMAX LOOKAHEAD PHASE ---
        let mut rng = rand::rng();
        let all_variants = crate::piece::common_pieces::get_all_variants();
        let top_results_len = top_results.len();

        for candidate in top_results.iter_mut() {
            let mut expected_value_sum = 0.0;
            let mut valid_samples = 0;

            for _ in 0..lookahead_samples {
                let p1 = &all_variants[rng.random_range(0..all_variants.len())];
                let p2 = &all_variants[rng.random_range(0..all_variants.len())];
                let p3 = &all_variants[rng.random_range(0..all_variants.len())];
                let future_pieces = vec![p1, p2, p3];

                // Slice up the remaining time budget among the future simulated branches
                let remaining_time = time_limit_ms
                    .saturating_div((top_results_len * lookahead_samples) as u64)
                    .max(1);

                let future_res = Self::find_best_turn(
                    candidate.final_board,
                    &future_pieces,
                    candidate.final_combo,
                    score_engine,
                    weights,
                    remaining_time,
                    lookahead_depth - 1, // decrement the depth!
                    lookahead_samples,
                );

                if let Some(res) = future_res {
                    // Total Expected Value is the point yield of THIS pending branch + the raw evaluation
                    // of the board at the end of the future branch.
                    let sample_ev = (candidate.points_generated as f32 * weights.score_reward)
                        + res.evaluation_score;
                    expected_value_sum += sample_ev;
                    valid_samples += 1;
                }
            }

            if valid_samples > 0 {
                candidate.evaluation_score = expected_value_sum / (valid_samples as f32);
            } else {
                candidate.evaluation_score = -999999.0; // This branch results in absolute death in all future samples
            }
        }

        // Re-sort current candidates based on expected values after simulated futures
        top_results.sort_by(|a, b| {
            b.evaluation_score
                .partial_cmp(&a.evaluation_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Some(top_results[0].clone())
    }

    /// Internal DFS wrapper that tracks and returns the Top N highest evaluated branches
    pub fn find_top_turns(
        board: Bitboard,
        pieces: &[&PieceMasks],
        start_combo: usize,
        score_engine: &ScoreEngine,
        weights: &Weights,
        time_limit_ms: u64,
        top_n: usize,
    ) -> Vec<SearchResult> {
        if pieces.is_empty() {
            return vec![];
        }

        let mut top_results: Vec<SearchResult> = Vec::with_capacity(top_n + 1);
        let mut nodes_evaluated = 0;
        let mut exhaustive = true;

        let start_time = Instant::now();
        let timeout = Duration::from_millis(time_limit_ms);

        let piece_indices: Vec<usize> = (0..pieces.len()).collect();
        let perms = Self::generate_permutations(&piece_indices);

        for perm in perms {
            let ordered_pieces: Vec<&PieceMasks> = perm.iter().map(|&i| pieces[i]).collect();

            let completed = Self::dfs(
                board,
                &ordered_pieces,
                &perm,
                0, // depth
                start_combo,
                0,      // accumulated points
                0,      // lines cleared this turn
                vec![], // current placements
                score_engine,
                weights,
                &mut top_results,
                top_n,
                &mut nodes_evaluated,
                start_time,
                timeout,
            );

            if !completed {
                exhaustive = false;
                break;
            }
        }

        // Attach global metrics
        for res in top_results.iter_mut() {
            res.nodes_evaluated = nodes_evaluated;
            res.exhaustive = exhaustive;
        }

        top_results
    }

    #[allow(clippy::too_many_arguments)]
    fn dfs(
        board: Bitboard,
        pieces: &[&PieceMasks],
        original_indices: &[usize],
        depth: usize,
        combo: usize,
        accumulated_points: u32,
        lines_cleared_this_turn: usize,
        mut placements: Vec<Placement>,
        score_engine: &ScoreEngine,
        weights: &Weights,
        top_results: &mut Vec<SearchResult>,
        top_n: usize,
        nodes_evaluated: &mut u64,
        start_time: Instant,
        timeout: Duration,
    ) -> bool {
        *nodes_evaluated += 1;

        // Efficient time limit check every 1024 nodes
        if *nodes_evaluated & 0x3FF == 0 && start_time.elapsed() > timeout {
            return false;
        }

        // Base Case: All pieces in this permutation have been placed
        if depth == pieces.len() {
            *nodes_evaluated += 1;

            let eval =
                Evaluator::evaluate(&board, lines_cleared_this_turn, accumulated_points, weights);

            let new_result = SearchResult {
                evaluation_score: eval,
                points_generated: accumulated_points,
                piece_sequence: original_indices.to_vec(),
                placements,
                final_board: board,
                final_combo: combo,
                nodes_evaluated: *nodes_evaluated,
                exhaustive: true, // Placeholder
            };

            if top_results.len() < top_n {
                top_results.push(new_result);
                top_results.sort_by(|a, b| {
                    b.evaluation_score
                        .partial_cmp(&a.evaluation_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            } else if eval > top_results.last().unwrap().evaluation_score {
                top_results.pop(); // kick out the worst result
                top_results.push(new_result);
                top_results.sort_by(|a, b| {
                    b.evaluation_score
                        .partial_cmp(&a.evaluation_score)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }

            return true;
        }

        // Recursive Case: Place piece at `depth`
        let piece = pieces[depth];

        // found_valid_placement is kept for future expansion of forced game over logic
        let mut _found_valid_placement = false;

        for placement in &piece.valid_placements {
            if board.can_place(placement.mask) {
                _found_valid_placement = true;

                let mut new_board = board;
                new_board.place(placement.mask);

                let (lines_cleared, final_board) = new_board.clear_lines();

                let points = score_engine.calculate_score(lines_cleared, combo, piece.cell_count);

                // Update combo logic:
                // If line cleared, increment combo.
                // If 0 lines, combo persists to next piece in same turn.
                let next_combo = if lines_cleared > 0 { combo + 1 } else { combo };
                let next_accumulated_lines = lines_cleared_this_turn + lines_cleared;

                placements.push(placement.clone());

                let completed = Self::dfs(
                    final_board,
                    pieces,
                    original_indices,
                    depth + 1,
                    next_combo,
                    accumulated_points + points,
                    next_accumulated_lines,
                    placements.clone(), // clone the vec of placements
                    score_engine,
                    weights,
                    top_results,
                    top_n,
                    nodes_evaluated,
                    start_time,
                    timeout,
                );

                placements.pop();

                if !completed {
                    return false;
                }
            }
        }

        true
    }

    /// Helper to generate all permutations of indices.
    /// E.g. [0, 1, 2] -> [[0,1,2], [0,2,1], [1,0,2], ...]
    fn generate_permutations(indices: &[usize]) -> Vec<Vec<usize>> {
        if indices.is_empty() {
            return vec![vec![]];
        }
        if indices.len() == 1 {
            return vec![indices.to_vec()];
        }

        let mut all_perms = Vec::new();
        for i in 0..indices.len() {
            let mut sub_indices = indices.to_vec();
            let first = sub_indices.remove(i);

            for mut sub_perm in Self::generate_permutations(&sub_indices) {
                let mut perm = vec![first];
                perm.append(&mut sub_perm);
                all_perms.push(perm);
            }
        }
        all_perms
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::piece::common_pieces;

    #[test]
    fn test_search_simple() {
        let board = Bitboard::empty();
        let engine = ScoreEngine::new();
        let weights = Weights::default();

        // Let's place a 2x2 square and a 1x4 line
        let p1 = common_pieces::square(2);
        let p2 = common_pieces::h_line(4);

        let pieces = vec![&p1, &p2];

        // test with search limit=1000, depth=0, samples=0
        let result = SearchEngine::find_best_turn(board, &pieces, 0, &engine, &weights, 1000, 0, 0);

        assert!(result.is_some());
        let res = result.unwrap();
        // Since both pieces fit easily it shouldn't fail.
        assert_eq!(res.piece_sequence.len(), 2);
        assert_eq!(res.points_generated, 8); // 4 + 4 cells, no lines cleared
        assert!(res.nodes_evaluated > 0);
        assert!(res.exhaustive);
    }
}
