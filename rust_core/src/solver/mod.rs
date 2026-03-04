//! Solver module for Block Blast Bot
//!
//! Contains the high-performance solver engine exported to Python via PyO3.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::board::Bitboard;
use crate::evaluator::{Profile, Weights};
use crate::piece::PieceMasks;
use crate::scoring::ScoreEngine;
use crate::search::SearchEngine;

/// The strategy profile the solver uses to evaluate boards
#[pyclass(eq, eq_int)]
#[derive(Clone, Copy, PartialEq)]
pub enum SolverProfile {
    SafeEndurance,
    Aggressive,
}

impl From<SolverProfile> for Profile {
    fn from(prof: SolverProfile) -> Self {
        match prof {
            SolverProfile::SafeEndurance => Profile::SafeEndurance,
            SolverProfile::Aggressive => Profile::Aggressive,
        }
    }
}

#[pyclass]
pub struct Solver {
    weights: Weights,
    score_engine: ScoreEngine,
}

impl Default for Solver {
    fn default() -> Self {
        Self::new(SolverProfile::SafeEndurance)
    }
}

#[pymethods]
impl Solver {
    #[new]
    #[pyo3(signature = (profile=SolverProfile::SafeEndurance))]
    pub fn new(profile: SolverProfile) -> Self {
        Solver {
            weights: Weights::new(Profile::from(profile)),
            score_engine: ScoreEngine::new(),
        }
    }

    /// Change the solver's playstyle dynamically mid-game, or during initialization
    pub fn set_profile(&mut self, profile: SolverProfile) {
        self.weights = Weights::new(Profile::from(profile));
    }

    /// Update the heuristic weights (used during tuning or loading presets)
    pub fn set_weights(
        &mut self,
        hole_penalty: f32,
        density_penalty: f32,
        score_reward: f32,
        combo_preservation_reward: f32,
        bumpiness_penalty: f32,
        well_depth_penalty: f32,
    ) {
        self.weights = Weights {
            hole_penalty,
            density_penalty,
            score_reward,
            combo_preservation_reward,
            bumpiness_penalty,
            well_depth_penalty,
        };
    }

    /// Main entry point for Python pipeline.
    /// `board`: u64 bitmask
    /// `pieces`: list of lists of (r, c) tuples
    /// `combo`: current combo count
    /// `time_limit_ms`: max search time
    #[pyo3(signature = (board_mask, pieces, combo, time_limit_ms=500, lookahead_depth=1, lookahead_samples=5))]
    pub fn solve<'py>(
        &self,
        py: Python<'py>,
        board_mask: u64,
        pieces: &Bound<'py, PyList>,
        combo: usize,
        time_limit_ms: u64,
        lookahead_depth: usize,
        lookahead_samples: usize,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let board = Bitboard(board_mask);
        let mut rust_pieces = Vec::new();

        // Convert Python pieces list to Rust PieceMasks
        for piece_obj in pieces.iter() {
            let py_list = piece_obj.downcast::<PyList>()?;
            let mut coords = Vec::new();

            for coord_obj in py_list.iter() {
                let py_tuple = coord_obj.downcast::<PyTuple>()?;
                let r: i32 = py_tuple.get_item(0)?.extract()?;
                let c: i32 = py_tuple.get_item(1)?.extract()?;
                coords.push((r, c));
            }

            rust_pieces.push(PieceMasks::new(&coords));
        }

        // Create array of references for the search engine
        let piece_refs: Vec<&PieceMasks> = rust_pieces.iter().collect();

        // RUN SEARCH - Release GIL because this is pure CPU bound Rust
        let result = py.allow_threads(|| {
            SearchEngine::find_best_turn(
                board,
                &piece_refs,
                combo,
                &self.score_engine,
                &self.weights,
                time_limit_ms,
                lookahead_depth,
                lookahead_samples,
            )
        });

        // Convert the result back to a nicely formatted Python dict
        if let Some(res) = result {
            let out_dict = PyDict::new_bound(py);

            out_dict.set_item("evaluation_score", res.evaluation_score)?;
            out_dict.set_item("points_generated", res.points_generated)?;
            out_dict.set_item("piece_sequence", res.piece_sequence)?;
            out_dict.set_item("final_board_mask", res.final_board.0)?;
            out_dict.set_item("nodes_evaluated", res.nodes_evaluated)?;
            out_dict.set_item("exhaustive", res.exhaustive)?;

            let placements_list = PyList::empty_bound(py);
            for p in res.placements {
                let p_dict = PyDict::new_bound(py);
                p_dict.set_item("target_row", p.target_row)?;
                p_dict.set_item("target_col", p.target_col)?;
                p_dict.set_item("mask", p.mask)?;
                placements_list.append(p_dict)?;
            }
            out_dict.set_item("placements", placements_list)?;

            Ok(Some(out_dict))
        } else {
            Ok(None)
        }
    }
}
