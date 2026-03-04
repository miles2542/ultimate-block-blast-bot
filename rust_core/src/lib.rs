//! Block Blast Bot v2.0 - Rust Core Library
//!
//! High-performance solver and input controller for Block Blast automation.

use pyo3::prelude::*;

pub mod board;
pub mod input;
pub mod piece;
pub mod solver;
pub mod types;

pub mod evaluator;
pub mod scoring;
pub mod search;

/// Test function to verify Rust module is working
#[pyfunction]
fn test_function() -> PyResult<String> {
    Ok("Rust core module loaded successfully!".to_string())
}

/// Block Blast Bot Rust Core Module
#[pymodule]
fn rust_core<'py>(m: &Bound<'py, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(test_function, m)?)?;

    // Export the high-performance Solver and its Config Enum
    m.add_class::<solver::SolverProfile>()?;
    m.add_class::<solver::Solver>()?;

    Ok(())
}
