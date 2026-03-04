"""
Block Blast Bot v2.0

A hybrid Rust+Python automation bot for Block Blast puzzle game.
"""

__version__ = "2.0.0"

# Import Rust core module (available after maturin build)
try:
    from block_blast_bot.rust_core import test_function
except ImportError:
    # Rust module not built yet
    def test_function() -> str:
        """Placeholder function when Rust module is not built."""
        return "Rust module not built. Run 'maturin develop' to build."

__all__ = ["__version__", "test_function"]
