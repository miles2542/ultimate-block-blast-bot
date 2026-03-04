import time
from block_blast_bot import rust_core


def print_board(board_mask: int):
    print("Board State:")
    for r in range(8):
        row_str = ""
        for c in range(8):
            bit = 1 << ((r * 8) + c)
            if board_mask & bit:
                row_str += "[]"
            else:
                row_str += " ."
        print(row_str)
    print("-" * 20)


def main():
    print("Initializing high-performance Rust Solver...")
    solver = rust_core.Solver()

    # Create a 2x2 hole in an otherwise empty board
    # Actually, let's just make an empty board
    board = 0

    # Let's define 3 pieces:
    # 1. A 3x3 square
    p1 = []
    for r in range(3):
        for c in range(3):
            p1.append((r, c))

    # 2. A 5x1 vertical line
    p2 = [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)]

    # 3. A 1x4 horizontal line
    p3 = [(0, 0), (0, 1), (0, 2), (0, 3)]

    pieces = [p1, p2, p3]

    print("Starting exhaustive search...")
    start_time = time.perf_counter()

    # Run the solver with a 2000ms time limit
    result = solver.solve(board, pieces, combo=0, time_limit_ms=2000)

    end_time = time.perf_counter()
    elapsed_ms = (end_time - start_time) * 1000

    if result:
        print(f"\nSearch complete in {elapsed_ms:.2f} ms")
        print(f"Nodes Evaluated: {result['nodes_evaluated']}")
        print(f"Exhaustive Search: {result['exhaustive']}")
        print(f"Best Sequence (Piece Indices): {result['piece_sequence']}")
        print(f"Heuristic Score: {result['evaluation_score']:.2f}")
        print(f"Game Points Generated: {result['points_generated']}")

        print("\nPlacing pieces...")
        for i, placement in enumerate(result["placements"]):
            print(
                f"Piece {result['piece_sequence'][i]} -> Row {placement['target_row']}, Col {placement['target_col']}"
            )

        print("\nFinal Resulting Board:")
        print_board(result["final_board_mask"])
    else:
        print("No valid placements found for this piece combination.")


if __name__ == "__main__":
    main()
