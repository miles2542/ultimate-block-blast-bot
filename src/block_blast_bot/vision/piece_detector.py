"""
Block Blast Bot v2.0 - Piece Detector

Detects game pieces from captured frames.

Key lessons from v1.0:
- Slot pieces are ~50% of grid cell size - use smaller detection window
- Popup pieces are ~100% (same as grid) - use calibration cell size
- Cell counting bug: Use min(width, height) not min(width, height) / 3
- Dark borders between blocks - use morphological closing to bridge gaps
- Popup position is fixed - piece pops up at fixed Y regardless of tap position
- Frame differencing approach: Capture before/after tap, compute difference
"""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from numpy.typing import NDArray

from block_blast_bot.calibration.data import CalibrationData, SlotCalibration
from block_blast_bot.game_types import DetectedPiece, PieceDetectionResult, Rect


# Minimum and maximum cell count for valid pieces (game constraint)
MIN_CELLS = 1
MAX_CELLS = 9

# Morphological kernel size for closing (bridges cell borders)
MORPH_KERNEL_SIZE = 5

# Minimum contour area (in pixels) for valid piece
MIN_CONTOUR_AREA = 100

# Scale factors for slot pieces relative to grid
# On modern phones with 18:9 to 21:9 aspect ratios, slot pieces
# are typically around 41-45% of the main grid cell size.
SLOT_CELL_SCALE = 0.44


class PieceDetector:
    """
    Piece detector using multi-method cascade.

    Methods (in order):
    1. LAB color distance + Otsu thresholding
    2. LAB color distance + hard threshold
    3. Multi-channel Canny edge detection
    4. Texture/local range analysis
    """

    def __init__(
        self,
        calibration: Optional[CalibrationData] = None,
        use_otsu: bool = True,
        hard_threshold: int = 24,
    ) -> None:
        """
        Initialize piece detector.

        Args:
            calibration: Calibration data for grid and slots
            use_otsu: If True, use Otsu thresholding; otherwise use hard threshold
            hard_threshold: LAB distance threshold when not using Otsu
        """
        self._calibration = calibration
        self._use_otsu = use_otsu
        self._hard_threshold = hard_threshold

    def detect_pieces(
        self,
        frame: NDArray[np.uint8],
        region: str = "slots",
    ) -> PieceDetectionResult:
        """
        Detect pieces in the frame.

        Args:
            frame: BGR frame
            region: "slots" for 3 slot pieces, "popup" for active popup piece

        Returns:
            PieceDetectionResult with detected pieces and debug visualization
        """
        if self._calibration is None:
            raise ValueError("Calibration data required for piece detection")

        if region == "slots":
            return self._detect_slot_pieces(frame)
        elif region == "popup":
            return self._detect_popup_piece(frame)
        else:
            raise ValueError(f"Unknown region: {region}")

    def _detect_slot_pieces(
        self,
        frame: NDArray[np.uint8],
    ) -> PieceDetectionResult:
        """
                Detect pieces in slot regions.

                Slot pieces are smaller (~50% of grid cell size), so we use
        a smaller cell size for detection.
        """
        if self._calibration is None or self._calibration.grid is None:
            raise ValueError("Calibration data required")

        # Calculate slot cell size (smaller than grid)
        grid_cell_size = self._calibration.grid.cell_size[0]
        slot_cell_size = int(grid_cell_size * SLOT_CELL_SCALE)

        # Get slot regions
        slot_rects = [
            SlotCalibration(bounds=s.bounds, center=s.center)
            for s in self._calibration.slots
        ]

        all_pieces: List[DetectedPiece] = []
        debug_frames: List[NDArray[np.uint8]] = []
        methods_used = []

        # Process each slot individually to avoid high-contrast pieces
        # skewing the threshold for low-contrast pieces in other slots.
        for i, slot_cal in enumerate(slot_rects):
            slot_pieces: List[DetectedPiece] = []
            slot_debug: Optional[NDArray[np.uint8]] = None

            # Try each detection method in cascade for THIS slot
            for method_name, detect_fn in [
                ("lab_otsu", self._detect_with_lab_otsu),
                ("lab_hard", self._detect_with_lab_hard),
                ("canny", self._detect_with_canny),
            ]:
                pieces, debug_frame = detect_fn(
                    frame, [slot_cal], slot_cell_size, method_name
                )

                if pieces:
                    # Assign slot index to all pieces found in this specific slot
                    for p in pieces:
                        p.slot_index = i

                    slot_pieces = pieces
                    slot_debug = debug_frame
                    if method_name not in methods_used:
                        methods_used.append(method_name)
                    break

            if slot_pieces:
                all_pieces.extend(slot_pieces)
                if slot_debug is not None:
                    debug_frames.append(slot_debug)

        # If no pieces found at all, return failure
        if not all_pieces:
            return PieceDetectionResult(
                pieces=[],
                method="none",
                debug_frame=frame.copy(),
            )

        # Combine debug frames (or just use the first one if we want simplicity)
        combined_debug = (
            self._combine_debug_frames(debug_frames) if debug_frames else frame.copy()
        )

        return PieceDetectionResult(
            pieces=all_pieces,
            method="+".join(methods_used),
            debug_frame=combined_debug,
        )

    def _detect_popup_piece(
        self,
        frame: NDArray[np.uint8],
    ) -> PieceDetectionResult:
        """
        Detect popup piece.

        Popup pieces are same size as grid cells (~100%).
        We also need to handle the "slot ghost" - when a piece pops up,
        the slot becomes empty but may still show in frame differencing.
        """
        if self._calibration is None or self._calibration.grid is None:
            raise ValueError("Calibration data required")

        # Use grid cell size for popup detection
        grid_cell_size = self._calibration.grid.cell_size[0]

        # Get slot regions to mask out (for ghost handling)
        slot_rects = [
            SlotCalibration(bounds=s.bounds, center=s.center)
            for s in self._calibration.slots
        ]

        all_pieces: List[DetectedPiece] = []
        debug_frames: List[NDArray[np.uint8]] = []
        method_used = ""

        # Try each detection method in cascade
        for method_name, detect_fn in [
            ("lab_otsu", self._detect_with_lab_otsu),
            ("lab_hard", self._detect_with_lab_hard),
            ("canny", self._detect_with_canny),
        ]:
            pieces, debug_frame = detect_fn(
                frame, slot_rects, grid_cell_size, method_name, mask_slots=True
            )

            if pieces:
                all_pieces = pieces
                debug_frames.append(debug_frame)
                method_used = method_name
                break

        # If no methods worked, return empty result
        if not all_pieces:
            debug_frame = frame.copy() if len(debug_frames) > 0 else frame.copy()
            return PieceDetectionResult(
                pieces=[],
                method="none",
                debug_frame=debug_frame,
            )

        # Combine debug frames
        combined_debug = (
            self._combine_debug_frames(debug_frames) if debug_frames else frame.copy()
        )

        return PieceDetectionResult(
            pieces=all_pieces,
            method=method_used,
            debug_frame=combined_debug,
        )

    def _detect_with_lab_otsu(
        self,
        frame: NDArray[np.uint8],
        slot_rects: List[SlotCalibration],
        cell_size: int,
        method_name: str,
        mask_slots: bool = False,
    ) -> Tuple[List[DetectedPiece], NDArray[np.uint8]]:
        """Detect using LAB color distance + Otsu thresholding."""
        return self._detect_with_lab(
            frame,
            slot_rects,
            cell_size,
            method_name,
            use_otsu=True,
            mask_slots=mask_slots,
        )

    def _detect_with_lab_hard(
        self,
        frame: NDArray[np.uint8],
        slot_rects: List[SlotCalibration],
        cell_size: int,
        method_name: str,
        mask_slots: bool = False,
    ) -> Tuple[List[DetectedPiece], NDArray[np.uint8]]:
        """Detect using LAB color distance + hard threshold."""
        return self._detect_with_lab(
            frame,
            slot_rects,
            cell_size,
            method_name,
            use_otsu=False,
            mask_slots=mask_slots,
        )

    def _detect_with_lab(
        self,
        frame: NDArray[np.uint8],
        slot_rects: List[SlotCalibration],
        cell_size: int,
        method_name: str,
        use_otsu: bool = True,
        mask_slots: bool = False,
    ) -> Tuple[List[DetectedPiece], NDArray[np.uint8]]:
        """
        Core LAB-based piece detection.

        Algorithm:
        1. Convert to LAB color space
        2. Sample perimeter for background color
        3. Compute Euclidean distance in LAB space
        4. Threshold (Otsu or hard)
        5. Morphological close (bridge cell borders)
        6. Find contours and extract pieces
        """
        # Get detection region (union of all slot bounds)
        region_bounds = self._compute_detection_region(slot_rects)
        if region_bounds is None:
            return [], frame.copy()

        x, y, w, h = region_bounds

        # Extract region from frame
        region = frame[y : y + h, x : x + w]

        # Step 1: Convert to LAB color space
        lab = cv2.cvtColor(region, cv2.COLOR_BGR2LAB)

        # Step 2: Sample perimeter for background color
        bg_lab = self._sample_perimeter(lab)

        # Step 3: Compute Euclidean distance in LAB space
        dist = self._compute_lab_distance(lab, bg_lab)

        # Step 4: Apply threshold
        if use_otsu:
            dist_normalized = cv2.normalize(dist, None, 0, 255, cv2.NORM_MINMAX).astype(
                np.uint8
            )
            _, mask = cv2.threshold(
                dist_normalized, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _, mask = cv2.threshold(
                dist.astype(np.uint8), self._hard_threshold, 255, cv2.THRESH_BINARY
            )

        # Step 5: Morphological close to bridge cell borders
        kernel = np.ones((MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Mask out slot regions if requested (for popup detection)
        if mask_slots:
            mask = self._mask_out_slots(mask, slot_rects, region_bounds)

        # Step 6: Find contours and extract pieces
        pieces = self._extract_pieces_from_mask(mask, region_bounds, cell_size)

        # Create debug visualization
        debug = self._create_debug_visualization(frame, mask, region_bounds, pieces)

        return pieces, debug

    def _detect_with_canny(
        self,
        frame: NDArray[np.uint8],
        slot_rects: List[SlotCalibration],
        cell_size: int,
        method_name: str,
        mask_slots: bool = False,
    ) -> Tuple[List[DetectedPiece], NDArray[np.uint8]]:
        """
        Detect using multi-channel Canny edge detection.

        Note: This method tends to produce loose bounding boxes.
        """
        # Get detection region
        region_bounds = self._compute_detection_region(slot_rects)
        if region_bounds is None:
            return [], frame.copy()

        x, y, w, h = region_bounds
        region = frame[y : y + h, x : x + w]

        # Convert to grayscale
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate to connect edges
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # Mask out slots if requested
        if mask_slots:
            dilated = self._mask_out_slots(dilated, slot_rects, region_bounds)

        # Find contours
        contours, _ = cv2.findContours(
            dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # Extract pieces
        pieces = self._extract_pieces_from_contours(contours, region_bounds, cell_size)

        # Create debug visualization
        debug = self._create_debug_visualization(frame, dilated, region_bounds, pieces)

        return pieces, debug

    def _compute_detection_region(
        self,
        slot_rects: List[SlotCalibration],
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Compute the bounding region that contains all slots.

        Returns:
            Tuple of (x, y, width, height) for detection region
        """
        if not slot_rects:
            return None

        min_x = min(r.bounds[0] for r in slot_rects)
        min_y = min(r.bounds[1] for r in slot_rects)
        max_x = max(r.bounds[0] + r.bounds[2] for r in slot_rects)
        max_y = max(r.bounds[1] + r.bounds[3] for r in slot_rects)

        # Add some padding
        padding = 20
        min_x = max(0, min_x - padding)
        min_y = max(0, min_y - padding)

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def _sample_perimeter(
        self,
        lab: NDArray[np.uint8],
    ) -> NDArray[np.float64]:
        """
        Sample perimeter of region to determine background color.

        Args:
            lab: LAB color space image

        Returns:
            Average LAB color of background
        """
        h, w = lab.shape[:2]

        # Sample from perimeter strips
        strip_w = max(5, w // 10)
        strip_h = max(5, h // 10)

        perimeter_pixels = []

        # Top strip
        perimeter_pixels.append(lab[:strip_h, :].reshape(-1, 3))
        # Bottom strip
        perimeter_pixels.append(lab[h - strip_h :, :].reshape(-1, 3))
        # Left strip
        perimeter_pixels.append(lab[:, :strip_w].reshape(-1, 3))
        # Right strip
        perimeter_pixels.append(lab[:, w - strip_w :].reshape(-1, 3))

        all_pixels = np.vstack(perimeter_pixels).astype(np.float64)

        # Use median for robustness
        bg_lab = np.median(all_pixels, axis=0)

        return bg_lab

    def _compute_lab_distance(
        self,
        lab: NDArray[np.uint8],
        bg_lab: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """
        Compute Euclidean distance in LAB space from background color.

        Args:
            lab: LAB color space image
            bg_lab: Background LAB color

        Returns:
            Distance map
        """
        lab_float = lab.astype(np.float64)
        diff = lab_float - bg_lab
        dist = np.sqrt(np.sum(diff**2, axis=2))

        return dist

    def _mask_out_slots(
        self,
        mask: NDArray[np.uint8],
        slot_rects: List[SlotCalibration],
        region_bounds: Tuple[int, int, int, int],
    ) -> NDArray[np.uint8]:
        """
        Mask out slot regions to handle "slot ghost".

        When a piece pops up, the slot becomes empty but may still show
        in frame differencing. We mask these areas to avoid false positives.

        Args:
            mask: Binary mask
            slot_rects: List of slot bounding rectangles
            region_bounds: Detection region bounds

        Returns:
            Mask with slot regions blacked out
        """
        masked = mask.copy()
        rx, ry, rw, rh = region_bounds

        for slot in slot_rects:
            sx, sy, sw, sh = slot.bounds

            # Convert to region coordinates
            local_x = max(0, sx - rx)
            local_y = max(0, sy - ry)
            local_w = min(sw, rw - local_x)
            local_h = min(sh, rh - local_y)

            if local_w > 0 and local_h > 0:
                # Draw black rectangle to mask out
                cv2.rectangle(
                    masked,
                    (local_x, local_y),
                    (local_x + local_w, local_y + local_h),
                    0,
                    -1,
                )

        return masked

    def _extract_pieces_from_mask(
        self,
        mask: NDArray[np.uint8],
        region_bounds: Tuple[int, int, int, int],
        cell_size: int,
    ) -> List[DetectedPiece]:
        """
        Extract pieces from binary mask using contour analysis.

        Args:
            mask: Binary mask of detected regions
            region_bounds: Detection region bounds (x, y, w, h)
            cell_size: Expected cell size in pixels

        Returns:
            List of detected pieces
        """
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return self._extract_pieces_from_contours(contours, region_bounds, cell_size)

    def _extract_pieces_from_contours(
        self,
        contours: List,
        region_bounds: Tuple[int, int, int, int],
        cell_size: int,
    ) -> List[DetectedPiece]:
        """
        Extract pieces from contours.

        Key insight from v1.0: Use min(width, height) NOT min(width, height) / 3
        for cell size calculation.
        """
        pieces = []
        rx, ry, rw, rh = region_bounds

        for contour in contours:
            # Filter by minimum area
            area = cv2.contourArea(contour)
            if area < MIN_CONTOUR_AREA:
                continue

            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)

            # Validate bounding box
            if not self._is_valid_piece_bbox(w, h, cell_size):
                continue

            # Calculate cells - CRITICAL: use min(w, h), NOT min(w, h) / 3
            # This was a major bug in v1.0!
            cols = max(1, int(round(w / cell_size)))
            rows = max(1, int(round(h / cell_size)))

            # Create mask for this contour
            # The contour is already in region coordinates (0 to rw, 0 to rh)
            # because it was found on the region mask.
            mask = np.zeros((rh, rw), dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)

            # Sample mask at grid points to find filled cells
            cells = []
            for r in range(rows):
                for c in range(cols):
                    cx = x + int((c + 0.5) * cell_size)
                    cy = y + int((r + 0.5) * cell_size)

                    if 0 <= cy < rh and 0 <= cx < rw:
                        if mask[cy, cx] > 0:
                            cells.append((r, c))

            # Validate cell count (1-9 cells per game constraint)
            if len(cells) < MIN_CELLS or len(cells) > MAX_CELLS:
                continue

            # Check if cells form valid connected shape
            if not self._is_connected_shape(cells):
                continue

            # Normalize cells to origin (top-left)
            if cells:
                min_r = min(r for r, c in cells)
                min_c = min(c for r, c in cells)
                cells = [(r - min_r, c - min_c) for r, c in cells]

            # Calculate piece center
            center_x = rx + x + w // 2
            center_y = ry + y + h // 2

            piece = DetectedPiece(
                cells=cells,
                bbox=(rx + x, ry + y, w, h),
                center=(center_x, center_y),
                cell_size=cell_size,
                confidence=1.0,
            )

            pieces.append(piece)

        return pieces

    def _is_valid_piece_bbox(
        self,
        width: int,
        height: int,
        cell_size: int,
    ) -> bool:
        """
        Check if bounding box is valid for a piece.

        Args:
            width: Bounding box width
            height: Bounding box height
            cell_size: Expected cell size

        Returns:
            True if valid
        """
        # Must be at least 1 cell wide and tall
        if width < cell_size * 0.5 or height < cell_size * 0.5:
            return False

        # Must not exceed 5-cell pieces with generous padding
        if width > cell_size * 5.5 or height > cell_size * 5.5:
            return False

        # Aspect ratio should be reasonable (not too elongated, max is 1x5)
        aspect_ratio = max(width, height) / max(min(width, height), 1)
        if aspect_ratio > 6.0:
            return False

        return True

    def _is_connected_shape(
        self,
        cells: List[Tuple[int, int]],
    ) -> bool:
        """
        Check if cells form a connected shape.

        Args:
            cells: List of (row, col) positions

        Returns:
            True if connected
        """
        if not cells:
            return False

        if len(cells) == 1:
            return True

        # Build adjacency set
        cell_set = set(cells)

        # BFS from first cell to see if all cells are reachable
        visited = set()
        queue = [cells[0]]

        while queue:
            r, c = queue.pop(0)

            if (r, c) in visited:
                continue

            visited.add((r, c))

            # Check 8-connected neighbors (including diagonals)
            for dr, dc in [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]:
                nr, nc = r + dr, c + dc
                if (nr, nc) in cell_set and (nr, nc) not in visited:
                    queue.append((nr, nc))

        return len(visited) == len(cells)

    def _create_debug_visualization(
        self,
        frame: NDArray[np.uint8],
        mask: NDArray[np.uint8],
        region_bounds: Tuple[int, int, int, int],
        pieces: List[DetectedPiece],
    ) -> NDArray[np.uint8]:
        """
        Create debug visualization of detection result.

        Args:
            frame: Original frame
            mask: Detection mask
            region_bounds: Detection region
            pieces: Detected pieces

        Returns:
            Debug visualization frame
        """
        debug = frame.copy()

        # Draw detection region
        x, y, w, h = region_bounds
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Draw mask overlay in the exact region location
        mask_colored = np.zeros_like(debug)
        # Place the region mask into the red channel
        mask_colored[y : y + h, x : x + w, 2] = mask
        cv2.addWeighted(debug, 0.7, mask_colored, 0.3, 0, debug)

        # Draw piece bounding boxes and centers
        for i, piece in enumerate(pieces):
            px, py, pw, ph = piece.bbox
            cv2.rectangle(debug, (px, py), (px + pw, py + ph), (0, 255, 255), 2)

            # Draw center
            cx, cy = piece.center
            cv2.circle(debug, (cx, cy), 5, (0, 255, 255), -1)

            # Draw cell grid
            for row, col in piece.cells:
                cell_x = px + int((col + 0.5) * piece.cell_size)
                cell_y = py + int((row + 0.5) * piece.cell_size)
                cv2.circle(debug, (cell_x, cell_y), 3, (0, 255, 0), -1)

            # Label with piece index
            cv2.putText(
                debug,
                f"P{i}",
                (px, py - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                2,
            )

        return debug

    def _combine_debug_frames(
        self,
        frames: List[NDArray[np.uint8]],
    ) -> NDArray[np.uint8]:
        """
        Combine multiple debug frames for visualization.

        Args:
            frames: List of debug frames

        Returns:
            Combined debug frame
        """
        if not frames:
            raise ValueError("No frames to combine")

        if len(frames) == 1:
            return frames[0]

        # Stack frames horizontally
        return np.hstack(frames)

    def detect_slot_pieces(
        self, frame: NDArray[np.uint8], slot_rects: List[Rect]
    ) -> List[DetectedPiece]:
        """
        Detect pieces in specific slot regions (legacy interface).

        Args:
            frame: BGR frame
            slot_rects: List of slot bounding rectangles

        Returns:
            List of detected pieces
        """
        result = self._detect_slot_pieces(frame)
        return result.pieces


def create_piece_detector(
    calibration: Optional[CalibrationData] = None,
) -> PieceDetector:
    """Factory function to create a PieceDetector instance."""
    return PieceDetector(calibration)


__all__ = ["PieceDetector", "create_piece_detector"]
