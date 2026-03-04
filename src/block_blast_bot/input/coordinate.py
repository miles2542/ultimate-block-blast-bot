"""
Block Blast Bot v2.0 - Coordinate System

Coordinate transformation utilities for window-to-device mapping.
"""

from dataclasses import dataclass


@dataclass
class InputCalibration:
    """
    Holds the constants derived from Phase 4 physical calibration
    to handle the game's non-linear piece dragging mechanism.
    """

    popup_y_offset: int = 0
    scale_x: float = 1.0
    scale_y: float = 1.0


def calculate_swipe(
    slot_center_x: int,
    slot_center_y: int,
    dest_center_x: int,
    dest_center_y: int,
    calibration: InputCalibration,
) -> tuple[int, int, int, int]:
    """
    Calculates the exact finger start and end coordinates required to move a piece
    from its slot to the exact destination pixel using the reverse-engineered drag math.

    Args:
        slot_center_x: Absolute pixel X of the piece in the tray.
        slot_center_y: Absolute pixel Y of the piece in the tray.
        dest_center_x: Absolute pixel X of where the piece center SHOULD land on the grid.
        dest_center_y: Absolute pixel Y of where the piece center SHOULD land on the grid.
        calibration: The loaded InputCalibration constants for the device.

    Returns:
        (x1, y1, x2, y2) absolute pixel coordinates for the ADB swipe command.
    """
    # 1. Start coordinate is always exactly where the piece rests.
    x1 = slot_center_x
    y1 = slot_center_y

    # 2. Calculate piece's starting coordinate when initially tapped (the popup origin)
    hover_start_x = slot_center_x
    hover_start_y = slot_center_y + calibration.popup_y_offset

    # 3. How far the piece *itself* must move to reach destination
    piece_travel_dx = dest_center_x - hover_start_x
    piece_travel_dy = dest_center_y - hover_start_y

    # 4. Calculate the required physical *finger* movement using scalars
    finger_travel_dx = piece_travel_dx / calibration.scale_x
    finger_travel_dy = piece_travel_dy / calibration.scale_y

    # 5. Output coordinate is the origin + the physical finger distance
    x2 = int(x1 + finger_travel_dx)
    y2 = int(y1 + finger_travel_dy)

    return (x1, y1, x2, y2)


__all__ = ["InputCalibration", "calculate_swipe"]
