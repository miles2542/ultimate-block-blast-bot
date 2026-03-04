# Cross-Device Compatibility Guide

To move the bot from your PC to your laptop (or any other machine) while using the existing [calibration.json](file:///x:/Programming/Python/Projects/Applications%20&%20programs/Block%20Blast%20bot/v2%20-%20Final/calibration.json), follow these steps.

## Option A: Sync Window Dimensions (Recommended)

If you don't want to recalibrate, you can force [scrcpy](file:///x:/Programming/Python/Projects/Applications%20&%20programs/Block%20Blast%20bot/v2%20-%20Final/src/block_blast_bot/capture/window_detector.py#94-181) to open at a specific resolution that matches the existing calibration.

For a phone with **1280x2800** resolution (as seen in scrcpy logs), use this command:

```bash
scrcpy --window-title="scrcpy" --stay-awake --video-codec=h265 --video-bit-rate=16M --max-fps=60 --video-buffer=0 --window-width=620 --window-height=1346
```

### Verification & Fine-Tuning
1.  **Open Calibration Tool**: Run `python tools/calibrate.py`.
2.  **Verify Overlay**: Load the existing calibration and turn on **"Show Live Detection"** in the Grid Settings.
3.  **Check Fit**: 
    - If the visual green/red circles fit the board perfectly, you are ready!
    - If they are slightly off, adjust the `--window-width` and `--window-height` in the scrcpy command until they align perfectly.

## Option B: Recalibrate

If syncing dimensions doesn't work or feels too finicky, just recalibrate.

1.  **Run Calibration**: Run `python tools/calibrate.py`.
2.  **Important Note**: The `popup_y_offset` in [config.yaml](file:///x:/Programming/Python/Projects/Applications%20&%20programs/Block%20Blast%20bot/v2%20-%20Final/config.yaml) is **fixed** and not updated by the calibration tool. If the bot fails to "grab" or "place" pieces correctly after a resolution change, you may need to manually adjust this value (e.g., if pieces are being released too early or too late).

## Stats & Performance
- **Highscore**: 4.6 Million+
- **Speed**: Can achieve 100k points in < 2 minutes.

## Future Roadmap
- **YOLO Segmentation**: Transitioning to a calibration-free experience using normalized coordinates (0-1) and device-agnostic segmentation.
- **Streamlined OOB**: Improving the "Out of the Box" experience for new users.
