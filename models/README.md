# Models

## Production model

- Active realtime checkpoint: `models/production/asl_wlasl300_realtime.pt`
- Companion artifacts:
  - `models/production/asl_wlasl300_realtime_history.json`
  - `models/production/asl_wlasl300_realtime_report.json`

This is the default model used by the WLASL300 webcam inference path:

```bash
python src/main.py --use-wlasl300
```

## Legacy fallback model

- Legacy checkpoint kept in place for the older non-WLASL300 path:
  - `models/bilstm_final.pt`

Keep this file where it is until the legacy inference flow is retired or explicitly migrated.

## Archive

- `models/archive/` stores older experiments, alternate checkpoints, validation snapshots, and run reports.
- Files are moved here for traceability. They are not deleted, and they should not be used as the default live model unless you intentionally promote one.

## Changing the active model safely

1. Put the candidate production checkpoint in `models/production/`.
2. Update the default realtime model path in:
   - `src/main.py`
   - `scripts/inference_bridge.py`
3. Keep the file name stable if possible so the webcam command does not change.
4. Verify the model loads before changing anything else:

```bash
python src/main.py --use-wlasl300
```

If you need to preserve compatibility with older commands or docs, keep a small path-resolution shim instead of editing training logic.
