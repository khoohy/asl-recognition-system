# Test Status

## Current State

The repository does not yet contain a full automated `tests/` suite with formal unit and integration test files.

The current validation flow is script-based:

- `python scripts/evaluation/test_pipeline.py`
- `python src/main.py --use-wlasl300`

The first command is the lightweight sanity check for preprocessing, model construction, and pipeline wiring. The second command is the manual end-to-end webcam verification path.

## Why This Folder Exists

This folder is included now so the repository structure supports future formal tests without another structural refactor.

## Planned Tests

- unit tests for preprocessing helpers
- unit tests for configuration behavior
- integration tests for checkpoint loading
- inference bridge smoke tests
- webcam path smoke tests where hardware access can be mocked

## Current Recommendation

Use the script-based checks for now, and treat formal pytest coverage as a next-stage improvement after backend and API boundaries are introduced.

