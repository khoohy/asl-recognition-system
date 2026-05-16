# Frontend Plan

This folder is reserved for a future React-based webcam dashboard.

## Planned Purpose

- Provide a cleaner demo interface for webcam inference
- Display live predictions, confidence scores, and session feedback
- Eventually integrate with a backend inference API instead of embedding model logic in the UI

## Expected Future Components

- React app scaffold
- webcam capture and preview components
- live prediction panel
- session history / analytics widgets
- API integration layer

## Current Status

- No frontend app is implemented yet
- The current user-facing experience is still the desktop webcam pipeline launched from `src/main.py`

## Refactor Constraint

Frontend work should be introduced only after the API boundary and inference contract are clear, so the current refactor intentionally stops at scaffolding and documentation.

