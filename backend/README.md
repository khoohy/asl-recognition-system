# Backend Plan

This folder is reserved for a future FastAPI-based inference service.

## Planned Purpose

- Expose model inference through HTTP endpoints
- Separate webcam/UI concerns from model-serving concerns
- Support local frontend development without embedding inference logic in the browser

## Expected Future Components

- `app/` or `service/` package for FastAPI routes
- request/response schemas
- model loading service
- health and readiness endpoints
- inference endpoint for uploaded landmark sequences or preprocessed features

## Current Status

- No backend service is implemented yet
- The active inference path remains `python src/main.py --use-wlasl300`
- When backend work starts, it should reuse the existing production model and shared preprocessing assumptions rather than rewriting model logic from scratch

## Refactor Constraint

The current repo keeps inference behavior inside the existing Python application until the API path can be introduced safely and verified.

