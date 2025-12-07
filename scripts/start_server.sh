#!/bin/bash
# Start server script
source .venv/bin/activate
uvicorn main:app --reload

