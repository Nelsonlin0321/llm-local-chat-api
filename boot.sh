#!/bin/bash
gunicorn --workers=${WORKERS:-1} --threads ${THREADS:-1} --timeout 1800 --bind :${PORT:-5000} --worker-class uvicorn.workers.UvicornWorker server:app
# kill -9 `lsof -t -i:5000`