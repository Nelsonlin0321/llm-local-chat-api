#!/bin/bash
gunicorn --workers=${WORKERS:-3} --threads ${THREADS:-3} --timeout 60 --bind :${PORT:-5000} --worker-class uvicorn.workers.UvicornWorker server:app
# kill -9 `lsof -t -i:5000`