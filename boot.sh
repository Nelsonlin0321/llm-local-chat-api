#!/bin/bash
gunicorn --workers=${WORKERS:-5} --threads ${THREADS:-2} --timeout 60 --bind :${PORT:-5000} --worker-class uvicorn.workers.UvicornWorker server:app