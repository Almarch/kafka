#!/bin/sh

jupyter lab \
    --no-browser \
    --ip=0.0.0.0 \
    --port=8000 \
    --allow-root \
    --ServerApp.token='' \
    --ServerApp.password='' &
PID1=$!

export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=7999
llamafactory-cli webui &
PID2=$!

wait $PID1 $PID2