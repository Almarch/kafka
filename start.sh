
jupyter lab \
    --no-browser \
    --ip=0.0.0.0 \
    --port=80 &
    
llamafactory-cli web \
    --host 0.0.0.0 \
    --port 7860 \
    --share false &

wait -n