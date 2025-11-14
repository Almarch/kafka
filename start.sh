
jupyter lab \
    --no-browser \
    --ip=0.0.0.0 \
    --port=8000 &
    
llamafactory-cli web \
    --host 0.0.0.0 \
    --port 7999 \
    --share false &

wait -n