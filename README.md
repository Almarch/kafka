## launch

```sh
docker build -t torch .
docker run -d \
  --name torch \
  -v ./project:/project \
  -p 8000:8000 \
  -p 7999:7999 \
  --gpus all \
  torch
```
