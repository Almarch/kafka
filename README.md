## launch

```sh
docker build -t torch .
docker run -d \
  --name torch \
  -v ./project:/project \
  -p 8000:80 \
  --gpus all \
  torch
```
