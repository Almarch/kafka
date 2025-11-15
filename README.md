<img alt="le_chateau" src="https://github.com/user-attachments/assets/3bd35107-412a-45ac-8eb7-f3ea168215f6" width="275px" align="right"/>

The goal of this project is to fine-tune a LLM to achieve the book of Kafka: the Castle; as the author never ended it.

It works with the French translation of the book available [here](https://ekladata.com/QAPtMO27HuI4V0hLEhOUd3sv0Nw/Kafka-Le-Chateau.pdf) (copyleft).

The project is structured as a docker container serving both a Jupyter notebook and a Llama Factory, sharing a GPU and a common volume. They are available at port [8000](http://localhost:8000) for the notebook and port [7999](http://localhost:7999) for the Llama factory. None of the service are secured: this is intended for local use.


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
