<img alt="le_chateau" src="https://github.com/user-attachments/assets/3bd35107-412a-45ac-8eb7-f3ea168215f6" width="350px" align="right"/>

The goal of this project is to fine-tune a LLM to continue Kafka's novel: the Castle, as the author never ended it.

It uses the French copyleft translation of the book available [here](https://ekladata.com/QAPtMO27HuI4V0hLEhOUd3sv0Nw/Kafka-Le-Chateau.pdf).

It is designed for a Nvidia GPU with at least 12 Go VRAM. The [Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is needed.

The base model is [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama_v1.1), a free-to-operate 1.1B model.

The general idea is to slightly overfit the model on Kafka - the Castle, using French literature data sets ([Gallica](https://huggingface.co/datasets/PleIAs/French-PD-Books) and [Gutenberg](https://huggingface.co/datasets/manu/project_gutenberg) projects) for French language stability and vocabulary enrichment.

🚧 This is an ongoing project 🚧

## 🐳 Launch

Build & launch the docker container:

```sh
docker build -t kafka .
docker run -d \
  --name kafka \
  -v ./project:/project \
  -p 8000:8000 \
  --gpus all \
  kafka
```

## ✨ Use

The project is made of a notebook, available at [port 8000](http://localhost:8000) once the container runs, and 2 python scripts.

- From the notebook, download the HF resources (the model and the literature corpus).
- Launch the data preprocessing: `docker exec -it kafka python prepare_data.py`.
- Make sure the VRAM is 100% available for the training (`nvtop`). From a detached `screen`:
    - Launch the first training step: `docker exec -it kafka python train_gallica.py`.
    - The the second step: `docker exec -it kafka python train_kafka.py`.
- Finally, from the notebook, you can then play with the new Kafkaesque model.

<div align="center">
<img width="500" alt="Llama" src="https://github.com/user-attachments/assets/91f06e0b-7c79-4de9-9386-8dab581f8289" />
</div>

