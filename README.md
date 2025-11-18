<img alt="le_chateau" src="https://github.com/user-attachments/assets/3bd35107-412a-45ac-8eb7-f3ea168215f6" width="350px" align="right"/>

The goal of this project is to fine-tune a LLM to continue Kafka's novel: the Castle, as the author never ended it.

It uses the French copyleft translation of the book available [here](https://ekladata.com/QAPtMO27HuI4V0hLEhOUd3sv0Nw/Kafka-Le-Chateau.pdf).

It is designed for a Nvidia GPU with at least 12 Go VRAM. The [Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is needed.

🚨 The base model is [OpenLlama 3B V2](https://huggingface.co/openlm-research/open_llama_3b_v2), which has been trained on NSFW material 🚨 We will try to replace this knowledge with French litteracy but some artifacts may subsist especially when testing the base model as is.

The general idea is to use a QLoRA fine-tuning of the base model on Kafka - the Castle, diluted within French literature from [Project Gutenberg](https://huggingface.co/datasets/manu/project_gutenberg) for stability.

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

The project is made of a notebook, available at [port 8000](http://localhost:8000) once the container runs, and 2 python scritps.

- From the notebook, download the HF resources (the model and the literature corpus).
- Launch the data preprocessing: `docker exec -it kafka python prepare_data.py`.
- Make sure the VRAM is 100% available for the training (`nvtop`).
- In a detached `screen`: Launch the training: `docker exec -it kafka python train.py`.
- Finally, from the notebook, you can then play with the new kafkayan model.

<div align="center">
<img width="500" alt="Llama" src="https://github.com/user-attachments/assets/91f06e0b-7c79-4de9-9386-8dab581f8289" />
</div>

