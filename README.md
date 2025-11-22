# Kafka-trained LLM
<img alt="le_chateau" src="https://github.com/user-attachments/assets/3bd35107-412a-45ac-8eb7-f3ea168215f6" width="350px" align="right"/>

The goal of this project is to fine-tune a LLM to continue Kafka's novel: the Castle, as the author never ended it. It is designed such as French <img src="https://upload.wikimedia.org/wikipedia/commons/c/c3/Flag_of_France.svg" alt="fr" width="20"/> is the model language.

It is designed for a Nvidia GPU with at least 12 Go VRAM. The [Nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is needed. A minimal 32Go RAM is safe for the data preparation step.

It takes as input:

- a free-to-operate base model: [TinyLlama](https://huggingface.co/TinyLlama/TinyLlama_v1.1), a 1.1B parameters with a 2048 tokens context window.
- a French copyleft translation of The Castle by Kafka, available [here](https://ekladata.com/QAPtMO27HuI4V0hLEhOUd3sv0Nw/Kafka-Le-Chateau.pdf).
- a French literature dataset: [Gallica](https://huggingface.co/datasets/PleIAs/French-PD-Books) for French language enrichment and narrative consistency.

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

The project is supported by a notebook, available at [port 8000](http://localhost:8000) once the container runs. This notebook must be used to donwload the HuggingFace resources (the model and the literature corpus), and may then be used to monotor the training and test the yielded models.

## ✨ Curriculum learning

The curriculum of this project is presented in a logical order. However, it is also possible to prepare all data beforehands (`1a`, `1b` and `1c`) then all the training steps (`2a`, `2b` and `2c`).

### Step 1 - French literature aculturation

The goal of this step is to reorient the base model towards a generator of French literature. It consists in a full-weight training over 1M samples of 512 tokens from the Gallica collection. The model should forget its chatbot abilities, its multilinguism and its coding knowledge; to learn about *boudoir intrigues* and French classical literature content and form.

- `docker exec -it kafka python 1a_prepare_gallica_fullweight.py`
- `docker exec -it kafka python 2a_train_gallica_fullweight.py`

### Step 2 - Strengthen the narrative arc

This step aims at teaching the model long (2048 tokens) and consistent narrative arcs, which is essential for a literature project. However, because the VRAM need increases quadratically with the context window, a QLoRA approach is undertaken from this step (and for the next one). LoRA adapters are trained over 250M samples of 2048, still from the Gallica collection. At the end ot this step, the model has seen 1B tokens of French literature.

- `docker exec -it kafka python 1b_prepare_gallica_QLoRA.py`
- `docker exec -it kafka python 2b_train_gallica_QLoRA.py`

### Step 3 - Stylistic imprinting on Kafka

Finally, the French litterature model and more specifically its previously pre-trained LoRA are fine-tuned on the target book: the French translation of The Castle by Kafka. This step takes as input 2048 token long sequences of the book, with a stride of 512, yielding 4 shuffled "pseudo-epochs" (each token of the book is seen 4 times).

- `docker exec -it kafka python 1c_prepare_kafka.py`
- `docker exec -it kafka python 2c_train_kakfa_QLoRA.py`

<div align="center">
<img width="500" alt="Llama" src="https://github.com/user-attachments/assets/91f06e0b-7c79-4de9-9386-8dab581f8289" />
</div>

