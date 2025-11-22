FROM python:3.11

RUN pip install --upgrade pip==25.3.0

RUN pip install --no-cache-dir torch==2.4.1+cu121 --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir \
    transformers==4.46.3 \
    datasets==3.1.0 \
    accelerate==1.1.1 \
    peft==0.13.2 \
    bitsandbytes==0.44.1 \
    sentencepiece==0.2.0 \
    protobuf==5.28.3 \
    scipy==1.14.1 \
    jupyterlab==4.2.5 \
    ipywidgets==8.1.5 \
    tqdm==4.67.1 \
    matplotlib==3.10.7

WORKDIR /project

EXPOSE 8000

CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--port=8000", "--allow-root", "--ServerApp.token=", "--ServerApp.password="]
