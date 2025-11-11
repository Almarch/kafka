FROM jupyter/base-notebook:python-3.11.6

RUN pip install --upgrade pip==24.3.1 \
    && pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121    
RUN pip install \
    numpy==2.1.3 \
    sentencepiece==0.2.0 \
    tqdm==4.66.5 \
    datasets==3.0.1 \
    transformers==4.44.2 \
    peft==0.13.2 \
    bitsandbytes==0.44.1 \
    accelerate==0.34.2

WORKDIR /project

# disable security (local use)
RUN echo "c.NotebookApp.token = ''" >> /etc/jupyter/jupyter_notebook_config.py

EXPOSE 80
CMD ["jupyter", "lab", "--no-browser", "--ip=0.0.0.0", "--port=80"]
