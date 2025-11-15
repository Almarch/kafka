FROM python:3.12.3

RUN pip install --upgrade pip==24.3.1 \
    && pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121

RUN pip install httpx==0.27.2 jupyterlab==4.2.5

RUN pip install "llamafactory[webui]==0.9.3"

WORKDIR /project

COPY start.sh /start.sh
RUN chmod +x /start.sh

EXPOSE 8888
EXPOSE 7999

CMD ["/start.sh"]
