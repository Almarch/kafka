FROM python:3.12.3

RUN pip install --upgrade pip==24.3.1 \
    && pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121

RUN pip install jupyterlab==4.2.5

RUN pip install llamafactory==0.9.3

WORKDIR /project

# disable security (local use)
RUN echo "c.ServerApp.token = ''" >> /etc/jupyter/jupyter_server_config.py

# --- COPY START SCRIPT ---
COPY start.sh /start.sh
RUN chmod +x /start.sh

# --- EXPOSE BOTH SERVICES ---
EXPOSE 8888
EXPOSE 7999

CMD ["/start.sh"]
