FROM nvcr.io/nvidia/pytorch:25.12-py3

WORKDIR /workspace

RUN apt-get update && apt-get install -y openssh-server && \
    mkdir -p /var/run/sshd && \
    rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

RUN python -m pip install --no-cache-dir \
    transformers \
    datasets \
    accelerate \
    peft \
    trl \
    bitsandbytes \
    wandb \
    scipy

CMD ["bash"]