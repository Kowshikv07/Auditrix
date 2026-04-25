FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    DEBIAN_FRONTEND=noninteractive \
    PORT=7860

RUN apt-get update && apt-get install -y \
    python3.11 python3.11-dev python3-pip \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel
RUN pip install -r requirements.txt

# Training deps on top of server deps
RUN pip install \
    "torch==2.4.0" "torchvision==0.19.0" \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install \
    "transformers==4.51.3" \
    "trl==0.19.1" \
    "peft==0.14.0" \
    "accelerate==0.34.2" \
    "bitsandbytes==0.44.1" \
    "datasets==3.1.0" \
    "huggingface_hub>=0.24.0"

COPY . .
RUN pip install -e . --no-deps

EXPOSE 7860

# Default = server. Override CMD to run training.
# CMD ["uvicorn", "openenv_compliance_audit.server:app", "--host", "0.0.0.0", "--port", "7860"]
CMD ["python", "train.py"]