# Use NVIDIA CUDA base image with Ubuntu
FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Upgrade pip
RUN python3 -m pip install --upgrade pip setuptools wheel

# Set working directory
WORKDIR /workspace/Robust_MMFM

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
# Install PyTorch with CUDA support first
RUN pip install torch==2.0.1 torchvision==0.15.2 --index-url https://download.pytorch.org/whl/cu117

# Install other requirements
RUN pip install -r requirements.txt

# Install transformers from git as specified in requirements
RUN pip install git+https://github.com/huggingface/transformers@d3cbc997a231098cca81ac27fd3028a5536abe67

# Install robustbench from git as specified in requirements
RUN pip install git+https://github.com/RobustBench/robustbench.git@e67e4225facde47be6a41ed78b576076e8b90cc5

# Install Java JDK 1.8 for CIDEr score computation
RUN wget https://builds.openlogic.com/downloadJDK/openlogic-openjdk/8u422-b05/openlogic-openjdk-8u422-b05-linux-x64.tar.gz && \
    tar -xzf openlogic-openjdk-8u422-b05-linux-x64.tar.gz && \
    mv openlogic-openjdk-8u422-b05-linux-x64 /opt/jdk1.8.0 && \
    rm openlogic-openjdk-8u422-b05-linux-x64.tar.gz

# Set JAVA_HOME
ENV JAVA_HOME=/opt/jdk1.8.0
ENV PATH=$PATH:$JAVA_HOME/bin

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p open_flamingo_datasets/COCO/train2014 \
    open_flamingo_datasets/COCO/val2014 \
    open_flamingo_datasets/COCO_CF \
    open_flamingo_datasets/Flickr30k/Images \
    open_flamingo_datasets/VizWiz/train \
    open_flamingo_datasets/VizWiz/val \
    open_flamingo_datasets/OKVQA \
    clip_train_datasets/MS_COCO/images \
    clip_train_datasets/MS_COCO_APGD_4/images \
    clip_train_datasets/MS_COCO_APGD_1/images \
    clip_train_datasets/MS_COCO_COCO_CF/images \
    image_classification_datasets \
    Results/open_flamingo \
    fine_tuned_clip_models

# Expose port for Gradio app
EXPOSE 7860

# Default command - launch Gradio app
CMD ["python", "gradio/gradio_app.py"]
