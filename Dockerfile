FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

WORKDIR /workspace

ENV TZ=Asia/Tokyo
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y \
    sudo \
    build-essential \
    tzdata \
    git \
    vim \
    wget \
    zsh \
    curl \
    libbz2-dev \
    libffi-dev \
    liblzma-dev \
    libncursesw5-dev \
    libreadline-dev \
    libsqlite3-dev \
    libssl-dev \
    libxml2-dev \
    libxmlsec1-dev \
    llvm \
    make \
    tk-dev \
    xz-utils \
    zlib1g-dev \
    python3 \
    python3-pip \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# add asdf install
RUN git clone https://github.com/asdf-vm/asdf.git ~/.asdf --branch v0.11.3
SHELL ["/bin/bash", "-c"]
RUN echo -e '\n. $HOME/.asdf/asdf.sh' >> ~/.bashrc \
    && echo -e '\n. $HOME/.asdf/completions/asdf.bash' >> ~/.bashrc \
    && echo -e '\n. $HOME/.asdf/asdf.sh' >> ~/.zshrc \
    && echo -e '\nfpath=(${ASDF_DIR}/completions $fpath)\nautoload -Uz compinit && compinit' >> ~/.zshrc

# Install Python packages
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

EXPOSE 5006

CMD ["bokeh", "serve", "--show", "/workspace/app.py", "--allow-websocket-origin=*", "--address=0.0.0.0", "--port=5006"]