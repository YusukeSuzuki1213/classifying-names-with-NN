FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \ # for pyenv
         emacs \
	 wget \ # for pyenv
	 llvm \ # for pyenv
	 libncurses5-dev \ # for pyenv
	 libncursesw5-dev \ # for pyenv
	 xz-utils \ # for pyenv
	 tk-dev \ # for pyenv
	 libffi-dev \ # for pyenv
	 liblzma-dev \ # for pyenv
	 python-openssl \ # for pyenv
	 make \ # for pyenv
	 libssl-dev \ # for pyenv
	 zlib1g-dev \ # for pyenv
	 libbz2-dev \ # for pyenv
	 libreadline-dev \ # for pyenv
	 libsqlite3-dev \ # for pyenv
	 libffi-dev && \ # for pyenv
     rm -rf /var/lib/apt/lists/*

RUN apt-get update

# install pyenv
RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.profile
RUN echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.profile
RUN echo -e 'if command -v pyenv 1>/dev/null 2>&1; then\n  eval "$(pyenv init -)"\nfi' >> ~/.profile
RUN source ~/.profile
RUN pyenv install 3.7.4
RUN pyenv install 2.7.16

# default dir
WORKDIR /mnt
RUN pyenv global 3.7.4
RUN pyenv local  3.7.4

