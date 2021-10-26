FROM ubuntu:20.04

# Timezone Configuration
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install common programs
RUN apt-get update && apt-get install -y libreadline-dev \
                                        wget \
                                        unzip \
                                        curl \
                                        git \
                                        vim \
                                        sudo \
                                        software-properties-common \
                                        libmatio-dev

# Work around torch dependencies
RUN add-apt-repository ppa:rock-core/qt4 && apt-get update && \
    git clone https://github.com/torch/distro.git ~/torch --recursive && \
    cd ~/torch && sed -i "s/python-software-properties/software-properties-common/g" install-deps && \
    sed -i "s/ipython/python3-ipython/g" install-deps && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y keyboard-configuration && \
    bash install-deps 2> /dev/null || true

# Torch installation
RUN cd ~/torch && /bin/sh -c "./install.sh"

RUN echo "luarocks install matio" >> ~/.bashrc && /bin/bash -c "source ~/.bashrc" 

RUN apt-get install -y python3-pip && pip3 install numpy \
                                                scipy \
                                                matplotlib \
                                                vtk