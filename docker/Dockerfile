FROM ubuntu:20.04

# Timezone Configuration
ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt-get update && apt-get install -y libreadline-dev \
                                        wget \
                                        unzip \
                                        curl \
                                        git \
                                        mc \
                                        vim \
                                        sudo \
                                        software-properties-common

RUN add-apt-repository ppa:rock-core/qt4 && apt-get update 

RUN git clone https://github.com/torch/distro.git ~/torch --recursive

RUN cd ~/torch && sed -i "s/python-software-properties/software-properties-common/g" install-deps

RUN apt install -y python3-pip && pip3 install ipython

RUN cd ~/torch && sed -i "s/ipython/python3-ipython/g" install-deps

RUN cd ~/torch && bash install-deps 2> /dev/null || true && source ~/./bashrc