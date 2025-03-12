# Copyright (C) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

FROM ubuntu:24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG KERNEL_HEADERS
ARG KERNEL_HEADERS_GENERIC
ARG KERNEL_MODULES
ARG KERNEL_IMAGE

ARG LIC_MAC=00:00:00:00:00:00
ENV LIC_MAC=${LIC_MAC}
ARG USER_ID
ARG GROUP_ID
ARG GROUP_NAME

RUN apt-get update
RUN apt-get install -y vim git flex bison make
RUN apt-get install -y dpkg-dev libelf-dev zstd wget
RUN apt-get install -y bc debhelper rsync kmod cpio libssl-dev:native
RUN apt-get install -y libelf1t64 kmod linux-base
RUN apt-get install -y cmake jq pkg-config libdrm-dev
RUN apt-get install -y python3 python3-pip ocl-icd-opencl-dev
RUN apt-get install -y sudo adduser perl locales
RUN apt-get install -y uuid-dev libopencv-dev python3-opencv iproute2 python3-venv

RUN git config --global http.postBuffer 157286400

## Install Kernel 6.10
COPY _work /root/
RUN dpkg -i /root/$KERNEL_HEADERS
RUN dpkg -i /root/$KERNEL_HEADERS_GENERIC
RUN dpkg -i /root/$KERNEL_MODULES
RUN dpkg -i /root/$KERNEL_IMAGE

## Compile xdna-driver
RUN git config --global url."https://github.com/".insteadOf "git@github.com:"
RUN cd /root && git clone https://github.com/amd/xdna-driver.git --recursive
RUN cd /root/xdna-driver && git checkout f60608b
RUN cd /root/xdna-driver && git submodule update --recursive

RUN /root/xdna-driver/xrt/src/runtime_src/tools/scripts/xrtdeps.sh -docker

RUN cd /root/xdna-driver/build && ./build.sh -release
RUN cd /root/xdna-driver/build && ./build.sh -package
RUN cd /root/xdna-driver/xrt/build && ./build.sh -noert

## Install xdna-driver
RUN -E dpkg -i /root/xdna-driver/xrt/build/Release/xrt_*-amd64-xrt.deb
RUN -E dpkg -i /root/xdna-driver/build/Release/xrt_plugin*-amdxdna.deb

## Create debs folder
RUN mkdir /root/debs
RUN cd /root/xdna-driver/build && ./build.sh -release
RUN cd /root/xdna-driver/build && ./build.sh -package

## Create npu user
SHELL ["/bin/bash", "-c"]

RUN deluser --remove-home $(getent passwd | awk -F: -v uid="$USER_ID" '$3 == uid {print $1}') || echo "Ignoring error, as uid:${USER_ID} does not exist"
RUN groupadd -g ${GROUP_ID} ${GROUP_NAME} || echo "Ignoring error, as ${GROUP_NAME} already exists"
RUN useradd -u $USER_ID -g $GROUP_ID -ms /bin/bash npu
RUN adduser npu sudo
RUN echo 'npu ALL=(ALL) NOPASSWD: ALL' > /etc/sudoers.d/npu
USER npu
ENV HOME /home/npu
WORKDIR /home/npu

RUN sudo chmod a+rwx /root

ARG DEBIAN_FRONTEND=noninteractive

## Generate locales
RUN sudo locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8
RUN DEBIAN_FRONTEND=noninteractive TZ=Etc/UTC sudo -E apt-get install -y tzdata

SHELL ["/bin/bash", "-l", "-c"]

# Setup the bashrc environment
RUN echo "source /opt/xilinx/xrt/setup.sh" >> ~/.bashrc
RUN sed -i '6,9d' ~/.bashrc # remove interactive only mode in bashrc
RUN echo "export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/xilinx/xrt/lib"

# Install license
COPY Xilinx.lic /opt/

RUN sudo -E echo 'export XILINXD_LICENSE_FILE=/opt/Xilinx.lic' >>  ~/.bashrc
RUN sudo -E echo 'sudo ip link add vmnic0 type dummy || true' >>  ~/.bashrc
RUN sudo -E echo 'sudo ip link set vmnic0 addr ${LIC_MAC} || true' >>  ~/.bashrc

ENV SHELL /bin/bash