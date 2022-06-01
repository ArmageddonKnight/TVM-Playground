ARG CUDA_BASE_VERSION=11.6.2
ARG UBUNTU_RELEASE_VERSION=20.04

FROM nvidia/cuda:${CUDA_BASE_VERSION}-devel-ubuntu${UBUNTU_RELEASE_VERSION}

# ==============================================================================
# C++ & Python
# ==============================================================================

ARG LLVM_VERSION=14

COPY quik_fix/docker/install_build_essentials.sh \
     /tmp/install_build_essentials.sh
RUN /tmp/install_build_essentials.sh --llvm ${LLVM_VERSION}
RUN rm -f /tmp/install_build_essentials.sh

# ==============================================================================
# CUDA Toolkit
# ==============================================================================

ARG CUDNN_VERSION=8.4.0.27
ARG TENSORRT_VERSION=8.2.5

COPY quik_fix/docker/install_cuda_toolkit.sh /tmp/install_cuda_toolkit.sh
RUN /tmp/install_cuda_toolkit.sh --cudnn ${CUDNN_VERSION} \
                                 --tensorrt ${TENSORRT_VERSION}
RUN rm -f /tmp/install_cuda_toolkit.sh

# ==============================================================================
# Other Dependencies
# ==============================================================================

RUN pip install pytest

WORKDIR /mnt
