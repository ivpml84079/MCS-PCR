FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

ARG EIGEN_VERSION=3.4.0
ARG PCL_VERSION=1.13.0
ARG INSTALL_PREFIX=/usr/local
ARG BUILD_TYPE=Release
ARG LINKER_FLAGS=""

# Build tools + PCL dependencies.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    git \
    ninja-build \
    pkg-config \
    wget \
    unzip \
    libboost-all-dev \
    libflann-dev \
    libqhull-dev \
    libomp-dev \
    libyaml-cpp-dev \
    libgl1-mesa-dev \
    libglu1-mesa-dev \
    libpng-dev \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Python tools & packages
RUN pip3 install --no-cache-dir \
    gdown \
    numpy \
    open3d \
    pyyaml \
    pandas

# -----------------------------------------------------------------------------
# Eigen 3.4.0 from source/header-only install
# -----------------------------------------------------------------------------
RUN wget -q https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION}/eigen-${EIGEN_VERSION}.tar.gz \
    && tar -xzf eigen-${EIGEN_VERSION}.tar.gz \
    && cmake -S eigen-${EIGEN_VERSION} -B eigen-build -G Ninja \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
        -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" \
        -DCMAKE_EXE_LINKER_FLAGS="${LINKER_FLAGS}" \
        -DCMAKE_SHARED_LINKER_FLAGS="${LINKER_FLAGS}" \
        -DBUILD_TESTING=OFF \
    && cmake --build eigen-build \
    && cmake --install eigen-build \
    && rm -rf eigen-${EIGEN_VERSION} eigen-build eigen-${EIGEN_VERSION}.tar.gz

# -----------------------------------------------------------------------------
# PCL 1.13.0 from source, built against the Eigen installed above.
# VTK/visualization/OpenNI are disabled to keep the image smaller.
# -----------------------------------------------------------------------------
RUN wget -q https://github.com/PointCloudLibrary/pcl/archive/refs/tags/pcl-${PCL_VERSION}.tar.gz \
    && tar -xzf pcl-${PCL_VERSION}.tar.gz \
    && cmake -S pcl-pcl-${PCL_VERSION} -B pcl-build -G Ninja \
        -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
        -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DCMAKE_CXX_FLAGS="-O3 -DNDEBUG" \
        -DCMAKE_EXE_LINKER_FLAGS="${LINKER_FLAGS}" \
        -DCMAKE_SHARED_LINKER_FLAGS="${LINKER_FLAGS}" \
        -DEigen3_DIR=${INSTALL_PREFIX}/share/eigen3/cmake \
        -DBUILD_apps=OFF \
        -DBUILD_examples=OFF \
        -DBUILD_global_tests=OFF \
        -DBUILD_tools=OFF \
        -DWITH_VTK=OFF \
        -DWITH_QT=OFF \
        -DWITH_OPENNI=OFF \
        -DWITH_OPENNI2=OFF \
        -DWITH_PCAP=OFF \
        -DWITH_CUDA=OFF \
        -DWITH_DOCS=OFF \
    && cmake --build pcl-build -j$(nproc) \
    && cmake --install pcl-build \
    && ldconfig \
    && rm -rf pcl-pcl-${PCL_VERSION} pcl-build pcl-${PCL_VERSION}.tar.gz

# Avoid Docker BuildKit UndefinedVar warnings by not expanding possibly unset vars.
ENV CMAKE_PREFIX_PATH=${INSTALL_PREFIX}
ENV LD_LIBRARY_PATH=${INSTALL_PREFIX}/lib

WORKDIR /mcs_pcr