#!/bin/bash -e

# Configuration
PYTHON_VERSION="3.12"
WORK_DIR=$(pwd)
NPROC=$(nproc)
PACKAGE_URL=https://github.com/microsoft/onnxruntime
PACKAGE_DIR=onnxruntime
PACKAGE_NAME=onnxruntime

# Source toolset
source /opt/rh/gcc-toolset-14/enable

# Get Python include path
PYTHON_INCLUDE=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")

export CPLUS_INCLUDE_PATH="$PYTHON_INCLUDE:$CPLUS_INCLUDE_PATH"
export C_INCLUDE_PATH="$PYTHON_INCLUDE:$C_INCLUDE_PATH"

# Install build dependencies
pip install --upgrade cmake ninja packaging numpy pybind11==2.12.0
pip install onnx --prefer-binary --extra-index-url=https://console.redhat.com/api/pypi/public-rhai/rhoai/3.3/cpu-ubi9/simple/

# Build Abseil-cpp
ABSEIL_VERSION="20240116.2"
ABSEIL_PREFIX="$WORK_DIR/abseil-install"
mkdir -p "$ABSEIL_PREFIX"

git clone https://github.com/abseil/abseil-cpp -b "$ABSEIL_VERSION" --depth 1
cd abseil-cpp
cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_INSTALL_PREFIX="$ABSEIL_PREFIX" \
    -DBUILD_SHARED_LIBS=ON \
    -DABSL_PROPAGATE_CXX_STD=ON \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
cmake --build build --parallel "$NPROC"
cmake --install build
cd "$WORK_DIR"

# Build Protobuf
PROTO_VERSION="v4.25.3"
PROTO_INSTALL="$WORK_DIR/protobuf-install"
mkdir -p "$PROTO_INSTALL"

git clone https://github.com/protocolbuffers/protobuf -b "$PROTO_VERSION" --depth 1
cd protobuf
git submodule update --init --recursive --depth 1

# Replace third-party abseil with built one
rm -rf third_party/abseil-cpp && cp -r "$WORK_DIR/abseil-cpp" third_party/

cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_INSTALL_PREFIX="$PROTO_INSTALL" \
    -Dprotobuf_BUILD_TESTS=OFF \
    -Dprotobuf_BUILD_LIBUPB=OFF \
    -Dprotobuf_BUILD_SHARED_LIBS=ON \
    -Dprotobuf_ABSL_PROVIDER="module" \
    -DCMAKE_PREFIX_PATH="$ABSEIL_PREFIX" \
    -Dprotobuf_JSONCPP_PROVIDER="package" \
    -Dprotobuf_USE_EXTERNAL_GTEST=OFF
cmake --build build --parallel "$NPROC"
cmake --install build

# Build Protobuf Python Package
cd python
export PROTOC="$PROTO_INSTALL/bin/protoc"
export LD_LIBRARY_PATH="$ABSEIL_PREFIX/lib:$PROTO_INSTALL/lib64:$LD_LIBRARY_PATH"

# Apply PowerPC patch
wget -q https://raw.githubusercontent.com/ppc64le/build-scripts/master/p/protobuf/set_cpp_to_17_v4.25.3.patch
git apply set_cpp_to_17_v4.25.3.patch || true
python3 setup.py install --cpp_implementation
cd "$WORK_DIR"

# Build ONNX Runtime
git clone "$PACKAGE_URL" --depth 1 -b "v$ONNXRUNTIME_VERSION"
cd "$PACKAGE_DIR"

# Optimization flags
export CFLAGS="-O3 -Wno-stringop-overflow"
export CXXFLAGS="-O3 -Wno-stringop-overflow"
NUMPY_INCLUDE=$(python3 -c "import numpy; print(numpy.get_include())")

# Apply CMake fixes
sed -i '193i if(NOT TARGET Python::NumPy)\n    find_package(Python3 COMPONENTS NumPy REQUIRED)\n    add_library(Python::NumPy INTERFACE IMPORTED)\n    target_include_directories(Python::NumPy INTERFACE ${Python3_NumPy_INCLUDE_DIR})\nendif()' cmake/onnxruntime_python.cmake

# Build onnxruntime wheel
./build.sh \
  --cmake_extra_defines \
    "onnxruntime_PREFER_SYSTEM_LIB=ON" \
    "Protobuf_PROTOC_EXECUTABLE=${PROTOC}" \
    "Protobuf_INCLUDE_DIR=${LIBPROTO_INSTALL}/include" \
    "onnxruntime_USE_COREML=OFF" \
    "onnxruntime_BUILD_UNIT_TESTS=OFF"  \
    "onnxruntime_ENABLE_TESTS=OFF"  \
    "Python3_NumPy_INCLUDE_DIR=${NUMPY_INCLUDE}" \
    "CMAKE_POLICY_DEFAULT_CMP0001=NEW" \
    "CMAKE_POLICY_DEFAULT_CMP0002=NEW" \
    "CMAKE_POLICY_VERSION_MINIMUM=3.5" \
  --cmake_generator Ninja \
  --build_shared_lib \
  --config Release \
  --update \
  --build \
  --skip_submodule_sync \
  --allow_running_as_root \
  --compile_no_warning_as_error \
  --build_wheel


cp build/Linux/Release/dist/*.whl /onnxruntime_wheels/
