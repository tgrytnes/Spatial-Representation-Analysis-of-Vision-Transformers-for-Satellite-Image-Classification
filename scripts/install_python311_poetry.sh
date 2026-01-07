#!/usr/bin/env bash
set -euo pipefail

PYTHON_VERSION="${PYTHON_VERSION:-3.11.9}"
PYTHON_PREFIX="${PYTHON_PREFIX:-/usr/local}"
POETRY_VERSION="${POETRY_VERSION:-1.8.3}"

if [[ "$(id -u)" -ne 0 ]]; then
  echo "Run as root (or via sudo) so Python can be installed to ${PYTHON_PREFIX}."
  exit 1
fi

echo "Installing build dependencies..."
apt-get update
apt-get install -y \
  build-essential \
  libssl-dev \
  zlib1g-dev \
  libncurses5-dev \
  libncursesw5-dev \
  libreadline-dev \
  libsqlite3-dev \
  libgdbm-dev \
  libdb5.3-dev \
  libbz2-dev \
  libexpat1-dev \
  liblzma-dev \
  tk-dev \
  libffi-dev \
  uuid-dev \
  wget \
  curl

PYTHON_BIN="${PYTHON_PREFIX}/bin/python3.11"
if [[ -x "${PYTHON_BIN}" ]]; then
  echo "Python 3.11 already installed at ${PYTHON_BIN}."
else
  echo "Downloading Python ${PYTHON_VERSION}..."
  cd /usr/src
  if [[ ! -f "Python-${PYTHON_VERSION}.tgz" ]]; then
    wget -q "https://www.python.org/ftp/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz"
  fi

  if [[ ! -d "Python-${PYTHON_VERSION}" ]]; then
    tar -xzf "Python-${PYTHON_VERSION}.tgz"
  fi

  echo "Building and installing Python ${PYTHON_VERSION}..."
  cd "/usr/src/Python-${PYTHON_VERSION}"
  ./configure --with-ensurepip=install
  make -j"$(nproc)"
  make altinstall
fi

echo "Installing Poetry ${POETRY_VERSION}..."
export POETRY_VERSION
curl -sSL https://install.python-poetry.org | "${PYTHON_BIN}" -

echo "Poetry installed at ${HOME}/.local/bin/poetry"
echo "Make sure ${HOME}/.local/bin is on your PATH."
