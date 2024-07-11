#!/bin/bash
bash scripts/clone_if_not_found.sh "git@github.com:LeelaChessZero/lc0.git" "lc0"
bash scripts/install_homebrew.sh

brew install meson ninja python3 zlib gcc
cd lc0 && mkdir build && cd build
meson setup --buildtype=release ..
ninja