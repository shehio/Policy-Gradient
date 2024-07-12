#!/bin/bash
bash scripts/clone_if_not_found.sh "git@github.com:LeelaChessZero/lc0.git" "lc0"
bash scripts/install_homebrew.sh

brew install meson ninja python3 zlib gcc
cd lc0 && mkdir build && cd build
meson setup --buildtype=release ..
ninja

# reference: https://lczero.org/dev/wiki/best-nets-for-lc0/
wget  -nc --tries=10 -P lc0/build/ https://storage.lczero.org/files/networks-contrib/t1-512x15x8h-distilled-swa-3395000.pb.gz