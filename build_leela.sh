#!/bin/bash
repo_src="git@github.com:LeelaChessZero/lc0.git"
local_repo="lc0"

if [ ! -d "$local_repo/.git" ]; then
  git clone "$repo_src" "$local_repo"
else
  git -C "$local_repo" pull
fi

bash scripts/install_homebrew.sh

brew install meson ninja python3 zlib gcc
cd lc0 && mkdir build && cd build
meson setup --buildtype=release ..
ninja