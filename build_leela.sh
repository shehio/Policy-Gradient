#!/bin/bash
repo_src="git@github.com:LeelaChessZero/lc0.git"
local_repo="lc0"

if [ ! -d "$local_repo/.git" ]; then
  git clone "$repo_src" "$local_repo"
else
  git -C "$local_repo" pull
fi

/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"

brew install meson ninja python3 zlib gcc
cd lc0
mkdir build && cd build
meson setup --buildtype=release ..
ninja