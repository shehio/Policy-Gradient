#!/bin/bash
repo_src="git@github.com:LeelaChessZero/lc0.git"
local_repo="lc0"

if [ ! -d "$local_repo/.git" ]; then
  git clone "$repo_src" "$local_repo"
else
  git -C "$local_repo" pull
fi

pretty_print() {
  printf "\n%b\n" "$1"
}

if ! command -v brew &> /dev/null; then
  pretty_print "Homebrew is not installed. Installing Homebrew..."
  
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  
  # Add Homebrew to the PATH
  echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zprofile
  eval "$(/opt/homebrew/bin/brew shellenv)"
  
  pretty_print "Homebrew installation completed."
else
  pretty_print "Homebrew is already installed."
fi

brew install meson ninja python3 zlib gcc
cd lc0 && mkdir build && cd build
meson setup --buildtype=release ..
ninja