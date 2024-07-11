#!/bin/bash

if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <repo_link> <local_repo_name>"
  exit 1
fi

repo_src=$1
local_repo=$2

if [ ! -d "$local_repo/.git" ]; then
  echo "Cloning repository..."
  git clone "$repo_src" "$local_repo"
else
  echo "Pulling latest changes..."
  git -C "$local_repo" pull
fi