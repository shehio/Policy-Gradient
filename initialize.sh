#!/bin/bash
repo_src="hclone https://github.com/official-stockfish/Stockfish.git"
local_repo="Stockfish"

if [ ! -d "$local_repo/.git" ]; then
  git clone "$repo_src" "$local_repo"
else
  git -C "$local_repo" pull
fi

cd Stockfish/src

file="misc.cpp"
old_string='constexpr std::string_view version = "dev";'
new_string='constexpr std::string_view version = "16.1";'
sed -i '' "s|$old_string|$new_string|g" "$file"

make clean
make build ARCH=apple-silicon