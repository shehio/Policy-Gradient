#!/bin/bash
bash common-scripts/clone_if_not_found.sh "git@github.com:official-stockfish/Stockfish.git" "Stockfish"
cd Stockfish/src

file="misc.cpp"
old_string='constexpr std::string_view version = "dev";'
new_string='constexpr std::string_view version = "16.1";'
sed -i '' "s|$old_string|$new_string|g" "$file"

make clean
make build ARCH=apple-silicon