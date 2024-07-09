REPOSRC="hclone https://github.com/official-stockfish/Stockfish.git"
LOCALREPO="Stockfish"

if [ ! -d "$LOCALREPO/.git" ]; then
  git clone "$REPOSRC" "$LOCALREPO"
else
  git -C "$LOCALREPO" pull
fi

cd Stockfish/src
make build ARCH=x86-64