curl -OL https://huggingface.co/cleanrl/BreakoutNoFrameskip-v4-dqn_atari-seed1/raw/main/pyproject.toml
curl -OL https://huggingface.co/cleanrl/BreakoutNoFrameskip-v4-dqn_atari-seed1/raw/main/poetry.lock
poetry install --all-extras

curl -OL https://huggingface.co/cleanrl/BreakoutNoFrameskip-v4-dqn_atari-seed1/raw/main/dqn.py