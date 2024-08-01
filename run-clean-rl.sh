brew install poetry
poetry install

pip3.10 install tensorboard
tensorboard --logdir runs &

poetry shell
poetry install -E atari
python cleanrl/dqn_atari.py --env-id BreakoutNoFrameskip-v4