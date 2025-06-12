# sh scripts/train_stylexd_onDocker.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/code/Jigsaw_matching
python _my/train_matching_stylexd.py --cfg experiments/finetune_matching_stylexd_Q124_onDocker.yaml