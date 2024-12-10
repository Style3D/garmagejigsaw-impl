# sh scripts/train_stylexd_onDocker.sh
export PYTHONUNBUFFERED=1
export PYTHONPATH=/home/code/Jigsaw_matching
python _my/train_matching_stylexd.py --cfg experiments/train_matching_stylexd_part_combination_onDocker.yaml