EXP=synface
CONFIG=synface
GPUS=2
PORT=${PORT:-29571}

mkdir -p results/${EXP}
CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    run.py \
    --launcher pytorch \
    --config configs/${CONFIG}.yml \
    2>&1 | tee results/${EXP}/log.txt
