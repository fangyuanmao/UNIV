./tools/dist_train.sh \
    configs/convmae/upernet_msrs.py 4 \
    --seed 24 --deterministic \
    --work-dir ./log/results \
    --options model.pretrained=/path/to/checkpoint \
