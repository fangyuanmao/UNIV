CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node 8 --master_port 12345 pretrain_mcmae.py \
    --output_dir=./log/UnIV_ir_rgb --config ./configs/mcmae.yaml