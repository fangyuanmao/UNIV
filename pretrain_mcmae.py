# Copyright notice: This code is copyrighted by Facebook, Inc. and its affiliates.
# It's licensed under the Apache License 2.0, with rules about usage, distribution,
# and warranty limitations.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import models as torchvision_models
import yaml
from peft import get_peft_model, LoraConfig
from copy import deepcopy
import shutil
import warnings
warnings.filterwarnings('ignore')

from datasets.co_data_augmentation import CoDataAugmentation
from datasets.datasets import RGB_pair_IR_dataset
import models.backbone.mcmae.models_convmae as models_convmae
from loss.cross_modality_loss import attention_simi_guided_loss
from loss.RGB_distillation_loss import RGB_patch_simi_loss
import utils.utils as utils
from utils.IR_info_richness import gray_value_rank

# Get names of all callable, lowercase, non-private torchvision model architectures
torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))


class DictToObject:
    def __init__(self, dictionary):
        for key, value in dictionary.items():
            if isinstance(value, dict):
                setattr(self, key, DictToObject(value))
            else:
                setattr(self, key, value)

# Function to load and parse a YAML config file
def get_config(config_file):
    with open(config_file, 'r') as config_f:
        return DictToObject(yaml.load(config_f, Loader=yaml.FullLoader))


# Create an argument parser for MCMAE-related command-line arguments
def get_args_parser():
    parser = argparse.ArgumentParser('MCMAE', add_help=False)
    parser.add_argument('--config', default='./configs/mcmae.yaml')
    # I/O related arguments
    parser.add_argument('--output_dir', default="/path/to/your/output_dir", type=str,
                        help='Path to save logs and checkpoints.')

    # Training and optimization arguments
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True,
                        help="""Whether to use half precision for training. It can speed up
                        training and save memory, but may cause instability or slight performance drop.
                        Disable it if the loss is unstable, patch size is reduced, or training large ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3,
                        help="""Max gradient norm for gradient clipping. Clipping between 0.3 - 1.0 can help
                        optimize larger ViT architectures. Set to 0 to disable.""")
    parser.add_argument('--freeze_last_layer', default=1, type=int,
                        help="""Number of epochs to keep the output layer fixed. Often, freezing it in the first
                        epoch helps training. Increase if the loss doesn't decrease.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
                        choices=['adamw', 'sgd', 'lars'],
                        help="""Type of optimizer. adamw is recommended for ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1,
                        help="Stochastic depth rate")

    # Distributed training arguments
    parser.add_argument("--dist_url", default="env://", type=str,
                        help="""URL for setting up distributed training""")
    parser.add_argument("--local-rank", default=0, type=int,
                        help="Don't set this argument, ignore it.")
    parser.add_argument("--dist_backend", default='nccl')

    return parser


# Main training function
def train(args, config):
    # Initialize distributed training mode
    utils.init_distributed_mode(args)
    # Fix random seeds for reproducibility
    utils.fix_random_seeds(config.training.seed)
    # Print all command-line arguments
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    # Enable cudnn benchmark for faster convolution
    cudnn.benchmark = True

    # ============ preparing data... ============
    # Create data transformation for RGB-IR pairs
    Cotransform = CoDataAugmentation(mean_RGB=config.dataset.mean_RGB,
                                     std_RGB=config.dataset.std_RGB,
                                     mean_IR=config.dataset.mean_IR,
                                     std_IR=config.dataset.std_IR)
    # Create the RGB-IR image pair dataset
    dataset = RGB_pair_IR_dataset(transform=Cotransform, data_path=config.dataset.path)
    # Create a distributed sampler for the dataset
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    # Create a data loader for the dataset
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=config.training.batch_size_per_gpu,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} image pairs.")

    # ============ building student and teacher networks... ============
    # Initialize student and teacher networks
    student = models_convmae.__dict__['convmae_convvit_base_patch16']()
    teacher = models_convmae.__dict__['convmae_convvit_base_patch16']()

    # Move networks to GPU
    student, teacher = student.to(args.local_rank), teacher.to(args.local_rank)

    # Synchronize batch norms if present
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)
        # Use DDP wrapper for synchronized batch norms in teacher
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.local_rank], find_unused_parameters=True)
        teacher_without_ddp = teacher.module
    else:
        teacher_without_ddp = teacher

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.local_rank], find_unused_parameters=True)
    # Load pre-trained weights for student and teacher
    student.module.load_state_dict(torch.load(config.model.pretrain_model_path, map_location="cpu")['model'], strict=False)
    try:
        teacher.module.load_state_dict(torch.load(config.model.pretrain_model_path, map_location="cpu")['model'], strict=False)
    except:
        teacher.load_state_dict(torch.load(config.model.pretrain_model_path, map_location="cpu")['model'], strict=False)

    # Don't compute gradients for teacher since no backprop
    for p in teacher_without_ddp.parameters():
        p.requires_grad = False
    if config.model.use_lora:
        for p in student.module.parameters():
            p.requires_grad = False
        # Configure LoRA settings
        lora_config = LoraConfig(target_modules=config.model.lora_config.target_modules,
                            inference_mode=False,
                            r=config.model.lora_config.lora_low_rank,
                            lora_alpha=config.model.lora_config.lora_alpha,
                            lora_dropout=config.model.lora_config.lora_dropout
                            )
        student.module = get_peft_model(student.module, lora_config)

    print(f"teacher backbone loaded, student initialized ---------------")

    # ============ preparing loss... ============
    if config.loss.cross_modality_loss == 'attention_simi_guided_loss':
        cross_modality_loss_fn = attention_simi_guided_loss(threshold=config.loss.atten_map_threshold,
                                                            temperature=config.loss.temperature
                                                            ).to(args.local_rank)
    if config.loss.rgb_distillation_loss == 'RGB_patch_simi_loss':
        rgb_loss_fn = RGB_patch_simi_loss().to(args.local_rank)
    elif config.loss.rgb_distillation_loss == 'attention_simi_guided_loss':
        rgb_loss_fn = attention_simi_guided_loss(threshold=config.loss.atten_map_threshold,
                                                 temperature=config.loss.temperature
                                                 ).to(args.local_rank)
    else:
        rgb_loss_fn = nn.MSELoss(reduction='mean').to(args.local_rank)

    # ============ preparing optimizer... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # Suitable for ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # LR set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # For convnets and large batches

    # For mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers... ============
    # Initialize learning rate scheduler
    lr_schedule = utils.cosine_scheduler(
        config.training.lr * (config.training.batch_size_per_gpu * utils.get_world_size()) / 256.,
        config.training.min_lr,
        config.training.epochs, len(data_loader),
        warmup_epochs=config.training.warmup_epochs,
    )
    # Initialize weight decay scheduler
    wd_schedule = utils.cosine_scheduler(
        config.training.weight_decay,
        config.training.weight_decay_end,
        config.training.epochs, len(data_loader),
    )

    print(f"Loss, optimizer and schedulers ready.")
    start_time = time.time()
    print("Starting UNIV training!")

    for epoch in range(0, config.training.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of MCMAE... ============
        train_stats = train_one_epoch(student, teacher, cross_modality_loss_fn, rgb_loss_fn,
                                      data_loader, optimizer, lr_schedule, wd_schedule,
                                      epoch, fp16_scaler, args, config)

        # ============ writing logs... ============
        if config.model.use_lora:
            student_merged = deepcopy(student.module)
            student_save = student_merged.merge_and_unload()
        else:
            student_save = student.module
        save_dict = {
            'student': student_save.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'cross_modality_loss_fn': cross_modality_loss_fn.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if config.training.saveckp_freq and epoch % config.training.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


# Function to train for one epoch
def train_one_epoch(student, teacher, cross_modality_loss_fn, rgb_loss_fn,
                    data_loader, optimizer, lr_schedule, wd_schedule, epoch,
                    fp16_scaler, args, config):
    # Create a metric logger
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, config.training.epochs)
    for it, (rgb_images, ir_images) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # Calculate global training iteration
        it = len(data_loader) * epoch + it
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # Only the first group has weight decay
                param_group["weight_decay"] = wd_schedule[it]

        # Move images to GPU
        rgb_images = rgb_images.cuda(non_blocking=True)
        ir_images = ir_images.cuda(non_blocking=True)
        images = torch.cat([rgb_images, ir_images], dim=0)
        if config.loss.ir_info:
            ir_info = gray_value_rank(ir_images[:, 0, :, :])
        else:
            ir_info = None

        # Forward pass for teacher and student, compute loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            rgb_teacher_embed, teacher_attention_map = teacher(rgb_images, mask_ratio=0, return_last_attention=True)
            rgb_ir_patch_embed, _ = student(images, mask_ratio=0, return_last_attention=True)

            rgb_patch_embed = rgb_ir_patch_embed[0:config.training.batch_size_per_gpu]
            ir_patch_embed = rgb_ir_patch_embed[config.training.batch_size_per_gpu:]

            irloss = config.loss.ir_alpha * cross_modality_loss_fn(rgb_teacher_embed, ir_patch_embed, teacher_attention_map, ir_info)
            rgbloss = config.loss.rgb_beta * rgb_loss_fn(rgb_teacher_embed, rgb_patch_embed, teacher_attention_map, ir_info)

            loss =  irloss + rgbloss

        # Stop training if loss is infinite
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training---------------".format(loss.item()), force=True)
            print(f'rgb loss is : {rgbloss}')
            print(f'ir loss : {irloss}')
            sys.exit(1)

        # Update student network
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # Logging
        torch.cuda.synchronize()
        metric_logger.update(irloss=irloss.item())
        metric_logger.update(rgbloss=rgbloss.item())
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # Synchronize metrics across processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('MCMAE', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    config = get_config(args.config)
    shutil.copyfile(args.config, Path(args.output_dir) / "config.yaml")
    train(args, config)