#!/bin/bash

set -e

# I like to use docopt...
hparams_name=$1
dst_root=$2
generator_warmup_epoch=$3
discriminator_warmup_epoch=$4
total_epoch=$5
experiment_id=$6

w_d=1

randstr=$(python -c "from datetime import datetime; print(str(datetime.now()).replace(' ', '_'))")
randstr=${experiment_id}_${randstr}

echo "Experiment id:" $experiment_id
echo "Name of hyper paramters:" $hparams_name
echo "Model checkpoints saved at:" $dst_root
echo "Experiment identifier:" $randstr
echo "Generator wamup epoch:" $generator_warmup_epoch
echo "Discriminator wamup epoch:" $discriminator_warmup_epoch
echo "Total epoch for GAN:" $total_epoch

max_files=-1 # -1 means `use full data`.

run_adversarial=1
run_generator_warmup=1
run_discriminator_warmup=1

# Generator warmup
# only train generator
if [ "${run_generator_warmup}" == 1 ]; then
    python train.py --hparams_name="$hparams_name" \
        --w_d=0 --hparams="nepoch=$generator_warmup_epoch" \
        --checkpoint-dir=$dst_root/gan_g_warmup \
        --log-event-path="log/${hparams_name}_generator_warmup_$randstr" \
fi

# Discriminator warmup
# only train discriminator
if [ "${run_discriminator_warmup}" == 1 ]; then
    python -W ignore train.py --hparams_name="$hparams_name" \
        --max_files=$max_files --w_d=${w_d} \
        --checkpoint-g=$dst_root/gan_g_warmup/checkpoint_epoch${generator_warmup_epoch}_Generator.pth\
        --discriminator-warmup --hparams="nepoch=$discriminator_warmup_epoch" \
        --checkpoint-dir=$dst_root/gan_d_warmup $inputs_dir $outputs_dir \
        --restart_epoch=0 \
        --log-event-path="log/${hparams_name}_discriminator_warmup_$randstr" \
fi

# Finally do joint training generator and discriminator
# start from ${generator_warmup_epoch}
if [ "${run_adversarial}" == 1 ]; then
    python -W ignore train.py --hparams_name="$hparams_name" \
        --max_files=$max_files \
        --w_d=${w_d} --hparams="nepoch=$total_epoch" \
        --checkpoint-g=$dst_root/gan_g_warmup/checkpoint_epoch${generator_warmup_epoch}_Generator.pth \
        --checkpoint-dir=$dst_root/gan \
        --reset_optimizers --restart_epoch=${generator_warmup_epoch} \
        $inputs_dir $outputs_dir \
        --log-event-path="log/${hparams_name}_adversarial_training_$randstr"
fi