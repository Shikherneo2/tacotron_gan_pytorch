#!/bin/bash
# GAN-based TTS demo

set -e
experiment_id=$1

# Set what you want to run
checkpoints_dir=./checkpoints/${experiment_id}

echo "Experimental id:" $experiment_id

# train_gan.sh args:
# 1. Hyper param name
# 2. Where to save checkpoints
# 3. Generator warmup epoch
# 4. discriminator_warmup_epoch
# 5. Total epoch for GAN
# 6. Experiment name

# Train model
if [ "$run_acoustic_training" == 1 ]; then
    ./train_gan.sh tts_acoustic \
        ${checkpoints_dir}/tts_acoustic \
        15 0 300 $experiment_id
fi