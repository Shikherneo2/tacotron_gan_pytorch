# coding: utf-8

import numpy as np
import tensorflow as tf
from os.path import join, dirname


def hparams_debug_string(params):
    values = params.values()
    hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
    return 'Hyperparameters:\n' + '\n'.join(hp)

# Hyper paramters for TTS duration model
tts_duration = tf.contrib.training.HParams(
    name="duration",

    # Linguistic features
    use_phone_alignment=False,
    subphone_features=None,
    add_frame_features=False,
    question_path=join(dirname(__file__), "nnmnkwii_gallery", "data",
                       "questions-radio_dnn_416.hed"),

    # Duration features
    windows=[
        (0, 0, np.array([1.0])),
    ],
    stream_sizes=[5],
    has_dynamic_features=[False],

    recompute_delta_features=False,

    # Streams used for computing adversarial loss
    adversarial_streams=[True],
    mask_nth_mgc_for_adv_loss=0,

    # Generator
    generator="GRURNN",
    generator_add_noise=False,
    generator_noise_dim=200,
    generator_params={
        "in_dim": None,  # None wil be set automatically
        "out_dim": None,
        "num_hidden": 6,
        "hidden_dim": 512,
        "bidirectional": True,
        "dropout": 0.0,
        "use_relu": 0,
        "rnn_dropout": 0.2,
        "last_sigmoid": False,
    },
    optimizer_g="Adam",
    optimizer_g_params={
        "lr": 0.0001,
        "betas": (0.5, 0.9),
        "weight_decay": 0,
    },


    # Discriminator
    discriminator_linguistic_condition=True,
    discriminator="GRURNN",
    discriminator_params={
        "in_dim": None,  # None wil be set automatically
        "out_dim": 1,
        "num_hidden": 6,
        "hidden_dim": 512,
        "bidirectional": True,
        "dropout": 0.0,
        "use_relu": 0,
        "rnn_dropout": 0.2,
        "last_sigmoid": False,
    },
    optimizer_d="Adam",
    optimizer_d_params={
        "lr": 0.0001,
        "betas": (0.5, 0.9),
        "weight_decay": 0,
    },

    # This should be overrided
    nepoch=200,

    # LR schedule
    lr_decay_schedule=False,
    lr_decay_epoch=25,

    # Datasets and data loader
    batch_size=30,
    num_workers=1,
    pin_memory=True,
    cache_size=1200,
)

# Hyper paramters for TTS acoustic model
tts_acoustic = tf.contrib.training.HParams(
    name="acoustic",

    # Linguistic
    use_phone_alignment=False,
    subphone_features="full",
    add_frame_features=True,
    question_path=join(dirname(__file__), "nnmnkwii_gallery", "data",
                       "questions-radio_dnn_416.hed"),

    # Acoustic features
    order=59,
    frame_period=5,
    f0_floor=71.0,
    f0_ceil=700,
    use_harvest=True,  # If False, use dio and stonemask
    windows=[
        (0, 0, np.array([1.0])),
        (1, 1, np.array([-0.5, 0.0, 0.5])),
        (1, 1, np.array([1.0, -2.0, 1.0])),
    ],
    f0_interpolation_kind="quadratic",
    mod_spec_smoothing=True,
    mod_spec_smoothing_cutoff=50,  # Hz

    recompute_delta_features=False,

    # Stream info
    # (mgc, lf0, vuv, bap)
    stream_sizes=[180, 3, 1, 3],
    has_dynamic_features=[True, True, False, True],

    # Streams used for computing adversarial loss
    # NOTE: you should probably change discriminator's `in_dim`
    # if you change the adv_streams
    adversarial_streams=[True, False, False, False],
    # Don't set the value > 0 unless you are sure what you are doing
    # mask 0 to n-th mgc for adversarial loss
    # e.g, for n=2, 0-th and 1-th mgc coefficients will be masked
    mask_nth_mgc_for_adv_loss=2,

    # Generator
    generator_add_noise=False,
    generator_noise_dim=200,
    generator="GRURNN",
    generator_params={
        "in_dim": None,  # None wil be set automatically
        "out_dim": None,
        "num_hidden": 6,
        "hidden_dim": 512,
        "bidirectional": True,
        "dropout": 0.2,
        "use_relu": 0,
        "rnn_dropout": 0.2,
        "last_sigmoid": False,
    },
    optimizer_g="Adam",
    optimizer_g_params={
        "lr": 0.0001,
        "betas": (0.5, 0.9),
        "weight_decay": 1e-7,
    },

    # Discriminator
    discriminator_linguistic_condition=True,
    discriminator="GRURNN",
    discriminator_params={
        "in_dim": None,  # None wil be set automatically
        "out_dim": 1,
        "num_hidden":6,
        "hidden_dim": 512,
        "bidirectional": True,
        "dropout": 0.4,
        "use_relu": 0,
        "rnn_dropout": 0.4,
        "last_sigmoid": False,
    },
    optimizer_d="Adam",
    optimizer_d_params={
        "lr": 0.0001,
        "betas": (0.5, 0.9),
        "weight_decay": 1e-7,
    },

    # This should be overrided
    nepoch=200,

    # LR schedule
    lr_decay_schedule=False,
    lr_decay_epoch=25,

    # Datasets and data loader
    batch_size=15,
    num_workers=1,
    pin_memory=True,
    cache_size=1200,
)
