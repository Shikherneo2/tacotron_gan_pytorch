# coding: utf-8

"""Trainining script for Tacotron speech synthesis model.

usage: train.py [options]

options:
    --hparams_name=<name>       Name of hyper params [default: vc].
    --hparams=<parmas>          Hyper parameters to be overrided [default: ].
    --checkpoint-g=<name>       Load generator from checkpoint if given.
    --checkpoint-d=<name>       Load discriminator from checkpoint if given.
    --checkpoint-r=<name>       Load reference model to compute spoofing rate.
    --max_files=<N>             Max num files to be collected. [default: -1]
    --discriminator-warmup      Warmup discriminator.
    --w_d=<f>                   Adversarial loss weight [default: 1.0].
    --mse_w=<f>                 Mean squared error (MSE) loss weight [default: 0.0].
    --restart_epoch=<N>         Restart epoch [default: -1].
    --reset_optimizers          Reset optimizers, otherwise restored from checkpoint.
    --log-event-path=<name>     Log event path.

    --checkpoint-dir=<dir>    Directory where to save model checkpoints [default: checkpoints].
    --checkpoint-path=<name>  Restore model from checkpoint path if given.
    -h, --help                Show this help message and exit
"""

import torch
import pickle
import numpy as np
from docopt import docopt
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
import torch.backends.cudnn as cudnn
from torch.utils import data as data_utils
from torch.autograd import grad as torch_grad
from sklearn.model_selection import train_test_split

import os
import sys
import math
import time
from tqdm import tqdm
from warnings import warn
from os.path import splitext, join, abspath, exists

# Make the tacotron files accessible
tacotron_lib_dir = join( expanduser("~"), "tacotron", "lib", "tacotron" )
sys.path.append( tacotron_lib_dir )
from tacotron_pytorch import Tacotron
from util import audio
from util.plot import plot_alignment

import tensorboard_logger
from tensorboard_logger import log_value

import gantts_models
from nnmnkwii import preprocessing as P
from nnmnkwii.datasets import FileSourceDataset, FileDataSource

from hparams import hparams
from text import text_to_sequence, symbols
from hparams_gan import hparams_debug_string
from hparams_gan import hparams as hparams_gan

hp = None  # to be initailized later
gp_weight = 10

global_epoch = 0
global_step_gen = 0
checkpoint_interval = 10
use_cuda = torch.cuda.is_available()

DATA_ROOT = join(expanduser("~"), "tacotron", "training_ljspeech_gantts")

def gradient_penalty(model_d, real_data, generated_data, lengths):
    batch_size = real_data.size()[0]

    # Calculate interpolation
    alpha = torch.rand(batch_size, 1, 1)
    alpha = alpha.expand_as(real_data)
    if use_cuda:
        alpha = alpha.cuda()
    interpolated = alpha * real_data.data + (1 - alpha) * generated_data.data
    interpolated = Variable(interpolated, requires_grad=True)
    if use_cuda:
        interpolated = interpolated.cuda()

    # Calculate probability of interpolated examples
    prob_interpolated = model_d(interpolated, lengths=lengths)

    # Calculate gradients of probabilities with respect to examples
    gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                           grad_outputs=torch.ones(prob_interpolated.size()).cuda() if use_cuda else torch.ones(
                           prob_interpolated.size()),
                           create_graph=True, retain_graph=True)[0]

    # Flatten Gradiengts to easily take norm per example in batch
    gradients = gradients.contiguous()
    gradients = gradients.view(batch_size, -1)

    # Derivatives of the gradient close to 0 can cause problems because of
    # the square root, so manually calculate norm and add epsilon
    gradients_norm = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-12)

    # Return gradient penalty
    return gp_weight * ((gradients_norm - 1) ** 2).mean()

# Pad all inputs with zeros to make them the same length
def _pad_2d(x, max_len):
    print(np.array(x).shape)
    print(max_len)
    x = np.pad(x, [(0, max_len - len(x)), (0, 0)],
               mode="constant", constant_values=0)
    return x


class TextDataSource(FileDataSource):
    def __init__(self):
        self._cleaner_names = [x.strip() fortqdm x in hparams.cleaners.split(',')]

    def collect_files(self):
        meta = join(DATA_ROOT, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        lines = list(map(lambda l: l.decode("utf-8").split("|")[-1], lines))
        return lines

    def collect_features(self, text):
        return np.asarray(text_to_sequence(text, self._cleaner_names),
                          dtype=np.int32)


class _NPYDataSource(FileDataSource):
    def __init__(self, col):
        self.col = col

    def collect_files(self):
        meta = join(DATA_ROOT, "train.txt")
        with open(meta, "rb") as f:
            lines = f.readlines()
        lines = list(map(lambda l: l.decode("utf-8").split("|")[self.col], lines))
        paths = list(map(lambda f: join(DATA_ROOT, f), lines))
        return paths

    def collect_features(self, path):
        return np.load(path)


class MelSpecDataSource(_NPYDataSource):
    def __init__(self):
        super(MelSpecDataSource, self).__init__(1)


class LinearSpecDataSource(_NPYDataSource):
    def __init__(self):
        super(LinearSpecDataSource, self).__init__(0)


class PyTorchDataset(object):
    def __init__(self, X, Mel, Y):
        self.X = X
        self.Mel = Mel
        self.Y = Y

    def __getitem__(self, idx):
        return self.X[idx], self.Mel[idx], self.Y[idx]

    def __len__(self):
        return len(self.X)


def collate_fn(batch):
    """Create batch"""
    r = hparams.outputs_per_step
    input_lengths = [len(x[0]) for x in batch]
    max_input_len = np.max(input_lengths)
    # Add single zeros frame at least, so plus 1
    max_target_len = np.max([len(x[1]) for x in batch]) + 1
    if max_target_len % r != 0:
        max_target_len += r - max_target_len % r
        assert max_target_len % r == 0

    a = np.array([_pad(x[0], max_input_len) for x in batch], dtype=np.int)
    x_batch = torch.LongTensor(a)

    input_lengths = torch.LongTensor(input_lengths)

    b = np.array([_pad_2d(x[1], max_target_len) for x in batch],
                 dtype=np.float32)
    mel_batch = torch.FloatTensor(b)

    c = np.array([_pad_2d(x[2], max_target_len) for x in batch],
                 dtype=np.float32)
    y_batch = torch.FloatTensor(c)
    return x_batch, input_lengths, mel_batch, y_batch

def save_alignment(path, attn):
    plot_alignment(attn.T, path, info="tacotron, step={}".format(global_step))

def save_spectrogram(path, linear_output):
    spectrogram = audio._denormalize(linear_output)
    plt.figure(figsize=(16, 10))
    plt.imshow(spectrogram.T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(path, format="png")
    plt.close()

def save_states(global_step, linear_outputs, attn, y,
                input_lengths, checkpoint_dir=None):
    print("Save intermediate states at step {}".format(global_step))

    idx = min(1, len(input_lengths) - 1)
    input_length = input_lengths[idx]

    # Alignment
    path = join(checkpoint_dir, "step{}_alignment.png".format(
        global_step))

    alignment = attn[idx].cpu().data.numpy()
    save_alignment(path, alignment)

    # Predicted spectrogram
    path = join(checkpoint_dir, "step{}_predicted_spectrogram.png".format(
        global_step))
    linear_output = linear_outputs[idx].cpu().data.numpy()
    save_spectrogram(path, linear_output)

    # Predicted audio signal
    signal = audio.inv_spectrogram(linear_output.T)
    path = join(checkpoint_dir, "step{}_predicted.wav".format(
        global_step))
    audio.save_wav(signal, path)

    # Target spectrogram
    path = join(checkpoint_dir, "step{}_target_spectrogram.png".format(
        global_step))
    linear_output = y[idx].cpu().data.numpy()
    save_spectrogram(path, linear_output)

def save_checkpoint(model, optimizer, epoch, step, checkpoint_dir, name):
    checkpoint_path = join(
        checkpoint_dir, "checkpoint_epoch{}_{}.pth".format(
            epoch, name))
    torch.save({
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "global_step": step,
        "global_epoch": epoch,
    }, checkpoint_path)
    print("Saved checkpoint:", checkpoint_path)


def update_discriminator(model_d, optimizer_d, x, y_static, y_hat, lengths,
                         mask, phase, eps=1e-20):
    
    y_static_adv, y_hat_static_adv = y_static, y_hat

    if hp.discriminator_linguistic_condition:
        y_static_adv = torch.cat((x, y_static_adv), -1)
        y_hat_static_adv = torch.cat((x, y_hat_static_adv), -1)

    # Real
    D_real = model_d(y_static_adv, lengths=lengths)
    real_correct_count = (D_real > 0.5).float().item()

    # Fake
    D_fake = model_d(y_hat_static_adv, lengths=lengths)
    fake_correct_count = (D_fake < 0.5).float().item()

    # Wasserstein Loss
    # Do we take an average
    loss_real_d = - (D_real + eps)
    loss_fake_d = - (D_fake + eps)
    loss_d = loss_real_d - loss_fake_d 

    loss_d = loss_d + gradient_penalty(model_d, y_static_adv, y_hat_static_adv, lengths)
    loss_d.backward(retain_graph=True)
    torch.nn.utils.clip_grad_norm(model_d.parameters(), 1.0)
    optimizer_d.step()

    return loss_d.item(), loss_fake_d.item(), loss_real_d.item(), real_correct_count, fake_correct_count


def update_generator(model_g, model_d, optimizer_g,
                     x, y, mel, mel_hat, mel_static, linear_output,
                     adv_w, lengths, mask, phase,
                     mse_w=None, mge_w=None, eps=1e-20):

    criterion = nn.L1Loss(size_average=False)

    # Adversarial loss
    if adv_w > 0:
        # Select streams
        mel_hat_static_adv = mel_hat

        if hp.discriminator_linguistic_condition:
            mel_hat_static_adv = torch.cat((x, mel_hat_static_adv), -1)
        model_local_output = model_d( mel_hat_static_adv, lengths=lengths )

        # No log in Wasserstein GAN
        loss_adv = model_local_output + eps
    else:
        loss_adv = Variable(mel.data.new(1).zero_())

    mel_loss = criterion( mel_hat, mel )
    n_priority_freq = int(3000 / (fs * 0.5) * linear_dim)
    linear_loss = 0.5 * criterion(y, linear_output)  + 0.5 * criterion(y[:, :, :n_priority_freq], linear_output[:, :, :n_priority_freq])
    
    # L-1 + ADV loss
    gen_loss = linear_loss + mel_loss
    loss_g =  gen_loss + (adv_w * loss_adv)
    loss_g.backward()
    torch.nn.utils.clip_grad_norm(model_g.parameters(), 1.0)
    optimizer_g.step()

    return gen_loss.item(), loss_adv.item(), loss_g.item()


def _learning_rate_decay(init_lr, global_step):
    warmup_steps = 4000.0
    step = global_step + 1.
    lr = init_lr * warmup_steps**0.5 * np.minimum(
        step * warmup_steps**-1.5, step**-0.5)
    return lr


def exp_lr_scheduler(optimizer, epoch, nepoch, init_lr=0.0001, lr_decay_epoch=100):
    """Decay learning rate by a factor of 0.1 every lr_decay_epoch epochs."""
    lr = init_lr * (0.1**(epoch // lr_decay_epoch))

    if epoch % lr_decay_epoch == 0:
        print('LR is set to {} at epoch {}'.format(lr, epoch))

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return optimizer


def apply_generator(model_g, x, R, mel, lengths):
    if model_g.include_parameter_generation():
        # Case: models include parameter generation in itself
        # Mulistream features cannot be used in this case
        mel_hat, mel_hat_static = model_g(x, mel, R, lengths=lengths)
    else:
        # Case: generic models (can be sequence model)
        assert hp.has_dynamic_features is not None
        mel_hat, linear_op, attn = model_g( x, mel, lengths=lengths )

        # Handle dimention mismatch
        # This happens when we use pad_packed_sequence
        if mel_hat.size(1) != x.size(1):
            mel_hat = F.pad(mel_hat.unsqueeze(
                0), (0, 0, x.size(1) - mel_hat.size(-2), 0)).squeeze(0)

    return mel_hat, linear_op, attn


def train_loop( models, optimizers, dataset_loaders, w_d=0.0, mse_w=0.0, mge_w=1.0, update_d=True, update_g=True ):
    
    # Intial learning rate
    init_lr = 0.002
    model_g, model_d = models
    optimizer_g, optimizer_d = optimizers
    if use_cuda:
        model_g, model_d = model_g.cuda(), model_d.cuda()

    Y_data_mean = dataset_loaders["train"].dataset.Y_data_mean
    Y_data_std = dataset_loaders["train"].dataset.Y_data_std
    Y_data_mean = torch.from_numpy(Y_data_mean)
    Y_data_std = torch.from_numpy(Y_data_std)
    if use_cuda:
        Y_data_mean = Y_data_mean.cuda()
        Y_data_std = Y_data_std.cuda()

    E_loss_mge = 1
    E_loss_adv = 1
    has_dynamic = np.any(hp.has_dynamic_features)
    linear_outputs = []
    attn = []
    global global_epoch, global_step_gen
    
    print("Running for "+str(hp.nepoch)+" epochs")
    for global_epoch in range(global_epoch + 1, hp.nepoch + 1):
        print("Epoch : "+str(global_epoch))

        if hp.lr_decay_schedule and update_d:
            optimizer_d = exp_lr_scheduler(optimizer_d, global_epoch - 1, hp.nepoch,
                                           init_lr=hp.optimizer_d_params["lr"],
                                           lr_decay_epoch=hp.lr_decay_epoch)

        running_loss = {"generator": 0.0, "mse": 0.0,
                        "loss_real_d": 0.0,
                        "loss_fake_d": 0.0,
                        "loss_adv": 0.0,
                        "discriminator": 0.0}
        
        model_g.train()
        running_metrics = {}
        real_correct_count, fake_correct_count = 0, 0
        regard_fake_as_natural = 0
        N = len(dataset_loaders)
        total_num_frames = 0
        
        # update_g_local implements More Disc iters per Gen iter.
        if(update_g and update_d):
            update_g_local = False
        else:
            update_g_local = True

        for ind, (x, input_lengths, mel, y) in tqdm(enumerate(dataset_loaders)):

            if ( update_g_local==False and ind!=0 and ind%5 ==0):
                # We will be updating the gebnerator in this iteration. So update the learning rate according to the current number of generator step.
                update_g_local=True
                
                current_lr = _learning_rate_decay(init_lr, global_step_gen)
                for param_group in optimizer_g.param_groups:
                    param_group['lr'] = current_lr

            # Sort by lengths. This is needed for pytorch's PackedSequence
            sorted_lengths, indices = torch.sort(
                lengths.view(-1), dim=0, descending=True)
            sorted_lengths = sorted_lengths.long()
            cpu_sorted_lengths = list(sorted_lengths)
            max_len = sorted_lengths[0]

            # Get sorted batch
            x, mel, y = x[indices], mel[indices], y[indices]

            # Generator noise
            if hp.generator_add_noise:
                z = torch.rand(x.size(0), max_len, hp.generator_noise_dim)
            else:
                z = None

            # Construct MLPG paramgen matrix for every batch
            if has_dynamic:
                R = unit_variance_mlpg_matrix(hp.windows, max_len)
                R = torch.from_numpy(R)
                R = R.cuda() if use_cuda else R
            else:
                R = None

            if use_cuda:
                x, y, mel = x.cuda(), y.cuda(), mel.cuda()
                sorted_lengths = sorted_lengths.cuda()
                z = z.cuda() if z is not None else None

            # Pack into variables
            x, y, mel = Variable(x), Variable(y), Variable(mel)
            z = Variable(z) if z is not None else None
            sorted_lengths = Variable(sorted_lengths)

            # Static features
            mel_static = get_static_features(
                mel, len(hp.windows), hp.stream_sizes, hp.has_dynamic_features)

            # Num frames in batch
            total_num_frames += sorted_lengths.float().sum().item()

            # Mask -- NEEDED?
        (sorted_lengths).unsqueeze(-1)
            mask = None

            # Reset optimizers state
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            # Apply model (generator)
            generator_input = torch.cat((x, z), -1) if z is not None else x
            mel_hat, linear_outputs, attn = apply_generator( model_g, generator_input, R, mel, cpu_sorted_lengths )

            ### Update discriminator ###
            # Natural: 1, Generated: 0
            if update_d:
                loss_d, loss_fake_d, loss_real_d, _real_correct_count,\
                    _fake_correct_count = update_discriminator(
                        model_d, optimizer_d, x, mel_static, mel_hat
                        cpu_sorted_lengths, mask, phase)
                running_loss["discriminator"] += loss_d
                running_loss["loss_fake_d"] += loss_fake_d
                running_loss["loss_real_d"] += loss_real_d
                real_correct_count += _real_correct_count
                fake_correct_count += _fake_correct_count

            ### Update generator ###
            if (update_g==True and update_d==False) or (update_g==True and update_d==True and update_g_local==True):
                if (E_loss_adv == 0):
                    adv_w = w_d * 1e+3
                else:    
                    adv_w = w_d * float(np.clip(E_loss_mge / E_loss_adv, 0, 1e+3))

                loss_tacotron, loss_adv, loss_g = update_generator(
                    model_g, model_d, optimizer_g, x, y, mel, mel_hat,
                    mel_static, mel_hat_static,linear_output,
                    adv_w, cpu_sorted_lengths, mask, phase,
                    mse_w=mse_w, mge_w=mge_w)

                running_loss["generator"] += loss_g
                running_loss["loss_adv"] += loss_adv
                running_loss["tacotron"] += loss_tacotron

                global_step_gen += 1
                update_g_local = False
                save_states( global_step, linear_outputs, attn, y, sorted_lengths, checkpoint_dir )

            # Log loss
            for ty, enabled in [("tacotron", update_g),
                                ("discriminator", update_d),
                                ("loss_real_d", update_d),
                                ("loss_fake_d", update_d),
                                ("loss_adv", update_g and update_d),
                                ("generator", update_g)]:
                if enabled:
                    ave_loss = running_loss[ty] / N
                    log_value(
                        "{} {} loss".format(phase, ty), ave_loss, global_epoch)

            # Log discriminator classification accuracy
            if update_d:
                log_value("Real {} acc".format(phase),
                          real_correct_count / total_num_frames, global_epoch)
                log_value("Fake {} acc".format(phase),
                          fake_correct_count / total_num_frames, global_epoch)

        # Save checkpoints
        if global_epoch % checkpoint_interval == 0:
            for model, optimizer, enabled, name in [
                    (model_g, optimizer_g, update_g, "Generator"),
                    (model_d, optimizer_d, update_d, "Discriminator")]:
                if enabled:
                    save_checkpoint( model, optimizer, global_epoch, global_step, checkpoint_dir, name )

    return 0


def load_checkpoint(model, optimizer, checkpoint_path):
    global global_epoch
    print("Load checkpoint from: {}".format(checkpoint_path))
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    global_epoch = checkpoint["global_epoch"]
    global_epoch = checkpoint["global_step"]


if __name__ == "__main__":
    since = time.time()
    args = docopt(__doc__)
    print("Command line args:\n", args)
    hp = getattr(hparams_gan, args["--hparams_name"])

    # Override hyper parameters
    hp.parse(args["--hparams_gan"])

    checkpoint_dir = args["--checkpoint-dir"]
    checkpoint_path_d = args["--checkpoint-d"]
    checkpoint_path_g = args["--checkpoint-g"]
    checkpoint_path_r = args["--checkpoint-r"]
    max_files = int(args["--max_files"])
    w_d = float(args["--w_d"])
    mse_w = float(args["--mse_w"])
    mge_w = float(args["--mge_w"])
    discriminator_warmup = args["--discriminator-warmup"]
    restart_epoch = int(args["--restart_epoch"])

    reset_optimizers = args["--reset_optimizers"]
    log_event_path = args["--log-event-path"]

    # Flags to update discriminator/generator or not
    update_d = w_d > 0
    update_g = False if discriminator_warmup else True

    if not exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    X = []
    Y = []
    # utt_lengths = []

    X = FileSourceDataset(TextDataSource())
    Mel = FileSourceDataset(MelSpecDataSource())
    Y = FileSourceDataset(LinearSpecDataSource())

    print("Size of dataset for {}: {}".format(phase, len(X)))

    ty = "acoustic" if hp == hparams_gan.tts_acoustic else "duration"
    X_data_min, X_data_max = P.minmax( X )
    Mel_data_mean, Mel_data_var = P.meanvar( Mel )
    Mel_data_std = np.sqrt( Mel_data_var )

    np.save(join(data_dir, "X_{}_data_min".format(ty)), X_data_min)
    np.save(join(data_dir, "X_{}_data_max".format(ty)), X_data_max)
    np.save(join(data_dir, "Mel_{}_data_mean".format(ty)), Mel_data_mean)
    np.save(join(data_dir, "Mel_{}_data_var".format(ty)), Mel_data_var)

    if hp.discriminator_params["in_dim"] is None:
        sizes = get_static_stream_sizes(
            hp.stream_sizes, hp.has_dynamic_features, len(hp.windows))
        D = int(np.array(sizes[hp.adversarial_streams]).sum())
        if hp.adversarial_streams[0]:
            D -= hp.mask_nth_mgc_for_adv_loss
        if hp.discriminator_linguistic_condition:
            D = D + X_data_min.shape[-1]
        hp.discriminator_params["in_dim"] = D
    
    dataset = PyTorchDataset(X, Mel, Y)
    data_loader = data_utils.DataLoader(
        dataset, batch_size=hparams.batch_size,
        num_workers=hparams.num_workers, shuffle=True,
        collate_fn=collate_fn, pin_memory=hparams.pin_memory)

    # Models
    model_g = Tacotron(n_vocab = len(symbols),
                     embedding_dim = 256,
                     mel_dim = hparams.num_mels,
                     linear_dim = hparams.num_freq,
                     r = hparams.outputs_per_step,
                     padding_idx = hparams.padding_idx,
                     use_memory_mask = hparams.use_memory_mask,
                     )
    model_d = getattr(gantts_models, hp.discriminator)(**hp.discriminator_params)
    print("Generator:", model_g)
    print("Discriminator:", model_d)

    if use_cuda:
        model_g, model_d = model_g.cuda(), model_d.cuda()

    # Optimizers
    optimizer_g = optim.Adam( model.parameters(),
                           lr = hparams.initial_learning_rate, betas = ( hparams.adam_beta1, hparams.adam_beta2 ),
                           weight_decay = hparams.weight_decay )
    optimizer_d = getattr(optim, hp.optimizer_d)(model_d.parameters(),
                                                 **hp.optimizer_d_params)

    # Load checkpoint
    if checkpoint_path_d:
        if reset_optimizers:
            load_checkpoint(model_d, None, checkpoint_path_d)
        else:
            load_checkpoint(model_d, optimizer_d, checkpoint_path_d)
    if checkpoint_path_g:
        if reset_optimizers:
            load_checkpoint(model_g, None, checkpoint_path_g)
        else:
            load_checkpoint(model_g, optimizer_g, checkpoint_path_g)

    # Restart iteration at restart_epoch
    if restart_epoch >= 0:
        global_epoch = restart_epoch

    # Setup tensorboard logger
    if log_event_path is None:
        log_event_path = "log/run-test" + str(np.random.randint(100000))
    print("Loss event path: {}".format(log_event_path))
    tensorboard_logger.configure(log_event_path)

    # Train
    print("Start training from epoch {}".format(global_epoch))
    train_loop((model_g, model_d), (optimizer_g, optimizer_d),
               dataset_loaders, w_d=w_d, update_d=update_d, update_g=update_g,
               mse_w=mse_w, mge_w=mge_w)

    # Save models -- Training Done. Save whatever progress was made since the last checkpoint was saved.
    for model, optimizer, enabled, name in [
            (model_g, optimizer_g, update_g, "Generator"),
            (model_d, optimizer_d, update_d, "Discriminator")]:
        if enabled:
            save_checkpoint(
                model, optimizer, global_epoch, global_step, checkpoint_dir, name)

    print("Finished!")
    sys.exit(0)