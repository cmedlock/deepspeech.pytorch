import argparse
import json
import os
import time

import torch.distributed as dist
import torch.utils.data.distributed
from tqdm import tqdm
from warpctc_pytorch import CTCLoss

#import warpctc_pytorch
#print(warpctc_pytorch.__file__)

from data.data_loader_cm import AudioDataLoader, FeatureDataset, BucketingSampler, supported_feature_types
from data.utils import reduce_tensor
from decoder import GreedyDecoder
from model import DeepSpeech, supported_rnns

import params_cm

# Read in rest of parameters
# Model and feature type
model_type_ = params_cm.model_type
feature_type_ = params_cm.feature_type
# Audio sampling parameters
sample_rate_ = params_cm.sample_rate
window_size_ = params_cm.window_size
window_stride_ = params_cm.window_stride
window_ = params_cm.window
# Audio noise parameters
noise_dir_ = params_cm.noise_dir
noise_prob_ = params_cm.noise_prob
noise_min_ = params_cm.noise_min
noise_max_ = params_cm.noise_max
# Dataset and model save location
# Note: for ResNet50 must use dataset with pre-aligned transcriptions (e.g., TIMIT)
labels_path_ = params_cm.labels_path
train_manifest_ = params_cm.train_manifest
val_manifest_ = params_cm.val_manifest
model_path_ = params_cm.model_path
# Model parameters if model type is DeepSpeech
hidden_size_ = params_cm.hidden_size
hidden_layers_ = params_cm.hidden_layers
bias_ = params_cm.bias
rnn_type_ = params_cm.rnn_type
rnn_act_type_ = params_cm.rnn_act_type
bidirectional_ = params_cm.bidirectional
# Training parameters
epochs_ = params_cm.epochs
learning_anneal_ = params_cm.learning_anneal
lr_ = params_cm.lr
momentum_ = params_cm.momentum
max_norm_ = params_cm.max_norm
l2_ = params_cm.l2
batch_size_ = params_cm.batch_size
augment_ = params_cm.augment
exit_at_acc_ = params_cm.exit_at_acc
num_workers_ = params_cm.num_workers
cuda_ = params_cm.cuda

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

def to_np(x):
    return x.data.cpu().numpy()


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == '__main__':
    
    # Load symbols
    with open(labels_path_) as label_file:
        labels = str(''.join(json.load(label_file)))

    # Load audio parameters
    audio_conf = dict(sample_rate=sample_rate_,
                      window_size=window_size_,
                      window_stride=window_stride_,
                      window=window_,
                      noise_dir=noise_dir_,
                      noise_prob=noise_prob_,
                      noise_levels=(noise_min_,noise_max_))

    # Define model
    rnn_type = rnn_type_.lower()
    assert rnn_type in supported_rnns, 'rnn_type should be either lstm, rnn or gru'
    assert feature_type_ in supported_feature_types,'feature_type_ should be rawspeech, rawframes, spectrogram, mfcc, or logmel'
    if feature_type_=='rawspeech' and model_type_=='DeepSpeech':
        print('Error: Use rawframes for DeepSpeech instead of rawspeech')
        raise SystemExit
    elif feature_type_=='rawframes' and model_type_=='Wav2Letter':
        print('Error: Use rawspeech for Wav2Letter instead of rawframes')
        raise SystemExit
    model = DeepSpeech(feature_type=feature_type_,
                       rnn_hidden_size=hidden_size_,
                       nb_layers=hidden_layers_,
                       labels=labels,
                       rnn_type=supported_rnns[rnn_type],
                       audio_conf=audio_conf,
                       bidirectional=bidirectional_)
    parameters = model.parameters()
    
    print(model)
    #print("Number of parameters: %d" % DeepSpeech.get_param_size(model))

    # Define optimizer
    optimizer = torch.optim.SGD(parameters, lr=lr_,momentum=momentum_, nesterov=True)

    # Define loss function for training
    criterion = CTCLoss()
    
    # Define decoder for validation during training
    decoder = GreedyDecoder(labels)
    
    # Load and pre-process datasets
    train_dataset = FeatureDataset(audio_conf=audio_conf, feature_type=feature_type_, manifest_filepath=train_manifest_,                                            labels=labels, normalize=True, augment=augment_)
    test_dataset = FeatureDataset(audio_conf=audio_conf, feature_type=feature_type_, manifest_filepath=val_manifest_,                                             labels=labels, normalize=True, augment=False)

    train_sampler = BucketingSampler(train_dataset, batch_size=batch_size_)

    train_loader = AudioDataLoader(train_dataset, num_workers=num_workers_, batch_sampler=train_sampler)
    test_loader = AudioDataLoader(test_dataset, batch_size=batch_size_, num_workers=num_workers_)

    if cuda_:
        model.cuda()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    loss_results = torch.Tensor(epochs_)
    cer_results = torch.Tensor(epochs_)
    wer_results = torch.Tensor(epochs_)

    best_wer = None

    avg_loss = 0
    start_epoch = 0
    start_iter = 0

    print("Shuffling batches for the following epochs")
    train_sampler.shuffle(start_epoch)

    for epoch in range(start_epoch, epochs_):
        model.train()
        end = time.time()
        start_epoch_time = time.time()
        for i, (data) in enumerate(train_loader, start=start_iter):
            if i == len(train_sampler):
                break
            inputs, targets, input_percentages, target_sizes = data
            input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

            # Measure data loading time
            data_time.update(time.time() - end)

            if cuda_:
                inputs = inputs.cuda()

            out, output_sizes = model(inputs, input_sizes)
            out = out.transpose(0, 1)  # TxNxH

            loss = criterion(out, targets, output_sizes, target_sizes)
            loss = loss / inputs.size(0)  # average the loss by minibatch

            inf = float("inf")
            loss_value = loss.item()
            if loss_value == inf or loss_value == -inf:
                print("WARNING: received an inf loss, setting loss value to 0")
                loss_value = 0

            avg_loss += loss_value
            losses.update(loss_value, inputs.size(0))

            # Compute gradient
            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm_)
            # SGD step
            optimizer.step()

            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                  (epoch + 1), (i + 1), len(train_sampler), batch_time=batch_time, data_time=data_time, loss=losses))
            del loss
            del out
            
        avg_loss /= len(train_sampler)

        epoch_time = time.time() - start_epoch_time
        print('Training Summary Epoch: [{0}]\t'
              'Time taken (s): {epoch_time:.0f}\t'
              'Average Loss {loss:.3f}\t'.format(epoch + 1, epoch_time=epoch_time, loss=avg_loss))

        start_iter = 0  # Reset start iteration for next epoch
        total_cer, total_wer = 0, 0
        model.eval()
        with torch.no_grad():
            for i, (data) in tqdm(enumerate(test_loader), total=len(test_loader)):
                inputs, targets, input_percentages, target_sizes = data
                input_sizes = input_percentages.mul_(int(inputs.size(3))).int()

                # Unflatten targets
                split_targets = []
                offset = 0
                for size in target_sizes:
                    split_targets.append(targets[offset:offset + size])
                    offset += size

                if cuda_:
                    inputs = inputs.cuda()

                out, output_sizes = model(inputs, input_sizes)

                decoded_output, _ = decoder.decode(out.data, output_sizes)
                target_strings = decoder.convert_to_strings(split_targets)
                wer, cer = 0, 0
                for x in range(len(target_strings)):
                    transcript, reference = decoded_output[x][0], target_strings[x][0]
                    wer += decoder.wer(transcript, reference) / float(len(reference.split()))
                    cer += decoder.cer(transcript, reference) / float(len(reference))
                total_cer += cer
                total_wer += wer
                del out
            wer = total_wer / len(test_loader.dataset)
            cer = total_cer / len(test_loader.dataset)
            wer *= 100
            cer *= 100
            loss_results[epoch] = avg_loss
            wer_results[epoch] = wer
            cer_results[epoch] = cer
            print('Validation Summary Epoch: [{0}]\t'
                  'Average WER {wer:.3f}\t'
                  'Average CER {cer:.3f}\t'.format(epoch + 1, wer=wer, cer=cer))

            # Anneal lr
            optim_state = optimizer.state_dict()
            optim_state['param_groups'][0]['lr'] = optim_state['param_groups'][0]['lr'] / learning_anneal_
            optimizer.load_state_dict(optim_state)
            print('Learning rate annealed to: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
            if (best_wer is None or best_wer > wer):
                print("Found better validated model, saving to %s" % model_path_)
                torch.save(DeepSpeech.serialize(model, optimizer=optimizer, epoch=epoch, loss_results=loss_results,
                                                wer_results=wer_results, cer_results=cer_results), model_path_)
                best_wer = wer
                avg_loss = 0

            print("Shuffling batches...")
            train_sampler.shuffle(epoch)
