# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional
import itertools
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix
from timm.data import Mixup
from timm.utils import accuracy

import matplotlib.pyplot as plt
import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import accuracy_score, f1_score

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None, device =torch.device, 
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))

    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        #with torch.cuda.amp.autocast(): #changed -> hashed out for 'cpu'
        outputs = model(samples)
        loss = criterion(outputs, targets)

        
        loss_value = loss.item()



        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        #torch.cuda.synchronize() #changed -> hashed out for 'cpu'

        metric_logger.update(loss=loss_value)

        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)


        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, args, confusion_matrix_plot=False, plot_save_name=None, plot_title=None, save_predictions=False):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    
    preds = list()
    targets = list()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)
            pred = torch.argmax(output.detach(), axis=1)
        
        preds.append(pred.cpu().numpy().tolist())
        targets.append(target.cpu().numpy().tolist())


        acc1, acc2 = accuracy(output, target, topk=(1, 2))     # changed - to top 2
        f1 = f1_score(target.cpu().numpy(), pred.cpu().numpy(), average='weighted')
        metric_logger.update(F1=f1)  
        
        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())

        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc2'].update(acc2.item(), n=batch_size)

    

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f'* Acc@1 {metric_logger.meters["acc1"].global_avg:.4f} '
        f'Acc@2 {metric_logger.meters["acc2"].global_avg:.3f} '
        f'loss {metric_logger.meters["loss"].global_avg:.3f} '
        f'F1 {metric_logger.meters["F1"].global_avg:.2f}')


    if confusion_matrix_plot:
        preds_list = list(itertools.chain.from_iterable(preds))
        targets_list = list(itertools.chain.from_iterable(targets))

        if save_predictions == True:
            torch.save(torch.tensor(preds_list), f'{args.data_path}/{plot_save_name}_pred.pt')


        if args.data == "UCIHAR":
            labels = ['Transition', 'Walking', 'Walking-upstairs', 'Walking-downstairs', 'Sitting', 'Standing', 'Laying']
        if args.data == "RISE":
            if args.use_transition_sub_label:
                if args.RISE_bin_label:
                    labels = ["sedentary", "active", "mixed"]
                else:
                    labels = ["sedentary", "standing", "stepping", "lying", "seated_transport", "active_transition", "mixed_transition"]
            else:
                if args.RISE_bin_label:
                    labels = ["sedentary", "active"]
                else:
                    labels = ["sedentary", "standing", "stepping", "lying", "seated_transport", "transition"]

        preds_tensor = torch.tensor(preds_list)
        targets_tensor = torch.tensor(targets_list)

        final_accordance = (preds_tensor == targets_tensor)
        final_acc1 = final_accordance.sum().item() / len(preds_list)

        # plot confusion matrix
        cm_test = confusion_matrix(targets_list, preds_list)
        plt.figure(figsize=(8, 8))
        sns.heatmap(cm_test, annot=True, cmap='Blues', fmt='d', xticklabels=labels,
                    yticklabels=labels, cbar=False, linewidth=.5, annot_kws={"fontsize":16})
        plt.xlabel('Predicted Labels', fontsize=13)
        plt.ylabel('True Labels', fontsize=13)
        plt.title(f'{plot_title}; Accuracy = {final_acc1:.4f}')
        plt.show()
        plt.savefig(f'/niddk-data-central/P2/mae_hr/RISE_PH/plots/{plot_save_name}_confusion_matrix.png', bbox_inches='tight')


    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}