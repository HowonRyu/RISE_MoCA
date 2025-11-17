from torchvision import transforms
import torch
import numpy as np

import matplotlib.pyplot as plt
from functools import partial

import torch
import torch.nn as nn

from models_vit import VisionTransformer

from timm.models.vision_transformer import PatchEmbed, Block
from timm.models.layers import to_2tuple

from util.pos_embed import get_2d_sincos_pos_embed

from models_mae import MaskedAutoencoderViT

import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def model_load_visualize(alt, chkpt_dir, patch_size1=1, image_length=200, n_patches=10):
    patch_length = int(image_length/n_patches)
    ##############################

    if alt:
        def mae_vit_tiny_patch16_dec256d8b(**kwargs):
            model = MaskedAutoencoderViT(
                img_size=[6, int(image_length)], patch_size=[patch_size1, int(patch_length)], in_chans=1,
                embed_dim=192, depth=12, num_heads=3, 
                decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=8,
                mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), alt=alt, **kwargs)
            return model
        def mae_vit_base_patch16_dec512d8b(**kwargs):
            model = MaskedAutoencoderViT(
                img_size=[6, int(image_length)], patch_size=[patch_size1, int(patch_length)], in_chans=1,
                embed_dim=768, depth=12, num_heads=12,
                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
            return model

    else:
        def mae_vit_tiny_patch16_dec256d8b(**kwargs):
            model = MaskedAutoencoderViT(
                img_size=[1, int(image_length)], patch_size=[patch_size1, int(patch_length)], in_chans=6,
                embed_dim=192, depth=12, num_heads=3, 
                decoder_embed_dim=256, decoder_depth=8, decoder_num_heads=8,
                mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), alt=alt, **kwargs)
            return model
        def mae_vit_base_patch16_dec512d8b(**kwargs):
            model = MaskedAutoencoderViT(
                img_size=[1, int(image_length)], patch_size=[patch_size1, int(patch_length)], in_chans=6,
                embed_dim=768, depth=12, num_heads=12,
                decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
            return model

    # set recommended archs
    mae_vit_base_patch_16 = mae_vit_base_patch16_dec512d8b

    # get model from checkpoint
    model_mae = mae_vit_base_patch_16()

    checkpoint = torch.load(chkpt_dir, map_location='cpu')
    model_mae.load_state_dict(checkpoint['model'], strict=False)
    print(model_mae.load_state_dict(checkpoint['model'], strict=False))

    return model_mae


def mae_pass(x, model, mask_ratio, alt, norm_pix_loss=False, import_mask=False, seed=None):
    model.alt = alt
    x = x.unsqueeze(dim=0) # changing the train1 tensor to conform to batch dimensions (1, h, w, c)
    in_chans = x.shape[-1]
    x = torch.einsum('nhwc->nchw', x) # (1, h, w, c) -> (1, c, h, w)

    # if norm_pix_loss enabled, normalize x patch-wise
    if norm_pix_loss:
        x_patch = model.patchify(x)  # patchify to get patch-wise mean and variance
        mean = x_patch.mean(dim=-1, keepdim=True)
        var = x_patch.var(dim=-1, keepdim=True)
        x_patch_norm = (x_patch - mean) / (var + 1.e-6)**.5
        x = model.unpatchify(x_patch_norm)


    # load reconstructed y and masks - deconstructed self.forward() to designate masking
    loss, y, mask = model(x.float(), mask_ratio=mask_ratio, import_mask=import_mask, seed=seed) 
    #print("y shape:", y.shape, ", mask (1-remove or 0-keep) shape:", mask.shape)

    y = model.unpatchify(y)
    y = torch.einsum('nchw->nhwc', y).detach().cpu()


    # make mask into the same dimension as x so we can apply operation in the next step
    #print("mask shape before:", mask.shape)
    mask = mask.detach()
    mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]* model.patch_embed.patch_size[1] * in_chans)  # (N, H*W, p0*p1*3)  #changed
    
    mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
    mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
    #print("mask shape after:", mask.shape)


    # revert x back to (1, h, w, c)
    x_prime = torch.einsum('nchw->nhwc', x)


    # masked image
    im_masked = x_prime * (1 - mask)

    # MAE reconstruction pasted with visible patches
    im_paste = x_prime * (1 - mask) + y * mask

    return im_masked, im_paste, x_prime, y, loss



def reconstruct_tjct(X, model, mask_ratio, norm_pix_loss, alt):
    masked_all = torch.Tensor()
    paste_all = torch.Tensor()
    x_prime_all = torch.Tensor()
    y_all = torch.Tensor()
    x_squeeze_all = torch.Tensor()
    loss_MSE = list()
    loss_nMSE = list()

    for i in range(len(X)):
        x = X[i,:,:,:]
        
        if alt:
            x = x.permute(2, 1, 0)
            
        masked, paste, x_prime, y, loss = mae_pass(x=x, model=model, mask_ratio=mask_ratio, alt=alt, norm_pix_loss=norm_pix_loss)
        x_squeeze = x.squeeze(dim=0)

        masked_all = torch.cat((masked_all, masked), dim=0)
        paste_all = torch.cat((paste_all, paste), dim=0)
        x_prime_all = torch.cat((x_prime_all, x_prime), dim=0)
        y_all = torch.cat((y_all, y), dim=0)

        # loss
        if alt:
            loss_MSE.append(loss[0].detach().numpy())
            loss_nMSE.append(loss[1].detach().numpy())
        else:
            loss_MSE.append(loss[0][0].detach().numpy())
            loss_nMSE.append(loss[0][1].detach().numpy())


        x_squeeze_all = torch.cat((x_squeeze_all, x_squeeze), dim=0)

    if alt:
        masked_all = masked_all.permute(0,3,2,1)
        paste_all = paste_all.permute(0,3,2,1)
        x_prime_all = x_prime_all.permute(0,3,2,1)
        y_all = y_all.permute(0,3,2,1)
        #x_squeeze_all = x_squeeze_all.permute(2, 1, 0)

    return masked_all, paste_all, x_prime_all, y_all, x_squeeze_all, loss_MSE, loss_nMSE




def plot_all_reconstruction(masked_all, x_prime_all, y_all, X_test_subseq_indicator, label_y, sample_sub, loss_MSE, loss_nMSE, test_only=False,  ms=0.5, figsize=(30, 50)):
    fig, axs = plt.subplots(12, 1, figsize=figsize, plot_save=False, plot_save_name=None)
    plot_labels = ['Gyr_X', 'Gyr_Y', 'Gyr_Z', 'Acc_X', 'Acc_Y', 'Acc_Z']


    image_length = masked_all.shape[2]
    cutoff = 17600
    n_images_in_trjt = int(cutoff/image_length)

    label_y_per_sub = label_y[sample_sub*n_images_in_trjt:(sample_sub+1)*n_images_in_trjt]


    test_indicator = list(X_test_subseq_indicator[:,0,0,0])
    test_loss_MSE = [a for a, b in  zip(loss_MSE, test_indicator) if b == 1]
    training_loss_MSE = [a for a, b in  zip(loss_MSE, test_indicator) if b == 0]

    test_loss_nMSE = [a for a, b in  zip(loss_nMSE, test_indicator) if b == 1]
    training_loss_nMSE = [a for a, b in  zip(loss_nMSE, test_indicator) if b == 0]


    test_loss_MSE = np.round(np.mean(test_loss_MSE),4)
    training_loss_MSE = np.round(np.mean(training_loss_MSE),4)

    test_loss_nMSE = np.round(np.mean(test_loss_nMSE),4)
    training_loss_nMSE = np.round(np.mean(training_loss_nMSE),4)

    print(f"training MSE: {training_loss_MSE}, training nMSE: {training_loss_nMSE}, test MSE: {test_loss_MSE}, test nMSE: {test_loss_nMSE}")

    if test_only:    
        title =  f"Test images, subject{sample_sub}"
    else:
        title =  f"Train + Test images, subject{sample_sub}"


    for i in range(6):
        # reshape data
        masked_plot = masked_all[sample_sub*n_images_in_trjt : (sample_sub+1)*n_images_in_trjt, 0, :, i].reshape(-1)
        #paste_plot = paste_all[sample_sub*n_images_in_trjt:(sample_sub+1)*n_images_in_trjt,0,:,i].reshape(-1)
        x_prime_plot = x_prime_all[sample_sub*n_images_in_trjt:(sample_sub+1)*n_images_in_trjt,0,:,i].reshape(-1)
        y_plot = y_all[sample_sub*n_images_in_trjt:(sample_sub+1)*n_images_in_trjt,0,:,i].reshape(-1)


        # test indicator
        test_indicator = X_test_subseq_indicator[sample_sub*n_images_in_trjt:(sample_sub+1)*n_images_in_trjt,0,:,0].reshape(-1)
        background_test_mask = np.where(test_indicator == 1, 1, np.nan).astype(int)



        # label
        label_y_list = np.repeat(np.array(label_y_per_sub), image_length).tolist()
        activity_labels = {0: "TRANSITION", 1: "WALKING", 2:"WALKING-DOWN", 3:"WALKING-UP", 4: "SITTING", 5: "STANDING", 6: "LAYING"}

        # y limits
        ymin, ymax = -2, 6.5

        if test_only:
            masked_plot_test_only =  [a for a, b in zip(masked_plot, test_indicator) if b == 1]
            x_prime_plot_test_only  = [a for a, b in zip(x_prime_plot, test_indicator) if b == 1]
            y_plot_test_only  = [a for a, b in zip(y_plot, test_indicator) if b == 1]
            label_y_list_test_only = [a for a, b in zip(label_y_list, test_indicator) if b == 1]

            # plot
            axs[i*2].plot(x_prime_plot_test_only, label="original (masked)", marker='o', color='blue', linestyle="-", markersize=ms) # original but will be painted over by un_masked
            unmasked_x = masked_plot_test_only != 0
            indices_x = np.where(masked_plot_test_only != 0)[0]
            axs[i*2].plot(indices_x, masked_plot_test_only[unmasked_x], label="original (unmasked)", color='orange', marker='o', linestyle="", markersize=ms)
            axs[i*2].legend()
            axs[i*2].set_title(f"{plot_labels[i]} original")
            axs[i*2].plot(label_y_list_test_only, label="label", marker='o', linestyle="-", color='black', markersize=ms)

            axs[i*2].set_yticks(range(0,7))  # Set the range from 1 to 6
            axs[i*2].set_yticklabels([activity_labels[label] for label in range(0,7)])  # Map labels to activity names

            axs[i*2+1].plot(y_plot_test_only, label="reconstructed", color='green', marker='o', linestyle="-", markersize=ms, alpha=1)
            axs[i*2+1].plot(label_y_list_test_only, label="label", marker='o', linestyle="-", color='black', markersize=ms)

            #axs[0].plot(paste[0,0,:,0], label="reconstruction + visible")
            axs[i*2+1].legend()
            axs[i*2+1].set_title(f"{plot_labels[i]} reconstructed")
            axs[i*2+1].set_ylim([ymin, ymax])
            axs[i*2+1].set_yticks(range(0,7))  # Set the range from 0 to 6
            axs[i*2+1].set_yticklabels([activity_labels[label] for label in range(0,7)])  # Map labels to activity names


        else:
            # plot
            axs[i*2].plot(x_prime_plot, label="original (masked)", marker='o', color='blue', linestyle="-", markersize=ms) # original but will be painted over by un_masked
            unmasked_x = masked_plot != 0
            indices_x = np.where(masked_plot != 0)[0]
            axs[i*2].plot(indices_x, masked_plot[unmasked_x], label="original (unmasked)", color='orange', marker='o', linestyle="", markersize=ms)
            axs[i*2].legend()
            axs[i*2].set_title(f"{plot_labels[i]} original")
            axs[i*2].plot(label_y_list, label="label", marker='o', linestyle="-", color='black', markersize=ms)
            axs[i*2].fill_between(np.arange(cutoff), ymin, ymax, where=background_test_mask, color='white', alpha=0.3)
            axs[i*2].fill_between(np.arange(cutoff), ymin, ymax, where=~background_test_mask, color='grey', alpha=0.3)
            axs[i*2].set_yticks(range(0,7))  # Set the range from 1 to 6
            axs[i*2].set_yticklabels([activity_labels[label] for label in range(0,7)])  # Map labels to activity names



            axs[i*2+1].plot(y_plot, label="reconstructed", color='green', marker='o', linestyle="-", markersize=ms, alpha=1)
            axs[i*2+1].plot(label_y_list, label="label", marker='o', linestyle="-", color='black', markersize=ms)

            #axs[0].plot(paste[0,0,:,0], label="reconstruction + visible")
            axs[i*2+1].legend()
            axs[i*2+1].set_title(f"{plot_labels[i]} reconstructed")
            axs[i*2+1].set_ylim([ymin, ymax])
            axs[i*2+1].set_yticks(range(0,7))  # Set the range from 0 to 6
            axs[i*2+1].set_yticklabels([activity_labels[label] for label in range(0,7)])  # Map labels to activity names
            axs[i*2+1].fill_between(np.arange(cutoff), ymin, ymax, where=background_test_mask, color='white', alpha=0.3)
            axs[i*2+1].fill_between(np.arange(cutoff), ymin, ymax, where=~background_test_mask, color='grey', alpha=0.3)


    plt.suptitle(title)
    plt.subplots_adjust(hspace=0.5, top=0.97)

    if plot_save:
        plt.savefig(f'{plot_save_name}_recon_plot.png')
    else:
        plt.show()


def plot_reconstruction(model, x, alt, mask_ratio=0.75, norm_pix_loss=False, title = None, marg = 0.2, import_mask=False, seed=None):
    if alt:
        x = x.permute(2, 1, 0)

    masked, paste, x_prime, y, loss = mae_pass(x=x, model=model, mask_ratio=mask_ratio, norm_pix_loss=norm_pix_loss, alt=alt, import_mask=import_mask, seed=seed)

    print(f"test loss= {loss}")
    if alt:
        masked = masked.permute(0,3,2,1)
        paste = paste.permute(0,3,2,1)
        x_prime = x_prime.permute(0,3,2,1)
        y = y.permute(0,3,2,1)
        

    # y limits
    y_squeeze = y.squeeze(dim=0).squeeze(dim=0)
    all_data = torch.cat((y, x_prime), dim=0)
    ymin_lims = torch.amin(all_data, dim=(0, 1, 2))
    ymax_lims = torch.amax(all_data, dim=(0, 1, 2))
    ms = 2
    linestyle = '-'
    linestyle_unmasked = ''
    modalities = ["Gyr", "Acc"]
    axes = ['X', "Y", "Z"]
    fig, axs = plt.subplots(len(modalities)*2, len(axes), figsize=(18, 13))

    for i, modal in enumerate(modalities):
        for j, ax in enumerate(axes):
            # xaxis
            axs[2*i,j].plot(x_prime[0, 0, :, 3*i+j], label="original (masked)", color="black", marker='o', linestyle=linestyle, markersize=ms) # original but will be painted over by un_masked
            unmasked_x = masked[0, 0, :, 3*i+j] != 0
            indices_x = np.where(masked[0, 0, :, 3*i+j] != 0)[0]
            axs[2*i,j].plot(indices_x, masked[0,0,unmasked_x,3*i+j], label="original (unmasked)", color="orange", marker='o', linestyle=linestyle_unmasked, markersize=ms)
            axs[2*i,j].legend()
            axs[2*i,j].set_title(f"{ax}_{modal}")
            axs[2*i,j].set_ylim([ymin_lims[3*i+j]-marg, ymax_lims[3*i+j]+marg])

            axs[2*i+1,j].plot(y[0,0,:,3*i+j], label="reconstructed", color="blue", marker='o', linestyle=linestyle, markersize=ms)
            #axs[i,j].plot(paste[0,0,:,i], label="reconstruction + visible")
            axs[2*i+1,j].legend()
            axs[2*i+1,j].set_title(f"{ax}_{modal}")
            axs[2*i+1,j].set_ylim([ymin_lims[3*i+j]-marg, ymax_lims[3*i+j]+marg])

    plt.suptitle(f"Reconstruction {title}")
    plt.subplots_adjust(hspace=0.5, top=0.97)
    plt.show()

def mae_classification_pass(X, y, model, alt):

    if alt:
        X = torch.Tensor(X).permute(0, 3, 2, 1)

    X = X.permute(0, 3, 1, 2) 
    dataset = torch.utils.data.TensorDataset(torch.Tensor(X),  y)
    sampler = torch.utils.data.SequentialSampler(dataset)
    data_loader = torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=50, num_workers=1, drop_last=False)
    
    preds = list()
    targets = list()
    for batch in iter(data_loader):
        images = batch[0]
        target = batch[-1]

        output = model(images)
        pred = np.argmax(output.detach().numpy(), axis=1)

        preds.append(pred)
        targets.append(target.detach().numpy())

    preds_list = list(itertools.chain.from_iterable(preds))
    targets_list = list(itertools.chain.from_iterable(targets))

    accordance = np.array(preds_list) == np.array(targets_list)
    accuracy = np.sum(accordance)/len(preds_list)

    return preds_list, targets_list, accuracy



