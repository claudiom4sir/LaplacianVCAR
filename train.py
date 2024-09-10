import os.path
from dataset import Dataset
import torch.optim as optim
from tensorboardX import SummaryWriter
import utils
from torchmetrics import PeakSignalNoiseRatio
from torch.utils.data import DataLoader
from stdf_laplacian import MFVQE, LaplacianLossImages
import random
import torch
import numpy as np
from tqdm import tqdm
from multiprocessing import cpu_count
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from colors import rgb2ycbcr
import argparse

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():

    parser = argparse.ArgumentParser()

    # experiment arguments
    parser.add_argument('--seed', default=12345, type=int, help='Seed for reproducibility')
    parser.add_argument('--exp', type=str, default='./exp/myexp/', help='The name of the experiment')

    # dataset arguments
    parser.add_argument('--in_path_compressed_train', type=str, default='/home/claudiorota/hdd/Datasets/MFQEv2_Dataset/MFQEv2_RGB/train_108/HM16.5_LDP/QP37/', help='Path to compressed sequences for training')
    parser.add_argument('--in_path_gt_train', type=str, default='/home/claudiorota/hdd/Datasets/MFQEv2_Dataset/MFQEv2_RGB/train_108/raw/', help='Path to reference sequences for training')
    parser.add_argument('--in_path_compressed_eval', type=str, default='/home/claudiorota/hdd/Datasets/MFQEv2_Dataset/MFQEv2_RGB/test_18/HM16.5_LDP/QP37/', help='Path to compressed sequences for evaluation')
    parser.add_argument('--in_path_gt_eval', type=str, default='/home/claudiorota/hdd/Datasets/MFQEv2_Dataset/MFQEv2_RGB/test_18/raw/', help='Path to reference sequences for evaluation')
    parser.add_argument('--excluded_path_txt', type=str, default='/home/claudiorota/hdd/Datasets/MFQEv2_Dataset/MFQEv2_RGB/test_18/excluded.txt', help='Txt file contaning the name of the sequences to exclude during evaluation')

    # training and test arguments
    parser.add_argument('--batch_size', default=8, type=int, help='The size of batch to use for training')
    parser.add_argument('--lr', default=1e-4, type=float, help='Initial learning rate')
    parser.add_argument('--lr_scheduler', type=dict, default={401:1e-5, 801:1e-6}, help='Dictionary contaning the scheduler for lr decay. It must be in format {epoch1:lr1, epoch2:lr2, ...}')
    parser.add_argument('--epochs', default=1000, type=int, help='Number of epochs for training')
    parser.add_argument('--pretrained_path', type=str, default='', help='Path to pre-trained models to resume training')
    parser.add_argument('--display_every', default=500, type=int, help='Display results on Tensorboard after every X iterations')
    parser.add_argument('--eval_every', default=100, type=int, help='Run evaluatation every X epochs')
    parser.add_argument('--patch_size', default=64, type=int, help='Dimension of the patchs for training')

    # network parameters
    parser.add_argument('--model_name', default='XS', choices=['XS', 'S', 'M', 'L', 'XL'], help='Model name (must be XS, S, M, L, XL)')
    parser.add_argument('--lap_levels', default=4, type=int, help='Number of levels in the Laplacian pyramid')
    parser.add_argument('--window_size', default=7, type=int, help='Temporal neighbourhood size')
    parser.add_argument('--std', type=float, default=1, help='Standard deviation for Gaussian blur in Laplacian decomposition')

    args = parser.parse_args()

    print('> Running with the following arguments:')
    for arg, value in vars(args).items():
        print(f'    {arg}: {value}')

    # deterministic
    seed = args.seed
    seed_everything(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # dataset paths
    path_compressed_train, path_gt_train = args.in_path_compressed_train, args.in_path_gt_train
    path_compressed_test, path_gt_test = args.in_path_compressed_eval, args.in_path_gt_eval
    excluded_path_test = args.excluded_path_txt

    # parameters
    batch_size = args.batch_size
    lr = args.lr
    epochs = args.epochs
    exp_name = args.exp
    pretrained_path = args.pretrained_path
    disp_every = args.display_every
    evaluate_every = args.eval_every
    temporal_neighborhood = args.window_size
    patch_size = args.patch_size
    lr_scheduler = args.lr_scheduler
    pyramid_levels = args.lap_levels
    model_name = args.model_name
    std = args.std

    print('> Experiment ' + exp_name)
    os.makedirs(exp_name, exist_ok=True)

    writer = SummaryWriter('runs/' + exp_name)

    # define model and optimizer
    device = torch.device('cuda')
    model = utils.create_model(model_name, pyramid_levels, std)
    model = model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    if pretrained_path != '':
        print('> Loading pretrained model from ' + pretrained_path)
        model.load_state_dict(torch.load(pretrained_path)['model_state_dict'])
        optimizer.load_state_dict(torch.load(pretrained_path)['optimizer_state_dict'])
        print('> Loading complete.')

    # load trainset and testset
    trainset = Dataset(path_compressed_train, path_gt_train, temp_size=temporal_neighborhood, max_video_number=1e4,
                       max_seq_length=1e4, is_cropped=False, patch_size=patch_size)
    trainset_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=cpu_count(), worker_init_fn=seed_worker, generator=g)
    testset = Dataset(path_compressed_test, path_gt_test, temp_size=temporal_neighborhood, is_test=True,
                      max_video_number=1e4, max_seq_length=50, excluded_path=excluded_path_test, is_cropped=False)
    testset_loader = DataLoader(testset, num_workers=1)

    # define loss function
    criterion = LaplacianLossImages(levels=pyramid_levels, std=1).to(device)
    psnr = PeakSignalNoiseRatio(data_range=1).to(device)

    best_delta_psnr = -1000 if pretrained_path == '' else torch.load(pretrained_path)['best_delta_psnr']
    start_epoch = 1 if pretrained_path == '' else torch.load(pretrained_path)['epoch'] + 1
    it = 1 if pretrained_path == '' else start_epoch * len(trainset_loader)

    print('> Start training...')
    for epoch in range(start_epoch, epochs + 1):
        print(f'> Training. Epoch {epoch}/{epochs}')
        model.train()
        epoch_delta_psnr_y = 0
        epoch_psnr_y = 0
        epoch_loss = 0
        if epoch in lr_scheduler.keys():
            for g in optimizer.param_groups:
                g['lr'] = lr_scheduler[epoch]
            print(f'> Current learning rate set to {lr_scheduler[epoch]}')
        pbar = tqdm(total=len(trainset_loader), ncols=100)
        for batch_n, data in enumerate(trainset_loader, 1):
            compressed, gt = data
            compressed, gt = compressed.to(device), gt.to(device)
            compressed_ycbcr = torch.cat([rgb2ycbcr(compressed[:, i:i + 3]) for i in range(0, compressed.shape[1], 3)], dim=1)
            compressed_y = torch.cat([compressed_ycbcr[:, i:i + 1] for i in range(0, compressed_ycbcr.shape[1], 3)], dim=1)
            gt_y = rgb2ycbcr(gt)[:, 0:1]

            restored_y, rec, laps = model(compressed_y)

            compressed_y = compressed_y[:, temporal_neighborhood // 2:temporal_neighborhood // 2 + 1]

            optimizer.zero_grad()

            loss = criterion(rec, gt_y)

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            with torch.no_grad():
                psnr_restored_y = psnr(restored_y, gt_y)
                psnr_compressed_y = psnr(compressed_y, gt_y)
                delta_psnr_y = psnr_restored_y - psnr_compressed_y
                epoch_delta_psnr_y += delta_psnr_y
                epoch_psnr_y += psnr_restored_y
                pbar.set_description(f'dPSNR_Y: {delta_psnr_y:.3f}, PSNR_Y: {psnr_restored_y:.3f}')
                if it % disp_every == 0:
                    writer.add_scalar('Train/Loss', epoch_loss / batch_n, it)
                    writer.add_scalar('Train/PSNR/Delta_Y', epoch_delta_psnr_y / batch_n, it)
                    writer.add_scalar('Train/PSNR/Restored_Y', epoch_psnr_y / batch_n, it)
                    utils.add_batch_in_tensorboard('Train/Compressed', writer, compressed_y.clamp(0, 1), it)
                    utils.add_batch_in_tensorboard('Train/Ground truth', writer, gt_y.clamp(0, 1), it)
                    utils.add_batch_in_tensorboard('Train/Restored', writer, restored_y.clamp(0, 1), it)
                    writer.flush()
            pbar.update()
            it += 1
        pbar.close()
        epoch_loss /= len(trainset_loader)
        epoch_psnr_y /= len(trainset_loader)
        epoch_delta_psnr_y /= len(trainset_loader)
        print(f'> Epoch {epoch} finished with dPSNR_Y: {epoch_delta_psnr_y:.3f}, PSNR_Y: {epoch_psnr_y:.3f}')
        if epoch % evaluate_every == 0:
            model.eval()
            test_loss = 0
            total_delta_psnr_y = 0
            total_psnr_y = 0
            print('> Evaluation.')
            pbar = tqdm(total=len(testset_loader), ncols=100)
            with torch.no_grad():
                for batch_n, data in enumerate(testset_loader, 1):
                    compressed, gt = data
                    compressed, gt = compressed.to(device), gt.to(device)

                    compressed_ycbcr = torch.cat(
                        [rgb2ycbcr(compressed[:, i:i + 3]) for i in range(0, compressed.shape[1], 3)], dim=1)
                    compressed_y = torch.cat(
                        [compressed_ycbcr[:, i:i + 1] for i in range(0, compressed_ycbcr.shape[1], 3)], dim=1)
                    gt_y = rgb2ycbcr(gt)[:, 0:1]

                    restored_y, rec, _ = model(compressed_y)

                    compressed_y = compressed_y[:, temporal_neighborhood // 2:temporal_neighborhood // 2 + 1]

                    loss = criterion(restored_y, gt_y)

                    test_loss += loss.item()
                    psnr_restored_y = psnr(restored_y, gt_y)
                    psnr_compressed_y = psnr(compressed_y, gt_y)
                    delta_psnr_y = (psnr_restored_y - psnr_compressed_y)
                    total_delta_psnr_y += delta_psnr_y
                    total_psnr_y += psnr_restored_y
                    pbar.set_description(
                        f'dPSNR_Y: {delta_psnr_y:.3f}, PSNR_Y: {psnr_restored_y:.3f}')
                    pbar.update()
            pbar.close()
            total_delta_psnr_y /= len(testset_loader)
            total_psnr_y /= len(testset_loader)
            test_loss /= len(testset_loader)
            print(f'> Evaluation finished with dPSNR_y: {total_delta_psnr_y:.3f}, PSNR_Y: {total_psnr_y:.3f}')
            writer.add_scalar('Test/PSNR/Delta_Y', total_delta_psnr_y, it)
            writer.add_scalar('Test/PSNR/Restored_Y', total_psnr_y, it)
            writer.flush()
            if total_delta_psnr_y > best_delta_psnr:
                best_delta_psnr = total_delta_psnr_y
                print(f'> Saving best model with dPSNR {best_delta_psnr:.3f}...')
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(), 'it': it, 'best_delta_psnr':
                                best_delta_psnr}, f'{exp_name}/best.pth')
            print(f'> Best dPSNR so far: {best_delta_psnr}')
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(), 'it': it, 'best_delta_psnr': best_delta_psnr},
                   f'{exp_name}/epoch_{epoch}.pth' % (exp_name, epoch))

    print(f'> Training complete. Best dPSNR on test set: {best_delta_psnr:.3f}')
    writer.close()

if __name__ == '__main__':
    main()