
from dataset import Dataset as Dataset
import utils
from torch.utils.data import DataLoader
from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
import torch
from tqdm import tqdm
from colors import rgb2ycbcr, ycbcr2rgb
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_path_restored', type=str, default='./Results/', help='Path to output directory')
    parser.add_argument('--in_path_compressed_eval', type=str, default='/home/claudiorota/hdd/Datasets/MFQEv2_Dataset/MFQEv2_RGB/test_18/HM16.5_LDP/QP37/', help='Path to compressed sequences for evaluation')
    parser.add_argument('--in_path_gt_eval', type=str, default='/home/claudiorota/hdd/Datasets/MFQEv2_Dataset/MFQEv2_RGB/test_18/raw/', help='Path to reference sequences for evaluation')
    parser.add_argument('--excluded_path_txt', type=str, default='/home/claudiorota/hdd/Datasets/MFQEv2_Dataset/MFQEv2_RGB/test_18/excluded.txt', help='Txt file contaning the name of the sequences to exclude during evaluation')
    parser.add_argument('--pretrained_path', type=str, default='/home/claudiorota/CAR/STDF/models_MFQEv2/DA_ELIMINARE_4LEVELS_SMALL_SMALL/', help='Path to pre-trained models to use for evaluation')
    parser.add_argument('--save', default=False, action='store_true')
    parser.add_argument('--model_name', default='XS', choices=['XS', 'S', 'M', 'L', 'XL'], help='Model name (must be XS, S, M, L, XL)')
    parser.add_argument('--lap_levels', default=4, type=int, help='Number of levels in the Laplacian pyramid')
    parser.add_argument('--window_size', default=7, type=int, help='Temporal neighbourhood size')
    parser.add_argument('--std', type=float, default=1, help='Standard deviation for Gaussian blur in Laplacian decomposition')

    args = parser.parse_args()
    print('> Running with the following arguments:')
    for arg, value in vars(args).items():
        print(f'    {arg}: {value}')


    # dataset paths
    path_compressed_test, path_gt_test = args.in_path_compressed_eval, args.in_path_gt_eval
    excluded_path = args.excluded_path_txt

    # parameters
    out_path = args.out_path_restored
    pretrained_path = args.pretrained_path

    temporal_neighborhood = args.window_size
    save_data = args.save

    # define model and optimizer
    device = torch.device('cuda')
    model = utils.create_model(args.model_name, args.lap_levels, args.std)

    if pretrained_path != '':
        print(f'> Loading pretrained model from {pretrained_path}')
        model.load_state_dict(torch.load(pretrained_path)['model_state_dict'])
        print('> Loading complete.')

    model = model.to(device)
    # load testset
    testset = Dataset(path_compressed_test, path_gt_test, temp_size=temporal_neighborhood, is_test=True,
                      max_video_number=1e4, max_seq_length=1e4, excluded_path=excluded_path,
                      return_name=True, is_cropped=False)
    testset_loader = DataLoader(testset, num_workers=1)

    psnr = PeakSignalNoiseRatio(data_range=1).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1).to(device)

    model.eval()
    delta_psnr_y = 0
    delta_ssim_y = 0
    psnr_y = 0
    ssim_y = 0
    print('> Evaluation.')
    pbar = tqdm(total=len(testset_loader), ncols=100)
    with torch.no_grad():
        for batch_n, data in enumerate(testset_loader, 1):
            torch.cuda.empty_cache()
            compressed, gt, name = data
            compressed, gt = compressed.to(device), gt.to(device)

            compressed_ycbcr = torch.cat([rgb2ycbcr(compressed[:, i:i + 3]) for i in range(0, compressed.shape[1], 3)],
                                         dim=1)
            compressed_y = torch.cat([compressed_ycbcr[:, i:i + 1] for i in range(0, compressed_ycbcr.shape[1], 3)],
                                     dim=1)
            gt_y = rgb2ycbcr(gt)[:, 0:1]
            if compressed_y.shape[3] > 1920:
                compressed_y_list = []
                compressed_y_list.append(compressed_y[:, :, 0:1600//2, 0:2560//2])
                compressed_y_list.append(compressed_y[:, :, 0:1600//2, 2560//2:])
                compressed_y_list.append(compressed_y[:, :, 1600//2:, 0:2560//2])
                compressed_y_list.append(compressed_y[:, :, 1600//2:, 2560//2:])
                gt_y_list = []
                gt_y_list.append(gt_y[:, :, 0:1600//2, 0:2560//2])
                gt_y_list.append(gt_y[:, :, 0:1600//2, 2560//2:])
                gt_y_list.append(gt_y[:, :, 1600//2:, 0:2560//2])
                gt_y_list.append(gt_y[:, :, 1600//2:, 2560//2:])
                restored_y_list = []
                for i in range(len(compressed_y_list)):
                    restored_y, _, _ = model(compressed_y_list[i].contiguous())
                    # restored_y = model(compressed_y_list[i].contiguous())
                    restored_y_list.append(restored_y)
                restored_y = torch.zeros(gt_y.shape).to(device)
                restored_y[:, :, 0:1600//2, 0:2560//2] = restored_y_list[0]
                restored_y[:, :, 0:1600//2, 2560//2:] = restored_y_list[1]
                restored_y[:, :, 1600//2:, 0:2560//2] = restored_y_list[2]
                restored_y[:, :, 1600//2:, 2560//2:] = restored_y_list[3]
            else:
                restored_y, _, _ = model(compressed_y)
                # restored_y = model(compressed_y)
            compressed_y = compressed_y[:, temporal_neighborhood // 2:temporal_neighborhood // 2 + 1]
            compressed_ycbcr = compressed_ycbcr[:, (temporal_neighborhood // 2) * 3: (temporal_neighborhood // 2 + 1) * 3]
            restored_ycbcr = torch.cat((restored_y, compressed_ycbcr[:, 1:]), dim=1)
            restored = ycbcr2rgb(restored_ycbcr)

            psnr_restored_y = psnr(restored_y, gt_y)
            psnr_compressed_y = psnr(compressed_y, gt_y)
            ssim_restored_y = ssim(restored_y, gt_y)
            ssim_compressed_y = ssim(compressed_y, gt_y)
            delta_ssim_y += (ssim_restored_y - ssim_compressed_y)
            delta_psnr_y += (psnr_restored_y - psnr_compressed_y)
            psnr_y += psnr_restored_y
            ssim_y += ssim_restored_y

            pbar.set_description(f'dPSNR_Y: {psnr_restored_y - psnr_compressed_y:.3f}, PSNR_Y: {psnr_restored_y:.3f}, dSSIM_Y: {ssim_restored_y - ssim_compressed_y:.3f}, SSIM_Y: {ssim_restored_y:.3f}')
            if save_data:
                utils.save_data(restored, f'{out_path}/{name[0]}')
            line = f'{name[0]} {(psnr_restored_y - psnr_compressed_y).item()} {(ssim_restored_y - ssim_compressed_y).item() * 1e2}'
            pbar.update()
    pbar.close()
    delta_psnr_y /= len(testset_loader)
    delta_ssim_y /= len(testset_loader)
    psnr_y /= len(testset_loader)
    ssim_y /= len(testset_loader)
    print(f'> Evaluation finished with dPSNR_y: {delta_psnr_y:.3f}, PSNR_Y: {psnr_y:.3f}, dSSIM_Y: {delta_ssim_y:.3f}, SSIM_Y: {ssim_y:.3f}')

if __name__ == '__main__':
    main()