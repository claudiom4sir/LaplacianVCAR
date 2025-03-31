# Scalable Residual Laplacian Network for HEVC-compressed Video Restoration (TOMM 2025)

[Claudio Rota](https://scholar.google.com/citations?user=HwPPoh4AAAAJ&hl=en), [Marco Buzzelli](https://scholar.google.com/citations?hl=en&user=kSFvKBoAAAAJ), [Simone Bianco](https://scholar.google.com/citations?hl=en&user=P08LSD0AAAAJ), and [Raimondo Schettini](https://scholar.google.com/citations?hl=en&user=ue60cV0AAAAJ)

[[Paper](https://dl.acm.org/doi/abs/10.1145/3727147)]

## Abstract
We present a novel Convolutional Neural Network that exploits the Laplacian decomposition technique, which is typically used in traditional image processing, to restore videos compressed with the High-Efficiency Video Coding (HEVC) algorithm. The proposed method decomposes the compressed frames into multi-scale frequency bands using the Laplacian decomposition, it restores each band using the ad-hoc designed Multi-frame Residual Laplacian Network (MRLN), and finally recomposes the restored bands to obtain the restored frames. By leveraging the multi-scale frequency representation of compressed frames provided by the Laplacian decomposition, MRLN can effectively reduce the compression artifacts and restore the image details with a reduced computational cost. In addition, our method can be easily instantiated in various versions to control the trade-off between efficiency and effectiveness, representing a versatile solution for scenarios with constrained computational resources. Experimental results on the MFQEv2 benchmark dataset show that our method achieves state-of-the-art performance in HEVC-compressed video restoration with a lower model complexity and shorter runtime with respect to existing methods.

## Overview
![image](https://github.com/user-attachments/assets/242e296a-ffcb-433f-8188-55f132325f5f)

## Usage

### Installation
The code is based on Python 3.7.13 and CUDA 11.7. See `requirements.txt` for the dependencies.

#### Conda setup
```
conda create -n laphevc python=3.7.13 -y
git clone https://github.com/claudiom4sir/LaplacianVCAR.git
cd LaplacianVCAR
conda activate laphevc
pip install -r requirements.txt
```
#### DCNv2 installation
```
cd ops/dcn/
bash build.sh
```
[Optional] Once installed, check if the installation was successful by executing this code:
```
python simple_check.py
```

### Dataset
1. Download the MFQEv2 dataset from [here](https://github.com/ryanxingql/mfqev2.0/wiki/MFQEv2-Dataset). Follow the page instructions. 

2. Decompose the sequences into RGB frames. `See from_yuv_to_png.py` to convert `.yuv` videos into `.png` frames.

3. As loading some frames may take a long time, create a cropped version of the dataset. Reading crops instead of full frames reduces dataloading time.
Generate the cropped version using `generate_crops.py` as follows:
```
python generate_crops.py --compressed_dir SET_YOUR_INPUT_DIR --target_dir SET_YOUR_OUTPUT_DIR
```
For each frame, it creates a folder containing frame crops. It may take time and memory space, but it is essential to save time during training. 

### Pretrained models
Pretrained models (QP37) are available [here](https://www.dropbox.com/scl/fo/vfbirbg14osmck1wwga8f/AJbigRmFy8BFZrNnctr1t84?rlkey=uj2on47z3rbzf3lopxk6216ly&st=7qwfbt2b&dl=0).

### Train
Train a new model using the following code:
```
python train.py
```
Adjust some arguments before using the script, such as `--in_path_compressed_train` and `--in_path_gt_train`. 
Add `--h` for more details. 
If you use our cropped version of the dataset, add `--cropped` to the input arguments.

### Test
Test a pretrained model using the following code:
```
python test.py
```
As for training, adjust some arguments before using the script. For example, use `--pretrained_path` to
set the path to the pretrained model you want to test. Add `--h` for more details. 

## Citations
```
@article{10.1145/3727147,
  author = {Rota, Claudio and Buzzelli, Marco and Bianco, Simone and Schettini, Raimondo},
  title = {Scalable Residual Laplacian Network for HEVC-compressed Video Restoration},
  year = {2025},
  publisher = {Association for Computing Machinery},
  doi = {10.1145/3727147},
  note = {Just Accepted},
  journal = {ACM Trans. Multimedia Comput. Commun. Appl.}
}
```

## Acknowledgement
The code is based on [STDF-pytorch](https://github.com/ryanxingql/stdf-pytorch).

