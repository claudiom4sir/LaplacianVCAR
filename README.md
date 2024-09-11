## Dataset
1. Download the MFQEv2 dataset from [here](https://github.com/ryanxingql/mfqev2.0/wiki/MFQEv2-Dataset). Follow the page instructions. 

2. Decompose the sequences into RGB frames. `See from_yuv_to_png.py` to convert `.yuv` videos into `.png` frames.

3. As loading some frames may take a long time, create a cropped version of the dataset. Reading crops instead of full frames reduces dataloading time.
Generate the cropped version using `generate_crops.py` as follows:
```
python generate_crops.py --compressed_dir SET_YOUR_INPUT_DIR --target_dir SET_YOUR_OUTPUT_DIR
```
For each frame, it creates a folder containing frame crops. It may take time and memory space, but it is essential to save time during training. 

## Usage

### Installation and requirements
The code is based on Python 3.7.13 and CUDA 11.7. See `requirements.txt` for the dependencies.

#### Conda enviroment
```
conda create -n laphevc python=3.7.13 -y
git@github.com:claudiom4sir/LaplacianVCAR.git
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

### Pretrained models
All the pretrained models are available at [TODO]()

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

## Acknowledgement
The code is based on [STDF-pytorch](https://github.com/ryanxingql/stdf-pytorch).

