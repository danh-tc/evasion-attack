# Random Parameter Pruning Attack(RaPA) Accepted by CVPR26
[arxiv](https://arxiv.org/abs/2504.18594)
## Env Install

- Create Conda Environment
```
conda create -n RaPA_ENV python=3.8 -y
```
- Install PyTorch, Torchaudio, and Torchvision

```
pip install torch==2.1.0 torchaudio==2.1.0 torchvision==0.16.0 -f https://download.pytorch.org/whl/cu121
```
- Install Requirements

```
pip install -r requirements.txt
```
- Install OpenMIM

```
pip install -U openmim && mim install "mmengine==0.10.5" && mim install "mmpretrain==1.2.0" 
```
- Install Local Code

```
cd RaPA_CODE 
pip install -v -e .
```

## Data download
Follow the download guidance the NIPS17 Adversarial Competition Dataset to get the dataset. 

The data should be organized as following:
```
data
├── ImageNetCompetition
│   ├── images
│   │   ├── xxx.png
│   │   ├── ....
│   │   └── xxx.png
│   └── images.csv
```


## Main result
You can run the following scripts to run experiments using CNN-based and Transformer-based surrogate models and get the main results of our method:
```
bash script/main_result/RaPA-CNN.sh
```
```
bash script/main_result/RaPA-Transformer.sh
```
