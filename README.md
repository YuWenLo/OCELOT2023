# Enhancing Cell Detection via FC-HarDNet and Tissue Segmentation: OCELOT 2023 Challenge Approach

> This GitHub is for **cell detection**.

## HarDNet Family
#### For Image Classification : [HarDNet](https://github.com/PingoLH/Pytorch-HarDNet) 78.0 top-1 acc. / 1029.76 Throughput on ImageNet-1K @224x224
#### For Object Detection : [CenterNet-HarDNet](https://github.com/PingoLH/CenterNet-HarDNet) 44.3 mAP / 60 FPS on COCO val @512x512
#### For Semantic Segmentation : [FC-HarDNet](https://github.com/PingoLH/FCHarDNet)  75.9% mIoU / 90.24 FPS on Cityscapes test @1024x2048
#### For Polyp Segmentation : [HarDNet-MSEG](https://github.com/james128333/HarDNet-MSEG) 90.4% mDice / 119 FPS on Kvasir-SEG @352x352


## Installation & Usage
### Environment setting (Prerequisites)
```
conda create -n fchardnet python=3.6
conda activate fchardnet
pip install -r requirements.txt
```

### FC-HarDNet Setting
```
cd CatConv2d/
python setup.py install
```

### Dataset preparation
1. Convert cell/tissue data to training format.
   - cell:
        ```
        python convert_masks.py --mode cell --cell_radius 5 --data_path path/to/cell_csv_folder/ --save_path path/to/cell_images/masks/
        ```
    - tissue:
        ```
        python convert_masks.py --mode tissue --data_path path/to/tissue_png_mask_folder/ --save_path path/to/tissue_images/masks/
        ```
2. Place the data (or create symlinks) to make the data folder like:
      ~~~
      ${FC_HarDNet_ROOT}
      |-- data
      `-- |-- cell
          `-- |-- images
           -- |-- masks
      `-- |-- tissue
          `-- |-- images
           -- |-- masks
    
      ~~~  
### Training

1. Download pretrain_weight: [hardnet_petite_base.pth](https://github.com/PingoLH/FCHarDNet/tree/master/weights) and change the --weight in Optional Args
2. Run:
    - cell:
    ```
    python train.py --mode cell --augmentation --data_path /work/wagw1014/OCELOT/ --batchsize 16 --seed 42 --dataratio 0.8 --modelname fchardnet --weight /home/wagw1014/FCHarDNet/weights/hardnet_petite_base.pth --trainsize 1024 --lr 0.0001 --epoch 300 --name all5fold_cell_1c_r5 --loss structure_loss --cell_size 5 --kfold 5

    Optional Args:
    --augmentation Activating data audmentation during training
    --kfold        Specifying the number of K-Fold Cross-Validation
    --k            Training the specific fold of K-Fold Cross-Validation
    --dataratio    Specifying the ratio of data for training
    --seed         Reproducing the result of data spliting in dataloader
    --data_path    Path to training data
    ```
    - tissue:
    ```
    python train.py --mod tissue --augmentation --data_path /work/wagw1014/OCELOT/tissue/ --batchsize 16 --seed 42 --dataratio 0.8 --modelname fchardnet --weight /home/wagw1014/FCHarDNet/weights/hardnet_petite_base.pth --trainsize 1024 --lr 0.0001 --epoch 1000 --name all5fold_tissue_strloss_1000e --loss structure_loss --kfold 5

    Optional Args:
    --augmentation Activating data audmentation during training
    --kfold        Specifying the number of K-Fold Cross-Validation
    --k            Training the specific fold of K-Fold Cross-Validation
    --dataratio    Specifying the ratio of data for training
    --seed         Reproducing the result of data spliting in dataloader
    --data_path    Path to training data
    ```

### Inference
- Input/output formats can be found in [OCELOT 2023: Detecting Cells from Cell-Tissue Interactions](https://github.com/lunit-io/ocelot23algo/tree/main).
- Do a simple test by running the following command. 
    Run:
    ```
    python process.py
    ```

## Acknowledgement
- This research is supported in part by a grant from the **Ministry of Science and Technology (MOST) of Taiwan**.   
We thank **National Center for High-performance Computing (NCHC)** for providing computational and storage resources.        

## Citation
If you find this project useful for your research, please use the following BibTeX entry.
```
  @inproceedings{chao2019hardnet,
  title={Hardnet: A low memory traffic network},
  author={Chao, Ping and Kao, Chao-Yang and Ruan, Yu-Shan and Huang, Chien-Hsiang and Lin, Youn-Long},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={3552--3561},
  year={2019}
  }
```