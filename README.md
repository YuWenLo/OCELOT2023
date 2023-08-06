# Enhancing Cell Detection via FC-HardNet and Tissue Segmentation: OCELOT 2023 Challenge Approach

> This GitHub is for **cell detection**.

## HarDNet Family
#### For Image Classification : [HarDNet](https://github.com/PingoLH/Pytorch-HarDNet) 78.0 top-1 acc. / 1029.76 Throughput on ImageNet-1K @224x224
#### For Object Detection : [CenterNet-HarDNet](https://github.com/PingoLH/CenterNet-HarDNet) 44.3 mAP / 60 FPS on COCO val @512x512
#### For Semantic Segmentation : [FC-HarDNet](https://github.com/PingoLH/FCHarDNet)  75.9% mIoU / 90.24 FPS on Cityscapes test @1024x2048
#### For Polyp Segmentation : [HarDNet-MSEG](https://github.com/james128333/HarDNet-MSEG) 90.4% mDice / 119 FPS on Kvasir-SEG @352x352


### Training

1. Download pretrain_weight: [hardnet_petite_base.pth](https://github.com/PingoLH/FCHarDNet/tree/master/weights) and place in the folder ``` /weights ``` 
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
  @misc{2022hardnetdfus,
  title = {HarDNet-DFUS: An Enhanced Harmonically-Connected Network for Diabetic Foot Ulcer Image Segmentation and Colonoscopy Polyp Segmentation},
  author = {Liao, Ting-Yu and Yang, Ching-Hui and Lo, Yu-Wen and Lai, Kuan-Ying and Shen, Po-Huai and Lin, Youn-Long},
  year = {2022},
  eprint={2209.07313},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
  }

  @inproceedings{chao2019hardnet,
  title={Hardnet: A low memory traffic network},
  author={Chao, Ping and Kao, Chao-Yang and Ruan, Yu-Shan and Huang, Chien-Hsiang and Lin, Youn-Long},
  booktitle={Proceedings of the IEEE International Conference on Computer Vision},
  pages={3552--3561},
  year={2019}
  }
```