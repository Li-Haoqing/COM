# A Level Set Annotation Framework With Single-Point Supervision for Infrared Small Target Detection

This is the official repository of the paper 'A Level Set Annotation Framework With Single-Point Supervision for Infrared Small Target Detection'.

## Requirements

```
python 3.7
numpy
scipy
matplotlib
scikit-image
cv2
```

## Click on Mask

```python3 Main.py --BIAS 20 --RATIO 2 --timestep 5 --iter_inner 5 --alfa 1.5 --epsilon 1.5```

In default, the Pseudo Mask will be saved at ```'COM_results/```.


## Related Datasets

IRSTD-1k: https://github.com/RuiZhang97/ISNet

NUAA-SIRST: https://github.com/YimianDai/sirst


## Get Started


### Training with COM

python3 model_training/train.py --img_size 256 --batch_size 8 --epochs 300 --warm_up_epochs 10 --learning_rate 0.001 --dataset 'IRSTD-1k' --model 'ILNet'


### Valuating

python3 model_training/val.py --img_size 256 --dataset 'IRSTD-1k' --batch-size 1 --model 'ILNet' --checkpoint ' .pth' 


## Acknowledgement

Part of the code draws on the work of the following authors:

https://github.com/Ramesh-X/Level-Set

model:

    ILNet: https://github.com/Li-Haoqing/ILNet/
    
    ACM: https://github.com/YimianDai/sirst
    
    AGPC: https://github.com/Tianfang-Zhang/AGPCNet
    
    DNA-Net: https://github.com/YeRen123455/Infrared-Small-Target-Detection
    
    UIUNet: https://github.com/danfenghong/IEEE_TIP_UIU-Net


## Citation:

    @ARTICLE{10409613,
      author={Li, Haoqing and Yang, Jinfu and Xu, Yifei and Wang, Runshi},
      journal={IEEE Signal Processing Letters}, 
      title={A Level Set Annotation Framework With Single-Point Supervision for Infrared Small Target Detection}, 
      year={2024},
      volume={31},
      pages={451-455},
      doi={10.1109/LSP.2024.3356411}}
