# Click on Mask: A Labor-efficient Annotation Framework with Level Set for Infrared Small Target Detection

This is the official repository of the paper 'Click on Mask: A Labor-efficient Annotation Framework with Level Set for Infrared Small Target Detection'.

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


## Training with COM

python3 model_training/train.py --img_size 256 --batch_size 8 --epochs 300 --warm_up_epochs 10 --learning_rate 0.001 --dataset 'IRSTD-1k' --model 'ILNet'


## Valuating

python3 model_training/val.py --img_size 256 --dataset 'IRSTD-1k' --batch-size 1 --model 'ILNet' --checkpoint ' .pth' 


## Thanks:

Part of the code draws on the work of the following authors:
https://github.com/Ramesh-X/Level-Set

model:
    ILNet: https://github.com/Li-Haoqing/ILNet/
    ACM: https://github.com/YimianDai/sirst
    AGPC: https://github.com/Tianfang-Zhang/AGPCNet
    DNA-Net: https://github.com/YeRen123455/Infrared-Small-Target-Detection
    UIUNet: https://github.com/danfenghong/IEEE_TIP_UIU-Net
