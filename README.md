# Efficient Global-Local Memory for Real-time Instrument Segmentation of Robotic Surgical Video
We propose, on the one hand, an efficient local memory by taking the complementary advantages of convolutional LSTM and non-local mechanisms towards the relating reception field. On the other hand, we develop an active global memory to gather the global semantic correlation in long temporal range to current one, in which we gather the most informative frames derived from model uncertainty and frame similarity. 

This paper has been accepted by [MICCAI](https://link.springer.com/chapter/10.1007/978-3-030-87202-1_33).
Get the full paper on [Arxiv](https://arxiv.org/abs/2109.13593).

![bat](./framework.jpg)
Fig. 1. Structure of DMNet.

## Message
We have updated the codes of Efficient LA and GA. As the active selection is used only at inferrence and written in jupyter, we will update this part later. -- by 10/12


## Code List

- [x] Pre-processing
- [x] Training Codes
- [ ] Network

For more details or any questions, please feel easy to contact us by email ^\_^

## Usage


## Citation
If you find DMNet useful in your research, please consider citing:

```
@inproceedings{wang2021efficient,
  title={Efficient Global-Local Memory for Real-Time Instrument Segmentation of Robotic Surgical Video},
  author={Wang, Jiacheng and Jin, Yueming and Wang, Liansheng and Cai, Shuntian and Heng, Pheng-Ann and Qin, Jing},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={341--351},
  year={2021},
  organization={Springer}
}
```

