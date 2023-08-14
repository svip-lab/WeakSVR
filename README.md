# WeakSVR: Weakly Supervised Video Representation Learning with Unaligned Text for Sequential Videos (CVPR2023)

Here is the official implementation for CVPR 2023 paper "Weakly Supervised Video Representation Learning with Unaligned Text for Sequential Videos".

## 🌱News
- 2023-06-01: The oral presentation of our work is available. [[Youtube](https://www.youtube.com/watch?v=AqozSRYP7Pc)] [[Bilibili](https://www.bilibili.com/video/BV1AW4y1R7um/)]
- 2023-03-29: We have updated the Chinese introduction of the paper. [[Zhihu](https://zhuanlan.zhihu.com/p/617926257)]
- 2023-03-24: The code has been released (need revisions).
- 2023-03-22: The preprint of the paper is available. [[Paper](https://arxiv.org/abs/2303.12370)]
- 2023-02-28: This paper has been accepted by **`CVPR 2023`**.

## Introduction
Sequential video understanding, as an emerging video understanding task, has driven lots of researchers’ attention because of its goal-oriented nature. This paper studies weakly supervised sequential video understanding where the accurate time-stamp level text-video alignment is not provided. We solve this task by borrowing ideas from CLIP. Specifically, we use a transformer to aggregate frame-level features for video representation and use a pre-trained text encoder to encode the texts corresponding to each action and the whole video, respectively. To model the correspondence between text and video, we propose a multiple granularity loss, where the video-paragraph contrastive loss enforces matching between the whole video and the complete script, and a fine-grained frame-sentence contrastive loss enforces the matching between each action and its description. As the frame-sentence correspondence is not available, we propose to use the fact that video actions happen sequentially in the temporal domain to generate pseudo frame-sentence correspondence and supervise the network training with the pseudo labels. Extensive experiments on video sequence verification and texttovideo matching show that our method outperforms baselines by a large margin, which validates the effectiveness of our proposed approach.
![](https://github.com/svip-lab/WeakSVR/blob/main/figs/sequence%20video.jpg)
## Video Presentation  
<center><a href="https://www.bilibili.com/video/BV1AW4y1R7um/" target="_blank" style="color: #990000"> Bilibili </a></center>       <br/> 
<center><a href="https://www.youtube.com/watch?v=AqozSRYP7Pc" target="_blank" style="color: #990000"> YouTube </a></center>  

## Usage  
Preparing

### Model Zoo
##### Weakly Supervised (w/o CLS)
|  Method   | CSV | Diving-SV | COIN-SV |
|  :---: | :-----: | :----:  | :----: | 
| Ours  | 80.81 [BaiduDisk(code:cvpr)](  https://pan.baidu.com/s/1x660E9b3_BvHidEG6wAbLQ?pwd=cvpr) | [BaiduDisk(code:cvpr)](https://pan.baidu.com/s/1zGixeyLq8HNaNCQptpwz3g?pwd=cvpr)  |  -  |

## Acknowledgement
Codebase from [SVIP](https://github.com/svip-lab/SVIP-Sequence-VerIfication-for-Procedures-in-Videos)

## Citation 
If you find the project or our method is useful, please consider citing the paper.  
```
@article{dong2023weakly,
  title={Weakly Supervised Video Representation Learning with Unaligned Text for Sequential Videos},
  author={Dong, Sixun and Hu, Huazhang and Lian, Dongze and Luo, Weixin and Qian, Yicheng and Gao, Shenghua},
  journal={arXiv preprint arXiv:2303.12370},
  year={2023}
}
```
```
@inproceedings{dong2023weakly,
  title={Weakly Supervised Video Representation Learning with Unaligned Text for Sequential Videos},
  author={Dong, Sixun and Hu, Huazhang and Lian, Dongze and Luo, Weixin and Qian, Yicheng and Gao, Shenghua},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2437--2447},
  year={2023}
}
```
