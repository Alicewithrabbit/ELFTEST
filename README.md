# ELFATT

This repo is the official PyTorch implementation of **ELFATT**.

+ [ELFATT: Efficient Linear Fast Attention for Vision Transformers](https://arxiv.org/abs/2501.06098v2)

## Introduction

ELFATT (**E**fficient **L**inear **F**ast **ATT**ention) is a new attention acceleration method for vision transformers. It uses a hybrid-head architecture combining sparse attention with global linear attention to significantly improve speed without losing performance.

**ELFATT formula for approximating a single block of a single-head vanilla attention mechanism**
```math
{\rm softmax}\left(\textit{\textbf{Q}}\textit{\textbf{K}}^{\top}\right)\textit{\textbf{V}} + L\left(\textit{\textbf{V}}\right) \approx \left\lbrack{\rm softmax}\left(\bar{\textit{\textbf{Q}}}\right){\rm softmax}\left(\bar{\textit{\textbf{K}}}\right)^{\top}\bar{\textit{\textbf{V}}}+L\left(\bar{\textit{\textbf{V}}}\right), g\left({\rm softmax}\left(f(\tilde{\textit{\textbf{Q}}}) f(\tilde{\textit{\textbf{K}}})^{\top}\right)f(\tilde{\textit{\textbf{V}}})+L\left(f(\tilde{\textit{\textbf{V}}})\right)\right)\right\rbrack,
```

**ELFATT formula for approximating a single block of a double-head vanilla attention mechanism**
```math
\left\lbrack{\rm softmax}\left(\bar{\textit{\textbf{Q}}}\bar{\textit{\textbf{K}}}^{\top}\right)\bar{\textit{\textbf{V}}}+L\left(\bar{\textit{\textbf{V}}}\right), {\rm softmax}\left(\tilde{\textit{\textbf{Q}}}\tilde{\textit{\textbf{K}}}^{\top}\right)\tilde{\textit{\textbf{V}}}+L\left(\tilde{\textit{\textbf{V}}}\right)\right\rbrack\approx \left\lbrack{\rm softmax}\left(\bar{\textit{\textbf{Q}}}\right){\rm softmax}\left(\bar{\textit{\textbf{K}}}\right)^{\top}\bar{\textit{\textbf{V}}}+L\left(\bar{\textit{\textbf{V}}}\right), g\left({\rm softmax}\left(f(\tilde{\textit{\textbf{Q}}}) f(\tilde{\textit{\textbf{K}}})^{\top}\right)f(\tilde{\textit{\textbf{V}}})+L\left(f(\tilde{\textit{\textbf{V}}})\right)\right)\right\rbrack,
```

where $\textit{\textbf{Q}} = \lbrack\bar{\textit{\textbf{Q}}},\tilde{\textit{\textbf{Q}}}\rbrack \in \mathbb{R}^{m \times c}$, $\textit{\textbf{K}} = \lbrack\bar{\textit{\textbf{K}}},\tilde{\textit{\textbf{K}}}\rbrack \in \mathbb{R}^{m \times c}$, $\textit{\textbf{V}} = \lbrack\bar{\textit{\textbf{V}}},\tilde{\textit{\textbf{V}}}\rbrack \in \mathbb{R}^{m \times c}$, $\bar{\textit{\textbf{Q}}}\in\mathbb{R}^{m \times c_1}$, $\tilde{\textit{\textbf{Q}}}\in\mathbb{R}^{m \times c_2}$, $\bar{\textit{\textbf{K}}}\in\mathbb{R}^{m \times c_1}$, $\tilde{\textit{\textbf{K}}}\in\mathbb{R}^{m \times c_2}$, $\bar{\textit{\textbf{V}}}\in\mathbb{R}^{m \times c_1}$, $\tilde{\textit{\textbf{V}}}\in\mathbb{R}^{m \times c_2}$, $c=c_1+c_2$, $f\left(\cdot\right)$ is a blockify function to separate a matrix with a size of $m\times c_2$ into $b$ blocks (each block has a size of $(m/b)\times c_2$), $g\left(\cdot\right)$ is an unblockify function to unblock $b$ blocks to a single matrix with a size of $m\times c_2$, and $L(\cdot)$ denotes a depthwise convolution operation with a kernel size of 3. 

ELFATT offers 4-7x speedups over the vanilla softmax-based attention mechanism in high-resolution vision tasks without losing performance. ELFATT is FlashAttention friendly. Using FlashAttention-2 acceleration, ELFATT still offers 2-3x speedups over the vanilla softmax-based attention mechanism on high-resolution vision tasks without losing performance. Even on edge GPUs, ELFATT still offers 1.6x to 2.0x speedups compared to state-of-the-art attention mechanisms in various power modes from 5W to 60W. Furthermore, ELFATT can be used to enhance and accelerate diffusion tasks directly without training.




## Main Results on ImageNet
| Method          | Pretrain     | Resolution | Acc@1 | FPS (nFA/FA)     |#      | FLOPs (nFA/FA) | 1K model |
|:---:            | :---:        |  :---:     | :---: |            :---: | :---: |          :---: |    :---: |
| CSWin-T-ELFATT  | ImageNet-1K  | 224<sup>2</sup>      | 83.1  | 2603/2856 imgs/s |  20M  |   4.44G/4.13G  |     [model](https://github.com/Alicewithrabbit/ELFATT/releases/download/v0.1.0/cswin_elfatt_tiny_224.pth)     | 
| CSWin-B-ELFATT  | ImageNet-1K  | 224<sup>2</sup>      | 84.7  | 1000/1187 imgs/s |  73M  |  15.47G/14.46G |      -   |
| CSWin-B-ELFATT  | ImageNet-1K  | 384<sup>2</sup>      | 85.8  | 272/355 imgs/s   |  73M  |  51.21G/42.48G |      -   |
| Swin-T-ELFATT   | ImageNet-1K  | 224<sup>2</sup>      | 82.7  | 2884/3159 imgs/s |  30M  |   4.99G/4.67G  |      -   | 
| Swin-B-ELFATT   | ImageNet-1K  | 224<sup>2</sup>      | 84.5  | 1314/1497 imgs/s |  91M  |  16.46G/15.68G |      -   |
| Swin-B-ELFATT   | ImageNet-1K  | 384<sup>2</sup>      | 85.5  | 372/457 img/s    |  91M  |  52.79G/46.08G |      -   |

Note: CSWin-T represents CSWin-T-24182, and CSWin-B represents CSWin-B-36292. Inference throughput (FPS) is obtained using a batch size of 512 for tiny models and 256/32 for base models with a resolution of 224<sup>2</sup>/384<sup>2</sup> using mixed precision on a single NVIDIA H20 (96 GB) GPU. "nFA/FA" denotes without/with using FlashAttention-2.

## Main Results on ADE20K
| Backbone       |   Method    |  Pretrain   |      Crop Size  | LR Schedule |            mAcc |            mIoU | FPS (nFA/FA)     |#      |   FLOPs (nFA/FA) |
| :---:          |    :---:    |    :---:    |           :---: |       :---: |           :---: |           :---: |            :---: | :---: |            :---: |  
| CSWin-T-ELFATT |   UperNet   | ImageNet-1K | 512<sup>2</sup> |        160k |            61.2 |            49.6 |     28/32 imgs/s |  50M  | 1014.26G/929.53G |
| Swin-T-ELFATT  |   UperNet   | ImageNet-1K | 512<sup>2</sup> |        160k |            59.3 |            47.7 |     34/38 imgs/s |  62M  |  991.27G/943.94G |

Note: "mAcc" denotes mean class accuracy and "mIoU" denotes mean intersection over union. FLOPs are calculated using an input size of 512x2048. "160k" denotes the 160k fine-tuning iterations. Inference throughput is obtained using a batch size of 1 with mixed precision on a single NVIDIA H20 (96 GB) GPU.

## Main Results on MS COCO 2017
| Backbone       |   Method    |  Pretrain   | LR Schedule |  AP<sup>b</sup> |  AP<sup>m</sup> | FPS (nFA/FA)     |#      |   FLOPs (nFA/FA) |
| :---:          |    :---:    |    :---:    |       :---: |           :---: |           :---: |            :---: | :---: |            :---: | 
| CSWin-T-ELFATT |  Mask-RCNN  | ImageNet-1K |          1x |            47.0 |            42.6 |     30/33 imgs/s |  40M  |  334.86G/254.40G |
| CSWin-T-ELFATT |  Mask-RCNN  | ImageNet-1K |        3xMS |            49.4 |            44.0 |     30/33 imgs/s |  40M  |  334.86G/254.40G |
| Swin-T-ELFATT  |  Mask-RCNN  | ImageNet-1K |          1x |            46.1 |            42.1 |     39/45 imgs/s |  50M  |  311.39G/266.43G |
| Swin-T-ELFATT  |  Mask-RCNN  | ImageNet-1K |        3xMS |            48.5 |            43.6 |     39/45 imgs/s |  50M  |  311.39G/266.43G |

Note: FLOPs are calculated using an input size of 1280x800. "1x" denotes the fine-tuning training schedule with 12 epochs and "3xMS" represents fine-tuning using the multiscale training schedule with 36 epochs. AP<sup>b</sup> denotes box average precision and AP<sup>m</sup> denotes mask average precision. Inference throughput is obtained using a batch size of 1 with mixed precision on a single NVIDIA H20 (96 GB) GPU.

## Citation

If you find this repo helpful, please consider citing us.

```latex
@misc{wu2025elfattefficientlinearfast,
      title={ELFATT: Efficient Linear Fast Attention for Vision Transformers}, 
      author={Chong Wu and Maolin Che and Renjie Xu and Zhuoheng Ran and Hong Yan},
      year={2025},
      eprint={2501.06098v2},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2501.06098v2}, 
}
```

## Contact

Please feel free to contact Chong Wu: [chong@innocimda.com](mailto:chong@innocimda.com) for help.
