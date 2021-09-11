[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvdnet-a-fast-network-for-deep-video/video-denoising-on-davis-sigma30)](https://paperswithcode.com/sota/video-denoising-on-davis-sigma30?p=dvdnet-a-fast-network-for-deep-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvdnet-a-fast-network-for-deep-video/video-denoising-on-davis-sigma40)](https://paperswithcode.com/sota/video-denoising-on-davis-sigma40?p=dvdnet-a-fast-network-for-deep-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvdnet-a-fast-network-for-deep-video/video-denoising-on-davis-sigma50)](https://paperswithcode.com/sota/video-denoising-on-davis-sigma50?p=dvdnet-a-fast-network-for-deep-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvdnet-a-fast-network-for-deep-video/video-denoising-on-set8-sigma20)](https://paperswithcode.com/sota/video-denoising-on-set8-sigma20?p=dvdnet-a-fast-network-for-deep-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvdnet-a-fast-network-for-deep-video/video-denoising-on-set8-sigma30)](https://paperswithcode.com/sota/video-denoising-on-set8-sigma30?p=dvdnet-a-fast-network-for-deep-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvdnet-a-fast-network-for-deep-video/video-denoising-on-set8-sigma40)](https://paperswithcode.com/sota/video-denoising-on-set8-sigma40?p=dvdnet-a-fast-network-for-deep-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvdnet-a-fast-network-for-deep-video/video-denoising-on-set8-sigma50)](https://paperswithcode.com/sota/video-denoising-on-set8-sigma50?p=dvdnet-a-fast-network-for-deep-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvdnet-a-fast-network-for-deep-video/video-denoising-on-davis-sigma10)](https://paperswithcode.com/sota/video-denoising-on-davis-sigma10?p=dvdnet-a-fast-network-for-deep-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvdnet-a-fast-network-for-deep-video/video-denoising-on-davis-sigma20)](https://paperswithcode.com/sota/video-denoising-on-davis-sigma20?p=dvdnet-a-fast-network-for-deep-video)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/dvdnet-a-fast-network-for-deep-video/video-denoising-on-set8-sigma10)](https://paperswithcode.com/sota/video-denoising-on-set8-sigma10?p=dvdnet-a-fast-network-for-deep-video)

# [DVDnet](https://hal.archives-ouvertes.fr/hal-02147604)
A state-of-the-art, simple and fast network for Deep Video Denoising

**NEW: a state-of-the-art algorithm for video denoising without motion compensation [FastDVDnet](https://github.com/m-tassano/fastdvdnet)**

## Overview

This source code provides a PyTorch implementation of DVDnet image denoising, as in 
Tassano, Matias and Delon, Julie and Veit, Thomas. "DVDnet: A Fast Network for Deep Video Denoising", IEEE ICIP 2019, arXiv preprint arXiv:1906.11890 (2019).

## Video examples
 You can download several denoised sequences with our algorithm and other methods [here](https://www.dropbox.com/sh/gccey7wuxiqr104/AAC_v6kb3fMYxMHBc6wcqu17a?dl=0 "DVDnet denoised sequences")
 
## User Guide

The code as is runs in Python +3.6 with the following dependencies:
### Dependencies
* [PyTorch v1.0.0](http://pytorch.org/)
* [scikit-image](http://scikit-image.org/)
* [numpy](https://www.numpy.org/)
* [OpenCV](https://pypi.org/project/opencv-python/)

## Usage

If you want to denoise an image sequence using the pretrained models
found under the *models* folder you can execute

```
python test_dvdnet.py \
        --test_path <path_to_input_sequence> \
	--save_path results \
        --noise_sigma 25 \
```

**NOTES**
* The image sequence should be stored under <path_to_input_sequence>
* Models have been trained for values of noise in [5, 55]
* run with *--no_gpu* to run on CPU instead of GPU
* run with *--save_noisy* to save noisy frames
* set *max_num_fr_per_seq* to set the max number of frames to load per sequence
* run with *--help* to see details on all input parameters

## Comparison of PSNRs
Two different testsets were used for benchmarking our method: the DAVIS-test testset, and Set8, which is composed of 4 color sequences from the [Derf’s Test Media collection](https://media.xiph.org/video/derf) and 4 color sequences captured with a GoPro camera. The DAVIS set contains 30 color sequences of resolution 854 x 480. The sequences of Set8 have been downscaled to a resolution of 960 x 540. In all cases, sequences were limited to a maximum of 85 frames. We used the DeepFlow algorithm to compute flow maps for DVDnet and VNLB. For Neat Video, the automatic noise profiling settings were used.

Note: values shown are the average for all sequences in the testset, the PNSR of a sequence is computed as the average of the PSNRs of each frame.

### PSNRs Set8 testset
| Noise std dev | DVDNet | VNLB [1] | V-BM4D [2] | Neat Video [3] |
|---|---|---|---|---|
| 10 | 36.08 | **37.26** | 36.05 | 35.67 | 
| 20 | 33.49 | **33.72** | 32.19 | 31.69 | 
| 30 | **31.79** | 31.74 | 30.00 | 28.84 | 
| 40 | **30.55** | 30.39 | 28.48 | 26.36 | 
| 50 | **29.56** | 29.24 | 27.33 | 25.46 | 

### PSNRs DAVIS testset
| Noise std dev | DVDNet | VNLB | V-BM4D |
|--|--|--|--|
| 10 | 38.13 | **38.85** | 37.58 |
| 20 | **35.70** | 35.68 | 33.88 |
| 30 | **34.08** | 33.73 | 31.65 |
| 40 | **32.86** | 32.32 | 30.05 |
| 50 | **31.85** | 31.13 | 28.80 |

## ABOUT

Copying and distribution of this file, with or without modification,
are permitted in any medium without royalty provided the copyright
notice and this notice are preserved. This file is offered as-is,
without any warranty.

* Author    : Matias Tassano `mtassano at gopro dot com`
* Copyright : (C) 2019 Matias Tassano
* Licence   : GPL v3+, see GPLv3.txt

The sequences are Copyright GoPro 2018

## References

[1] P. Arias and J.-M. Morel, “Video denoising via empirical Bayesian estimation of space-time patches,” Journal of Mathematical Imaging and Vision, vol. 60, no. 1, pp. 70—-93, 2018

[2] M. Maggioni, G. Boracchi, A. Foi, K. Egiazarian, “Video denoising, deblocking, and enhancement through separable 4-D nonlocal spatiotemporal transforms,” IEEE Trans. IP, vol. 21, no. 9, pp. 3952–3966, 2012.

[3] ABSoft, “Neat Video,” https://www.neatvideo.com, 1999–2019.
