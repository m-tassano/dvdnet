# DVDnet
A State-of-the-art, simple and fast network for Deep Video Denoising

## Video examples
 You can download several denoised sequences with our algorithm and other methods [here](https://www.dropbox.com/sh/gccey7wuxiqr104/AAC_v6kb3fMYxMHBc6wcqu17a?dl=0 "DVDnet denoised sequences")
 
 ## Comparison of PSNRs
Two different testsets were used for benchmarking our method: the DAVIS-test testset, and Set8, which is composed of 4 color sequences from the [Derf’s Test Media collection](https://media.xiph.org/video/derf) and 4 color sequences captured with a GoPro camera. The DAVIS set contains 30 color sequences of resolution 854 x 480. The sequences of Set8 have been downscaled to a resolution of 960 x 540. In all cases, sequences were limited to a maximum of 85 frames. We used the DeepFlow algorithm to compute flow maps for DVDnet and VNLB.

 ### PSNRs Set8 testset
| Noise std dev | DVDNet | VNLB | V-BM4D | NeatVideo |
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

 ## Code
Coming soon


The sequences are Copyright GoPro 2018
