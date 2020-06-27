# Blind Super-Resolution Kernel Estimation using an Internal-GAN
# "KernelGAN"
### Sefi Bell-Kligler, Assaf Shocher, Michal Irani 
*(Official implementation)*

Paper: https://arxiv.org/abs/1909.06581

Project page: http://www.wisdom.weizmann.ac.il/~vision/kernelgan/  

**Accepted NeurIPS 2019 (oral)**


## Usage:

### Quick usage on your data:  
To run KernelGAN on all images in <input_image_path>:

``` python train.py --input-dir <input_image_path> ```


This will produce kernel estimations in the results folder

### Extra configurations:  
```--X4``` : Estimate the X4 kernel

```--SR``` : Perform ZSSR using the estimated kernel

```--real``` : Real-image configuration (effects only the ZSSR)

```--output-dir``` : Output folder for the images (default is results)


### Data:
Download the DIV2KRK dataset: [dropbox](https://www.dropbox.com/s/gkx5abm90ij74nc/DIV2KRK_public.zip?dl=0)

Reproduction code for your own Blind-SR dataset: [github](https://github.com/assafshocher/BlindSR_dataset_generator)
