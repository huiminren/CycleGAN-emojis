# University of Toronto CSC321 [Assignment 4](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf) -- CycleGAN

## Comparison of Loss Functions

* D_X_loss1 = ((D_X(images_X)-1)\**2).sum()/len(images_X)

|      sample-001000-X-Y   |   sample-020000-X-Y       |   sample-040000-X-Y   |
|:------------------------:|:-------------------------:|:---------------------:|
|<img src="samples_cyclegan_div/sample-001000-X-Y.png" width="100%">|<img src="samples_cyclegan_div/sample-020000-X-Y.png" width="100%">|<img src="samples_cyclegan_div/sample-040000-X-Y.png" width="100%">|


* D_X_loss2 = torch.mean((D_X(images_X)-1)\**2) 

|      sample-001000-X-Y   |   sample-020000-X-Y       |   sample-040000-X-Y   |
|:------------------------:|:-------------------------:|:---------------------:|
|<img src="samples_cyclegan_torchmean/sample-001000-X-Y.png" width="100%">|<img src="samples_cyclegan_torchmean/sample-020000-X-Y.png" width="100%">|<img src="samples_cyclegan_torchmean/sample-040000-X-Y.png" width="100%">|

* D_X_loss3 = criterion(D_X(images_X),real_labels_X)

|      sample-001000-X-Y   |   sample-020000-X-Y       |   sample-040000-X-Y   |
|:------------------------:|:-------------------------:|:---------------------:|
|<img src="samples_cyclegan_criterion/sample-001000-X-Y.png" width="100%">|<img src="samples_cyclegan_criterion/sample-020000-X-Y.png" width="100%">|<img src="samples_cyclegan_criterion/sample-040000-X-Y.png" width="100%">|

* Loss plots

|      Loss1   |   Loss2       |   Loss3   |
|:------------:|:-------------:|:---------:|
|<img src="samples_cyclegan_div/Loss.png" width="100%">|<img src="samples_cyclegan_torchmean/Loss.png" width="100%">|<img src="samples_cyclegan_criterion/Loss.png" width="100%">|

## Acknowlegements
This code is inspired by [pytorch-cycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

Thanks [Luois](https://github.com/lluo5779/CSC321/tree/master/4.%20GAN%20and%20CycleGan) and 
[Guanxiong](https://github.com/liuguanxiong/CSC321) to share their code.
