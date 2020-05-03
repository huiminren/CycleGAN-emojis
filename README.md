# University of Toronto CSC321 [Assignment 4](http://www.cs.toronto.edu/~rgrosse/courses/csc321_2018/assignments/a4-handout.pdf) -- CycleGAN

## Comparison of Loss Functions

* D_X_loss1 = ((D_X(images_X)-1)\**2).sum()/len(images_X)

<p align="center">
<img src="samples_cyclegan_div/sample-001000-X-Y.png">
<img src="samples_cyclegan_div/sample-040000-X-Y.png" >
</p>


* D_X_loss2 = torch.mean((D_X(images_X)-1)**2) 

* D_X_loss3 = criterion(D_X(images_X),real_labels_X)

## Acknowlegements

This code is inspired by [pytorch-cycleGAN](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

Thanks [Luois](https://github.com/lluo5779/CSC321/tree/master/4.%20GAN%20and%20CycleGan) and 
[Guanxiong](https://github.com/liuguanxiong/CSC321) to share their code.
