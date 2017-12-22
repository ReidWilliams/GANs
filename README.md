# It's the GANs
A highly commented Tensorflow implementation of DCGAN and WGAN for images.

This repo builds on [DCGAN](https://github.com/carpedm20/DCGAN-tensorflow) but adds lots of comments. If you're learning Tensorflow, deep neural networks, or GANs, it may be a good learning resource. If you just want to apply GANs asap or you really want to experiment with different architectures or loss functions, you may want to check out the [TFGAN](https://research.googleblog.com/2017/12/tfgan-lightweight-library-for.html) library.

## Getting Started

You'll need Python 3, Tensorflow, NumPy, Pillow, and SciPy. You'll also need a training dataset. You can try starting with the CelebA celebrity faces [dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

All images in the training set need to be the same size and both width and height dimensions need to be a multiple of 32. Rather than doing cropping/resizing of the dataset in the model, I use [ImageMagick](https://www.imagemagick.org). I have an ImageMagick cheat sheet [here](https://gist.github.com/ReidWilliams/53845baae2d2d2cc5fe49ca2d7c90b8a#file-imagemagick-sh).

Once you have everything installed and a dataset:
```
$ python main.py --datadir /path/to/data
```
## Details

The model can handle various training image sizes as long as both width and height are divisible by 32. It will automagically figure out image size when it loads the training images. As image sizes get larger, the model does not add additional convolutional layers, it just increases the number of units in the generator and discriminator fully connected layers. This may limit how large you can go. I've trained models on 576x256 pixel images on a 12GB GPU card (with a batch size of 16).

Three output directories are created for you: `logs` for logfiles for Tensorboard, `output` for samples of images from the generator, and `checkpoints` for Tensorflow session checkpoint files. If you kill `main.py` and restart, it will pick up where it left off using the checkpoint files.

If you want to use the Wasserstein GAN (WGAN) loss function instead of the default DCGAN loss:
```
$ python main.py --datadir /path/to/data --wgan
```
There are more command line options. Check out the source for `main.py` for details.

![made with a GAN](https://raw.githubusercontent.com/ReidWilliams/GANs/cleanup/mulletguy.png)
