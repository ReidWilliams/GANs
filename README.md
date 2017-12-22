# It's the GANs
A highly commented Tensorflow implementation of DCGAN and WGAN for images.

This repo builds on [DCGAN](https://github.com/carpedm20/DCGAN-tensorflow) but adds lots of comments. If you're learning Tensorflow, deep neural networks, or GANs, it may be a good learning resource. If you just want to apply GANs asap or you really want to experiment with different architectures or loss functions, you may want to check out [TFGAN](https://research.googleblog.com/2017/12/tfgan-lightweight-library-for.html).

## Getting Started

You'll need Python 3, Tensorflow, NumPy, Pillow, and SciPy. You'll also need a training dataset. You can try starting with the CelebA celebrity faces [dataset](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).

All images in the training set need to be the same size and both width and height dimensions need to be a multiple of 32. Rather than build cropping/resizing of the dataset into the model, I use [ImageMagick](https://www.imagemagick.org). I have an ImageMagick cheat sheet [here](https://gist.github.com/ReidWilliams/53845baae2d2d2cc5fe49ca2d7c90b8a#file-imagemagick-sh).

Once you have everything installed and a dataset:
```
> python main.py --datadir /data
```





Keras / Tensorflow implementation of Larsen, https://arxiv.org/abs/1512.09300

- Try reduce mean instead of reduce sum cost
- learning rates


