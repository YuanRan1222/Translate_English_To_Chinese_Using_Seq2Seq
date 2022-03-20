# A simplified implementation of Seq2Seq for machine translation from English to Chinese

## Requirements

- 64-bit Python 3.7 installation.
- Tensorflow 2.2.0.
- No less than 16G RAM.
- One or more high-end NVIDIA GPUs is highly recommended to accelerate training process.

## Notice
Due to the hardware limitation of my laptop, the network model (seq2se1.h5 in the "model" folder) was trained on a subset of original corpus, i.e., num_samples=1000, in order to avoid memory overflow. If your hardware configuration is high enough, you can choose to set num_samples=None to use the whole corpus for training.
