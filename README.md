# transformer-vit-seg

# 在线运行
https://www.kaggle.com/code/weililong/transformer-vit-seg-v3/notebook

# 说明
研究了几天，现在能跑通，效果不太理想，紧做为transformer图像分割实验，应该是解码部分代的问题，如果大老能优化下，那就牛了， 代码一简单为主。

# 参考配置

vit_large_patch16_384: (占用显存：12GB / Max 15.9GB , bs=8 ，  要达到0.4B这个才有效果)
    image_size: 384
    patch_size: 16
    d_model: 1024
    n_heads: 16
    n_layers: 24
    normalization: vit


vit_large_patch32_384: (这个更大，没试过)
    image_size: 384
    patch_size: 32
    d_model: 1024
    n_heads: 16
    n_layers: 24
    normalization: vit


