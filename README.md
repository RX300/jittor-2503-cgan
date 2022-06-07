# jittor-2503-cgan
| 第二届计图挑战赛开源模板

# Jittor CGAN手写数字生成 baseline


## 简介

本项目包含了第二届计图挑战赛计图 - 手写数字生成的代码实现。本项目的特点是：在数字图片数据集 MNIST 上训练 Conditional GAN模型，通过输入一个随机向量 z 和额外的辅助信息 y(如类别标签)，生成特定数字的图像
## 安装 

本项目可在CPU上运行，训练时间约为 3 小时。

#### 运行环境
- Windows
- python >= 3.7
- jittor >= 1.3.0

#### 安装依赖
执行以下命令安装 python 依赖
```
pip install -r requirements.txt
```

#### 预训练模型
预训练模型模型下载地址:[here](https://anticheat.oss-accelerate.aliyuncs.com/4528842b-1cf1-4f42-a12c-288c6f22b8d0?response-content-disposition=attachment%3Bfilename%3Dresult.zip&Expires=1654871972&OSSAccessKeyId=LTAIFqYvfrh1znFK&Signature=BynIDicB2O7l%2FZPzJO6NlFp3Eyw%3D)


## 训练

终端中运行以下命令：
```
python train.py
```

## 致谢

此项目基于论文 *A Style-Based Generator Architecture for Generative Adversarial Networks* 实现，部分代码参考了 [jittor-gan](https://github.com/Jittor/gan-jittor)。

