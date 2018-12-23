## CycleGAN
1. instance Normalization
2. 使用LSGAN(最小二乘GAN least square gan)
3. 使用replay buffer
4. 使用L1距离作为cycle loss
5. 没有随机输入z,没有dropout

## DualGAN
1. 没有随机输入z，但是有dropout随机
2. 使用WGAN
3. 使用L1距离作为cycle loss

## DiscoGAN
1. 生成器：conv，deconv和leaky relu
2. 判别器：conv+leaky relu
3. 使用L2距离作为cycle loss