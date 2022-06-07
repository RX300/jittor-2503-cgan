# ----------
#  模型训练
# ----------
import jittor as jt
import numpy as np
from CGAN import opt, dataloader, generator, discriminator, adversarial_loss, optimizer_G, optimizer_D, sample_image

for epoch in range(opt.n_epochs):
    for i, (imgs, labels) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        # 数据标签，valid=1表示真实的图片，fake=0表示生成的图片
        valid = jt.ones([batch_size, 1]).float32().stop_grad()
        fake = jt.zeros([batch_size, 1]).float32().stop_grad()

        # 真实图片及其类别
        real_imgs = jt.array(imgs)
        labels = jt.array(labels)

        # -----------------
        #  训练生成器
        # -----------------

        # 采样随机噪声和数字类别作为生成器输入
        z = jt.array(np.random.normal(0, 1, (batch_size, opt.latent_dim))).float32()
        gen_labels = jt.array(np.random.randint(0, opt.n_classes, batch_size)).float32()

        # 生成一组图片
        gen_imgs = generator(z, gen_labels)
        # 损失函数衡量生成器欺骗判别器的能力，即希望判别器将生成图片分类为valid
        validity = discriminator(gen_imgs, gen_labels)
        g_loss = adversarial_loss(validity, valid)
        g_loss.sync()
        optimizer_G.step(g_loss)

        # ---------------------
        #  训练判别器
        # ---------------------

        validity_real = discriminator(real_imgs, labels)
        d_real_loss = adversarial_loss(validity_real, valid)

        validity_fake = discriminator(gen_imgs.stop_grad(), gen_labels)
        d_fake_loss = adversarial_loss(validity_fake, fake)

        # 总的判别器损失
        d_loss = (d_real_loss + d_fake_loss) / 2
        d_loss.sync()
        optimizer_D.step(d_loss)
        if i % 50 == 0:
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.data, g_loss.data)
            )

        batches_done = epoch * len(dataloader) + i
        if batches_done % opt.sample_interval == 0:
            sample_image(n_row=10, batches_done=batches_done)

    if epoch % 10 == 0:
        generator.save("generator_last.pkl")
        discriminator.save("discriminator_last.pkl")


