import os
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from PIL import Image

# Change matplot backend engine
matplotlib.use('Agg')


def extract_image_from_event(name, tag_name, mode='RGB'):
    path = f"./logs/{name}"
    output_path = f"./output/{name}"
    os.makedirs(output_path, exist_ok=True)
    count = 0
    print(f"Extracting images from {path} - \"{tag_name}\"")
    for file_name in os.listdir(path):
        if file_name[-3:] == ".v2":
            for e in tf.compat.v1.train.summary_iterator(
                    os.path.join(path, file_name)):
                for v in e.summary.value:
                    if v.tag == tag_name:
                        arr = tf.make_ndarray(v.tensor)
                        decoded = tf.image.decode_png(arr[2])
                        img = np.array(decoded)
                        if mode == 'L':
                            img = img.reshape(
                                int(arr[0]), int(arr[1]))
                        else:
                            img = img.reshape(
                                int(arr[0]), int(arr[1]), 3)
                        img = Image.fromarray(img, mode=mode)
                        img.save(
                            f"{output_path}/step_{count:05d}.png")
                        print(f"Extracted {count} images", end=" \r")
                        count += 1
    print(f"Extracted {count} images from {path} - \"{tag_name}\"")


def generate_and_save_images(shape, path, model, epoch, seed):
    gen_images = model(seed, training=False)

    plt.figure(figsize=shape)
    gs1 = gridspec.GridSpec(shape[0], shape[1])
    gs1.update(wspace=0.0, hspace=0.0)

    for i in range(gen_images.shape[0]):
        ax1 = plt.subplot(gs1[i])
        ax1.imshow(gen_images[i] * 0.5 + 0.5)
        ax1.axis('on')
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_aspect('equal')

    # dpi => 27" 2560*1440 = 221
    # 28" 3860*2160 = 331
    plt.savefig(f'{path}/image_at_epoch_{epoch:05d}.png', dpi=442)
    plt.close()


def generate_and_save_samples(name, model, x, y, noise_dim):

    # seed = tf.random.uniform([x * y, noise_dim], -1, 1)
    seed = tf.random.normal([x * y, noise_dim])

    gen_images = model(seed, training=False)

    plt.figure(figsize=(x, y))
    for i in range(gen_images.shape[0]):
        plt.subplot(x, y, i + 1)
        reshaped_image = gen_images[i] * 0.5 + 0.5
        plt.imshow(reshaped_image)
        plt.axis('off')

    os.makedirs('./samples', exist_ok=True)

    # dpi => 27" 2560*1440 = 221
    # 28" 3860*2160 = 331
    plt.savefig(
        f'./samples/{name}_SAMPLES_{x}-{y}_{x*y}.png',
        dpi=442)
    plt.close()


if __name__ == "__main__":
    # name = "wgan-gp_cars_1568757827"
    # extract_image_from_event(name, "Generated image")
    # generate_gif(f'./output/{name}', name, duration=50)
