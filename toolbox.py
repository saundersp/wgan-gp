import os
import numpy as np
import vtkplotter as vp
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.image import imread
from PIL import Image

# Change matplot backend engine
matplotlib.use('Agg')


def show_min_max_images(src):
    min_w, min_h = 9999, 9999
    max_w, max_h = 0, 0

    if isinstance(src, list):
        for s in src:
            for image_path in os.listdir(s):
                img = np.array(
                    Image.open(
                        os.path.join(
                            s, image_path)))
                min_w = min(img.shape[0], min_w)
                min_h = min(img.shape[1], min_h)
                max_w = max(img.shape[0], max_w)
                max_h = max(img.shape[1], max_h)
    else:
        for image_path in os.listdir(src):
            img = np.array(Image.open(os.path.join(src, image_path)))
            min_w = min(img.shape[0], min_w)
            min_h = min(img.shape[1], min_h)
            max_w = max(img.shape[0], max_w)
            max_h = max(img.shape[1], max_h)

    print(min_w, min_h, max_w, max_h)


def show_below_size_images(src, size):

    count = 0

    if isinstance(src, list):
        for s in src:
            for image_path in os.listdir(s):
                img = np.array(
                    Image.open(
                        os.path.join(
                            s, image_path)))
                if img.shape[0] < size[0] or img.shape[1] < size[1]:
                    # print(image_path)
                    os.remove(os.path.join(s, image_path))
                    count += 1
    else:
        for image_path in os.listdir(src):
            img = np.array(Image.open(os.path.join(src, image_path)))
            if img.shape[0] < size[0] or img.shape[1] < size[1]:
                # print(image_path)
                os.remove(os.path.join(s, image_path))
                count += 1

    print(f"Total of {count} images")


def normalize_images(src, dst, size):
    os.makedirs(dst, exist_ok=True)

    if isinstance(src, list):
        name = 0
        i = 0
        for s in src:
            images = os.listdir(s)
            len_images = len(images)
            print(f"Normalizing {len_images} images in {s}")
            for image_path in images:
                im = Image.open(os.path.join(s, image_path))
                if im.mode != 'RGB':
                    im = im.convert('RGB')
                im = im.resize(size, Image.LANCZOS)
                im.save(os.path.join(dst, f"{name:05d}.jpg"))
                name += 1
                i += 1
                print(
                    f"Normalized {i}/{len_images} = {name/len_images:.2%}",
                    end="\r")
            print(f'{s} Normalized in {dst}!')
            i -= len_images
    else:
        images = os.listdir(src)
        len_images = len(images)
        print(f"Normalizing {len_images} images in {src}")
        for i in range(len_images):
            image_path = images[i]
            im = Image.open(os.path.join(src, image_path))
            if im.mode != 'RGB':
                im = im.convert('RGB')
            im = im.resize(size, Image.LANCZOS)
            im.save(os.path.join(dst, image_path))
            print(
                f"Normalized {i}/{len_images} = {i/len_images:.2%}",
                end="\r")
        print(f'{src} Normalized in {dst}!')


def normalize_and_crop(src, dst, size):
    images = os.listdir(src)
    len_images = len(images)
    print(f"Normalizing {len_images} images in {src}")
    result = []
    for n in range(len_images):
        image_path = images[n]
        im = Image.open(os.path.join(src, image_path))
        if im.size != size:
            w = 120
            h = 120
            j = 38  # (218 - w) // 2
            i = 60  # (178 - h) // 2
            im = im.crop([j, i, j + w, i + h])
            im = im.resize(size, Image.LANCZOS)
        im.save(os.path.join(dst, image_path))
        result.append(np.array(im))
        print(
            f"Normalized {n}/{len_images} = {n/len_images:.2%}",
            end="\r")
    print(f'{src} Normalized in {dst}!')

    return (np.array(result).astype(np.float32) - 127.5) / 127.5


def extract_mnist(data, img_shape=(28, 28, 1), label=None):
    (x_train, y_train), (x_test, y_test) = data.load_data()

    if label:
        x_train = x_train[(y_train.reshape(-1) == label)]
        x_test = x_test[(y_test.reshape(-1) == label)]

    x_train = np.concatenate([x_train, x_test])

    # Reshaping
    x_train = x_train.reshape(
        x_train.shape[0],
        img_shape[0],
        img_shape[1],
        img_shape[2]).astype(
        np.float32)
    # Normalization to [-1, 1]
    x_train = x_train / 127.5 - 1.0

    return x_train


def extract_images_list(path, files):
    images = []
    len_files = len(files)
    for i in range(len_files):
        image = imread(os.path.join(path, files[i]))
        images.append(
            (np.array(image).astype(
                np.float32) -
                127.5) /
            127.5)
    images = np.array(images).reshape(
        len_files,
        images[0].shape[0],
        images[0].shape[1],
        images[0].shape[2])
    return images


def extract_images(path):

    images = []
    files = os.listdir(path)
    total = len(files)
    for i in range(total):
        image_path = os.path.join(path, files[i])
        image = np.array(
            imread(image_path)).astype(
            np.float32) / 127.5 - 1.0
        images.append(image)
        print(f"{i:06d}/{total}", end='\r')

    return np.array(images)


def get_binary_array(path):
    binary_dir = os.path.join('./datasets/_binarynumpy', path)
    if os._exists(binary_dir):
        return np.load(binary_dir)

    binary_data = extract_images(os.path.join("./datasets", path))
    np.save(binary_dir, binary_data)
    return binary_data


def format_time(time):
    if time < 60:
        return f"{time:.2f}s"
    # 60*60
    elif time < 3600:
        return f"{int(time/60):02d}m {int(time%60):02d}s"
    # 60*60*24
    elif time < 86400:
        nb_hours = int(time / 3600)
        left = time - (nb_hours * 3600)
        return f"{nb_hours:02d}h {int(left/60):02d}m {int(left%60):02d}s"
    # 60*60*24*7
    elif time < 604800:
        nb_days = int(time / 86400)
        left = time - (nb_days * 86400)
        nb_hours = int(left / 3600)
        left = left - (nb_hours * 3600)
        return f"{nb_days:02d}d {nb_hours:02d}h {int(left/60):02d}m {int(left%60):02d}s"
    else:
        nb_week = int(time / 604800)
        left = time - (nb_week * 604800)
        nb_days = int(left / 86400)
        left = left - (nb_days * 86400)
        nb_hours = int(left / 3600)
        left = left - (nb_hours * 3600)
        return f"{nb_week}w {nb_days:02d}d {nb_hours:02d}h {int(left/60):02d}m {int(left%60):02d}s"


def show_data(x, y, path):
    i = 0
    plt.figure(figsize=(x, y))
    for f in os.listdir(path):
        image_path = os.path.join(path, f)
        img = np.array(Image.open(image_path))
        i += 1
        plt.subplot(x, y, i)
        plt.imshow(img)
        plt.axis('off')
        if i >= x * y:
            i = 0
            break

    plt.show()


def generate_gif(path,
                 name,
                 duration=50,
                 skip_frame=0,
                 nb_frame=40000,
                 begin_frame=0):
    frames = []

    print(f"Generating {name}.gif")

    files = os.listdir(path)[begin_frame:begin_frame + nb_frame]
    print(f"First image: {files[0]}")

    i = 0
    for image_path in files:
        if skip_frame == 0 or i % skip_frame == 0:
            im = Image.open(os.path.join(path, image_path))
            frames.append(im)
        i += 1

    print(f"Last image: {image_path}")

    n_frame = len(frames)

    length = round(duration * n_frame / 1000, 2)

    print(
        f"Generating {name}.gif nb_frame {n_frame} of length {length}s")

    os.makedirs('./samples', exist_ok=True)
    frames[0].save(f'./samples/{name}.gif',
                   format='gif',
                   append_images=frames[1:],
                   save_all=True,
                   duration=duration)
    print(f"Generated {name} !")


def view_3d(path):
    p = vp.Plotter(title='3D Model viewer')
    # .color('blue').wireframe(True).alpha(0.05).normalize().print()
    p.load(path)
    # p.load(output_model).color('red').wireframe(True).alpha(0.05)#.print()
    p.show()


if __name__ == "__main__":

    '''
    src = './datasets/pokemons'
    dst = './datasets/normalized_pokemons'
    size = (64, 64)
    '''
    '''
    src = ['./datasets/cars_test', './datasets/cars_train']
    dst = './datasets/normalized_cars'
    size = (64, 64)
    '''
    '''
    src = './datasets/LAGdataset_200'
    dst = './datasets/normalized_LAGdataset_48'
    size = (48, 48)
    '''
    '''
    src = './datasets/celebA'
    dst = './datasets/normalized_celebA'
    size = (128, 128)
    '''
    '''
    normalized = normalize_and_crop(src, dst, size)
    np.save('./datasets/_binarynumpy/celebA', normalized)
    '''

    # view_3d("D:/Download/bed/Bed Tokio Lux 180x200 Dream Land N130519.obj")

    # images = extract_images(dst)
    # show_min_max_images(src)
    # show_below_size_images(src, size)

    # show_data(5, 5, './datasets/normalized_cars')

    # get_binary_array('normalized_cars')
