from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Activation
from tensorflow.keras.layers import Conv2D, ReLU, LeakyReLU, Conv2DTranspose
from tensorflow.keras.optimizers import Adam
from wgan_gp import d_loss_fnc, g_loss_fnc, generate_images
from tensorflow.keras import backend as K, metrics
import tensorflow as tf
from functools import partial
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.summary import scalar, create_file_writer
from tensorflow.summary import image
import numpy as np
import time
import os


def gradient_penalty(fnc, real, fake):
    alpha = tf.random.uniform((BATCH_SIZE, 1, 1, 1), 0.0, 1.0)
    diff = fake - real
    inter = real + (alpha * diff)
    with tf.GradientTape() as tape:
        tape.watch(inter)
        pred = fnc(inter)
    grad = tape.gradient(pred, [inter])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=[1, 2, 3]))
    gp = tf.reduce_mean((slopes - 1.) ** 2)
    return gp


def make_generator(min_wh, gen_filters):

    model = Sequential(name="generator")
    model.add(Input((1, 1, Z_SIZE)))

    model.add(Dense(min_wh ** 2 * gen_filters[0],
                    kernel_initializer='he_normal'))
    # model.add(BatchNormalization(momentum=BN_MOMENTUM))
    model.add(ReLU())
    model.add(Reshape((min_wh, min_wh, gen_filters[0])))

    for i in range(1, len(gen_filters)):
        model.add(Conv2DTranspose(gen_filters[i], strides=2, use_bias=False,
                                  kernel_size=KERNEL_SIZE, padding='same',
                                  kernel_initializer='he_normal'
                                  ))
        # model.add(BatchNormalization(momentum=BN_MOMENTUM))
        model.add(ReLU())

    model.add(Conv2DTranspose(output_shape[2], kernel_size=KERNEL_SIZE,
                              use_bias=False, padding='same',
                              kernel_initializer='he_normal'
                              ))
    model.add(Activation("tanh"))

    return model


def make_discriminator(disc_filters):

    model = Sequential(name="discriminator")
    model.add(Input(output_shape))

    for i in range(len(disc_filters)-1):
        model.add(Conv2D(disc_filters[i], kernel_size=KERNEL_SIZE,
                         use_bias=False, strides=2, padding='same',
                         kernel_initializer='he_normal'
                         ))
        # model.add(LayerNormalization(axis=[1, 2, 3]))
        model.add(LeakyReLU(alpha=LR_ALPHA))

    model.add(Flatten())
    # 'he_normal' instead of RandomNormal(stddev=RN_STDDEV)
    model.add(Dense(1, kernel_initializer='he_normal'))

    return model


def make_combined(min_wh, weights):
    generator = make_generator(min_wh, np.flip(weights))
    discriminator = make_discriminator(weights)

    discriminator.summary()
    print(discriminator.metrics_names)

    generator.summary()
    print(generator.metrics_names)

    discriminator_opt = Adam(LEARNING_RATE_G, beta_1=BETA_1, beta_2=BETA_2)
    generator_opt = Adam(LEARNING_RATE_G, beta_1=BETA_1, beta_2=BETA_2)

    return (generator, discriminator, generator_opt, discriminator_opt)


def make_noise(size):
    return tf.random.uniform((size, 1, 1, Z_SIZE))
    # return tf.random.normal((size, 1, 1, Z_SIZE))


def train(X_train, log_dir, writer):

    fixed_seed = make_noise(BATCH_SIZE)

    for epoch in range(NB_EPOCH):

        g_train_loss, d_train_loss = metrics.Mean(), metrics.Mean()

        for _ in range(TRAINING_RATIO):
            d_loss = train_discriminator(X_train)
            d_train_loss(d_loss)

        g_loss = train_generator()
        g_train_loss(g_loss)

        da_loss = -np.mean(np.array(d_train_loss.result()))
        ga_loss = -np.mean(np.array(g_train_loss.result()))
        g_train_loss.reset_states()
        d_train_loss.reset_states()

        with writer.as_default():
            scalar("da_loss", da_loss, step=epoch)
            scalar("ga_loss", ga_loss, step=epoch)
            save_images(generator, fixed_seed, writer, epoch)
            writer.flush()

        print(f"Epoch: {epoch:05d}/{NB_EPOCH} = da_loss {da_loss:.5f}, ga_loss {ga_loss:.5f}")


def save_images(generator, fixed_seed, writer, epoch):
    # gen_images = np.uint8((generator.predict(fixed_seed) + 1.0) * 127.5)
    gen_images = (generate_images(generator, fixed_seed) + 1.0) / 2.0
    # img scale is [-1, 1] but back to [0, 1]
    rows = []
    y, x = sample_shape
    for i in range(y):
        row = [gen_images[i * y + j] for j in range(x)]
        row = np.concatenate(row, axis=1)
        rows.append(row)
    out_image = np.concatenate([row for row in rows], axis=0)
    image("Generated image", np.array([out_image]), step=epoch)
    # Image.fromarray(out_image).save(f"./output/{name}/{epoch:05d}.png")


@tf.function
def train_generator():
    z = make_noise(BATCH_SIZE)
    with tf.GradientTape() as tape:
        x_fake = generator(z, training=True)
        fake_logits = discriminator(x_fake, training=False)
        loss = g_loss_fnc(fake_logits)
    grad = tape.gradient(loss, generator.trainable_variables)
    generator_opt.apply_gradients(zip(grad, generator.trainable_variables))
    return loss


@tf.function
def train_discriminator(x_real):
    z = make_noise(BATCH_SIZE)
    with tf.GradientTape() as tape:
        x_fake = generator(z, training=False)
        fake_logits = discriminator(x_fake, training=True)
        real_logits = discriminator(x_real, training=True)
        cost = d_loss_fnc(fake_logits, real_logits)
        fnc = partial(discriminator, training=True)
        gp = gradient_penalty(fnc, x_real, x_fake)
        cost += GRADIENT_PENALTY_WEIGHT * gp
    grad = tape.gradient(cost, discriminator.trainable_variables)
    grad = zip(grad, discriminator.trainable_variables)
    discriminator_opt.apply_gradients(grad)
    return cost


def prepare_log(name):
    log_dir = f"./logs/{name}"
    os.makedirs(log_dir, exist_ok=True)
    tensorboard = TensorBoard(log_dir=log_dir)
    tensorboard.set_model(discriminator)

    writer = create_file_writer(log_dir)

    return (log_dir, writer)


if __name__ == "__main__":

    # Hyper-parameters as per the paper
    LEARNING_RATE_D = 1e-4
    LEARNING_RATE_G = LEARNING_RATE_D
    BETA_1 = 0
    BETA_2 = 0.9
    TRAINING_RATIO = 5
    GRADIENT_PENALTY_WEIGHT = 10
    Z_SIZE = 1  # Random vector noise size

    # Global hyper parameters
    NB_EPOCH = 10000
    BATCH_SIZE = 4

    # Layer hyper parameters
    BN_MOMENTUM = 0.8
    LR_ALPHA = 0.2
    KERNEL_SIZE = 4
    RN_STDDEV = 0.02

    # cars
    '''
    sample_shape = (2, 2)
    output_shape = (64, 64, 3)
    min_wh = 4
    X_train = np.array(np.load('./datasets/_binarynumpy/normalized_cars.npy')[:4])
    min_wei = 5  # 2 ** 5 => 32
    nb_layers = 5
    weights = [pow(2, i) for i in range(min_wei, min_wei+nb_layers)]
    # weights = [pow(2, min_wei) for i in range(nb_layers)]
    name = f"test_cars_{int(time.time())}"
    '''

    # LAG128
    sample_shape = (2, 2)
    output_shape = (128, 128, 3)
    min_wh = 4
    X_train = np.array(np.load('./datasets/_binarynumpy/normalized_LAGdataset_128.npy')[:4])
    min_wei = 5  # 2 ** 5 => 32
    nb_layers = 6
    weights = [pow(2, i) for i in range(min_wei, min_wei+nb_layers)]
    # weights = [pow(2, min_wei) for i in range(nb_layers)]
    name = f"test_LAG128_{int(time.time())}"

    # Experiment thought :
    # 1 filters per transpose, kernel_size = size of image
    # See custom layer

    # Thought 2:
    # Adverserial 1 Generator 1 discriminator for 3 images
    # with indexes 0-0.5-1 per images
    # To see if we can generates the distributions

    K.clear_session()
    (generator, discriminator, generator_opt, discriminator_opt) = make_combined(min_wh, weights)
    (log_dir, writer) = prepare_log(name)

    train(X_train, log_dir, writer)
