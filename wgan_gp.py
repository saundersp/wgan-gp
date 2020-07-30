from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Reshape, Flatten, Activation
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, LeakyReLU
from tensorflow.keras.layers import Conv2DTranspose, LayerNormalization, UpSampling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.summary import scalar, create_file_writer, image, text
from tensorflow.keras import metrics
import tensorflow as tf
from functools import partial
from toolbox import format_time
import numpy as np
import os
import time
import matplotlib
from matplotlib.image import imread

# Change matplot backend engine
matplotlib.use('Agg')

def d_loss_fnc(fake_logits, real_logits):
	return tf.reduce_mean(fake_logits) - tf.reduce_mean(real_logits)

def g_loss_fnc(fake_logits):
	return -tf.reduce_mean(fake_logits)

def gradient_penalty(fnc, BATCH_SIZE, real, fake):
	alpha = tf.random.uniform((BATCH_SIZE, 1, 1, 1), 0.0, 1.0)
	diff = fake - real
	inter = real + (alpha * diff)
	with tf.GradientTape() as tape:
		tape.watch(inter)
		pred = fnc(inter)
	grad = tape.gradient(pred, [inter])[0]
	slopes = tf.sqrt(tf.reduce_sum(tf.square(grad), axis = [1, 2, 3]))
	gp = tf.reduce_mean((slopes - 1.) ** 2)
	return gp

def make_noise(size, Z_SIZE):
	return tf.random.normal((size, 1, 1, Z_SIZE))

class WGAN_GP():
	def __init__(self, name, LEARNING_RATE_D, LEARNING_RATE_G, BETA_1, BETA_2,
				 TRAINING_RATIO, GRADIENT_PENALTY_WEIGHT, Z_SIZE, BN_MOMENTUM,
				 LR_ALPHA, KERNEL_SIZE, RN_STDDEV):

		self.name = name

		# Hyper-parameters as per the paper
		self.LEARNING_RATE_D = LEARNING_RATE_D
		self.LEARNING_RATE_G = LEARNING_RATE_G
		self.BETA_1 = BETA_1
		self.BETA_2 = BETA_2
		self.TRAINING_RATIO = TRAINING_RATIO
		self.GRADIENT_PENALTY_WEIGHT = GRADIENT_PENALTY_WEIGHT
		self.Z_SIZE = Z_SIZE

		# Layer hyper parameters
		self.BN_MOMENTUM = BN_MOMENTUM
		self.LR_ALPHA = LR_ALPHA
		self.KERNEL_SIZE = KERNEL_SIZE
		self.RN_STDDEV = RN_STDDEV

	def create_model(self, min_wei, min_wh, weights, nb_layers):
		self.min_wh = min_wh
		self.min_wei = min_wei
		self.nb_layers = nb_layers
		self.make_combined(min_wh, weights)
		self.prepare_dirs()

	def set_output(self, sample_shape, output_shape):
		self.sample_shape = sample_shape
		self.output_shape = output_shape

	def print_desc(self, resume_from):
		desc = f"NAME: {self.name}\n"
		desc += f"DATA_DIR: {self.data_dir}\n"
		desc += f"BATCH_SIZE: {self.BATCH_SIZE}\n"
		desc += f"BUFFER_SIZE: {self.BUFFER_SIZE}\n"
		desc += f"PREFETCH_SIZE: {self.PREFETCH_SIZE}\n"
		desc += f"NB_BATCHES: {self.NB_BATCHES}\n"
		desc += f"SAMPLE_SHAPE: {self.sample_shape}\n"
		desc += f"OUTPUT_SHAPE: {self.output_shape}\n"
		desc += f"MIN_WIDTH/HEIGHT: {self.min_wh}\n"
		desc += f"MIN_WEIGHT: {self.min_wei} == {2 ** self.min_wei}\n"
		desc += f"NB_LAYERS: {self.nb_layers}\n"
		desc += f"LEARNING_RATE_D: {self.LEARNING_RATE_D}\n"
		desc += f"LEARNING_RATE_G: {self.LEARNING_RATE_G}\n"
		desc += f"BETA_1: {self.BETA_1}\n"
		desc += f"BETA_2: {self.BETA_2}\n"
		desc += f"TRAINING_RATIO: {self.TRAINING_RATIO}\n"
		desc += f"GRADIENT_PENALTY_WEIGHT: {self.GRADIENT_PENALTY_WEIGHT}\n"
		desc += f"Z_SIZE: {self.Z_SIZE}\n"
		desc += f"LR_ALPHA: {self.LR_ALPHA}\n"
		desc += f"BN_MOMENTUM: {self.BN_MOMENTUM}\n"
		desc += f"KERNEL_SIZE: {self.KERNEL_SIZE}\n"
		print(desc)

		with self.writer.as_default():
			tf.summary.trace_on()
			text("Hyper-parameters", desc, step = resume_from)
			self.writer.flush()

	def feed_data(self, data, data_dir, tensor_to_img, batch_size, buffer_size,
				  prefetch_size):
		self.BATCH_SIZE = batch_size
		self.BUFFER_SIZE = buffer_size
		self.PREFETCH_SIZE = prefetch_size
		self.NB_BATCHES = data.shape[0] // batch_size
		self.tensor_to_img = tensor_to_img
		self.data_dir = data_dir
		X_train = tf.data.Dataset.from_tensor_slices(data).repeat()
		X_train = X_train.shuffle(buffer_size = buffer_size,
								  reshuffle_each_iteration = True)
		self.X_train = X_train.batch(batch_size).prefetch(prefetch_size)

	def make_generator(self, min_wh, gen_filters):

		model = Sequential(name = "generator")
		model.add(Input((1, 1, self.Z_SIZE)))

		model.add(Dense(min_wh ** 2 * gen_filters[0],
						kernel_initializer = 'he_normal'))
		# model.add(BatchNormalization(momentum = self.BN_MOMENTUM))
		# model.add(ReLU())
		model.add(LeakyReLU(alpha = self.LR_ALPHA))
		model.add(Reshape((min_wh, min_wh, gen_filters[0])))

		for i in range(1, len(gen_filters)):
			model.add(Conv2DTranspose(gen_filters[i], strides = 2, use_bias = False,
			# model.add(UpSampling2D((2, 2)))
			# model.add(Conv2D(gen_filters[i], strides = 1, use_bias = False,
									  kernel_size = self.KERNEL_SIZE, padding = 'same',
									  kernel_initializer = 'he_normal'
									  ))

			# model.add(BatchNormalization(momentum = self.BN_MOMENTUM))
			# model.add(ReLU())
			model.add(LeakyReLU(alpha = self.LR_ALPHA))

		model.add(Conv2D(self.output_shape[2], kernel_size = self.KERNEL_SIZE,
						 use_bias = False, padding = 'same',
						 kernel_initializer = 'he_normal'))
		model.add(Activation("tanh"))

		return model

	def make_discriminator(self, disc_filters):

		model = Sequential(name = "discriminator")
		model.add(Input(self.output_shape))

		for i in range(len(disc_filters)-1):
			model.add(Conv2D(disc_filters[i], kernel_size = self.KERNEL_SIZE,
							 use_bias = False, strides = 2, padding = 'same',
							 kernel_initializer = 'he_normal'
							 ))
			# model.add(LayerNormalization(axis = [1, 2, 3]))
			model.add(LeakyReLU(alpha = self.LR_ALPHA))

		model.add(Flatten())
		model.add(Dense(1, kernel_initializer = 'he_normal'))

		return model

	def make_combined(self, min_wh, weights):
		self.generator = self.make_generator(min_wh, np.flip(weights))
		self.discriminator = self.make_discriminator(weights)

		self.discriminator.summary()
		print(self.discriminator.metrics_names)

		self.generator.summary()
		print(self.generator.metrics_names)

		self.discriminator_opt = Adam(self.LEARNING_RATE_G, beta_1 = self.BETA_1, beta_2 = self.BETA_2)
		self.generator_opt = Adam(self.LEARNING_RATE_G, beta_1 = self.BETA_1, beta_2 = self.BETA_2)

	def extract_images_tensor(self, data_dir, tensor):
		files = tensor.numpy()
		images = []
		len_files = len(files)
		for i in range(len_files):
			file_path = os.path.join(data_dir, files[i].decode('utf-8'))
			image = imread(file_path)
			image = np.array(image).astype(np.float32) / 127.5
			images.append(image - 1.0)
		images = np.array(images).reshape((len_files, self.output_shape[0], self.output_shape[1], self.output_shape[2]))
		return images

	def train(self, NB_EPOCH, save_checkpoint, resume_from):

		if resume_from != 0:
			print(f"RESUMED_FROM: {resume_from}/{NB_EPOCH}")
			fixed_seed = np.load(os.path.join(self.log_dir, 'seed.npy'))
			fixed_seed = tf.constant(fixed_seed)
			resume_from += 1
		else:
			fixed_seed = make_noise(self.sample_shape[0] * self.sample_shape[1], self.Z_SIZE)
			np.save(os.path.join(self.log_dir, 'seed'), fixed_seed.numpy())

		print(f"Training for {NB_EPOCH} epochs, NB_BATCHES: {self.NB_BATCHES}")
		time_left = "is be determined"

		for epoch in range(resume_from, NB_EPOCH):
			ga_loss, da_loss = [], []
			g_train_loss, d_train_loss = metrics.Mean(), metrics.Mean()
			start_time = time.time()
			for i, image_batch in enumerate(self.X_train.take(self.NB_BATCHES)):
				print(f"Epoch    : {epoch:05d}/{NB_EPOCH} in progress {i}/{self.NB_BATCHES} ending {time_left}", end = '\r')
				if self.tensor_to_img is True:
					image_batch = self.extract_images_tensor(self.data_dir, image_batch)

				for _ in range(self.TRAINING_RATIO):
					d_loss = self.train_discriminator(image_batch)
					d_train_loss(d_loss)

				g_loss = self.train_generator()
				g_train_loss(g_loss)

				da_loss.append(d_train_loss.result())
				ga_loss.append(g_train_loss.result())

			g_train_loss.reset_states()
			d_train_loss.reset_states()
			da_loss = -np.mean(da_loss)
			ga_loss = -np.mean(ga_loss)
			with self.writer.as_default():
				scalar("da_loss", da_loss, step = epoch)
				scalar("ga_loss", ga_loss, step = epoch)
				self.save_images(self.generator, fixed_seed, self.writer, epoch)
				self.writer.flush()
			if epoch % save_checkpoint == 0:
				self.checkpoint.save(file_prefix = self.checkpoint_prefix)
			date = time.strftime('%d/%m/%Y %H:%M:%S')
			time_spent = time.time() - start_time
			time_left = "in " + format_time(time_spent * (NB_EPOCH - epoch))
			time_spent = format_time(time_spent)

			#if epoch % save_checkpoint == 0:
			#	print(f"Epoch CHK: {epoch:05d}/{NB_EPOCH} {date} = da_loss {da_loss:.5f}, ga_loss {ga_loss:.5f}, time_spent {time_spent}")
			#else:
			#	print(f"Epoch	 : {epoch:05d}/{NB_EPOCH} {date} = da_loss {da_loss:.5f}, ga_loss {ga_loss:.5f}, time_spent {time_spent}")
			print(f'Epoch {"CHK" if epoch % save_checkpoint == 0 else "   "}: {epoch:05d}/{NB_EPOCH} {date} = da_loss {da_loss:.5f}, ga_loss {ga_loss:.5f}, time_spent {time_spent}')

	@tf.function
	def train_generator(self):
		z = make_noise(self.BATCH_SIZE, self.Z_SIZE)
		with tf.GradientTape() as tape:
			x_fake = self.generator(z, training = True)
			fake_logits = self.discriminator(x_fake, training = False)
			loss = g_loss_fnc(fake_logits)
		grad = tape.gradient(loss, self.generator.trainable_variables)
		grad = zip(grad, self.generator.trainable_variables)
		self.generator_opt.apply_gradients(grad)
		return loss

	@tf.function
	def train_discriminator(self, x_real):
		z = make_noise(self.BATCH_SIZE, self.Z_SIZE)
		with tf.GradientTape() as tape:
			x_fake = self.generator(z, training = False)
			fake_logits = self.discriminator(x_fake, training = True)
			real_logits = self.discriminator(x_real, training = True)
			cost = d_loss_fnc(fake_logits, real_logits)
			fnc = partial(self.discriminator, training = True)
			gp = gradient_penalty(fnc, self.BATCH_SIZE, x_real, x_fake)
			cost += self.GRADIENT_PENALTY_WEIGHT * gp
		grad = tape.gradient(cost, self.discriminator.trainable_variables)
		grad = zip(grad, self.discriminator.trainable_variables)
		self.discriminator_opt.apply_gradients(grad)
		return cost

	def prepare_dirs(self):
		self.log_dir = f"./logs/{self.name}"
		if not os.path.exists(self.log_dir):
			os.makedirs(self.log_dir)
			tensorboard = TensorBoard(log_dir = self.log_dir)
			tensorboard.set_model(self.discriminator)

		self.writer = create_file_writer(self.log_dir)

		checkpoint_dir = f'.\\checkpoints\\{self.name}'
		self.checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
		self.checkpoint = tf.train.Checkpoint(generator = self.generator,
											  discriminator = self.discriminator)
		manager = tf.train.CheckpointManager(self.checkpoint, checkpoint_dir,
											 max_to_keep = 5)
		if manager.latest_checkpoint:
			self.checkpoint.restore(manager.latest_checkpoint)
			print(f"Restored from {manager.latest_checkpoint}")

	@tf.function
	def generate_images(self, model, z):
		return model(z, training = False)

	def save_images(self, generator, fixed_seed, writer, epoch):
		gen_images = (self.generate_images(generator, fixed_seed) + 1.0) / 2.0
		rows = []
		y, x = self.sample_shape
		for i in range(y):
			row = [gen_images[i * y + j] for j in range(x)]
			row = np.concatenate(row, axis = 1)
			rows.append(row)
		out_image = np.concatenate([row for row in rows], axis = 0)
		image("Generated image", np.array([out_image]), step = epoch)
		# Image.fromarray(out_image).save(f"./output/{name}/{epoch:05d}.png")

if __name__ == "__main__":
	print("Use launcher.py to use this file")