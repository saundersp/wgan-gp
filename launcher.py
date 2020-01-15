import argparse
import time
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser(description="WGAN-GP")

    # Saving parameters
    parser.add_argument("--name", "-n", "-id", type=str, default=str(int(time.time())),
                        help="Name/ID of the current training model")
    parser.add_argument("--resume_from", "-rf", type=int, default=0,
                        help="Number of epoch to resume from (if existing)")
    parser.add_argument("--checkpoint_interval", "-ci", type=int, default=20,
                        help="Number of epoch before saving a checkpoint (0 to disable checkpoints) (default=20)")

    # Model hyper parameters
    parser.add_argument("--learning_rate_d", "-lrd", type=float, default=2e-4,
                        help="Learning rate of the critic (default=2e-4)")
    parser.add_argument("--learning_rate_g", "-lrg", type=float, default=2e-4,
                        help="Learning rate of the generator (default=2e-4)")
    parser.add_argument("--beta_1", "-b1", type=float, default=0.5,
                        help="BETA 1 of the optimizer (default=0.5)")
    parser.add_argument("--beta_2", "-b2", type=float, default=0.9,
                        help="BETA 2 of the optimizer (default=0.9)")
    parser.add_argument("--training_ratio", "-tr", type=int, default=5,
                        help="Training ratio of the critic (default=5)")
    parser.add_argument("--gradient_penalty_weight", "-gpd", type=int, default=10,
                        help="Gradient penalty weight applied to the critic (default=10)")
    parser.add_argument("--z_size", type=int, default=128,
                        help="Size of the noise vector of the generator (default=128)")

    # General hyper parameters
    parser.add_argument("--epoch", type=int, default=10000,
                        help="Number of epoch to train (default=10000)")
    parser.add_argument("--batch_size", "-bs", type=int, default=512,
                        help="Size of the dataset mini-batch (default=512)")
    parser.add_argument("--buffer_size", "-bus", type=int, default=2048,
                        help="Size of the buffer of the dataset iterator (default=2048)")
    parser.add_argument("--prefetch_size", "-ps", type=int, default=10,
                        help="Size of prefetching of the dataset iterator (default=10)")

    # Layers hyper parameters
    parser.add_argument("--bn_momentum", "-bm", type=float, default=0.8,
                        help="Momentum of the batch normalization layer (default=0.8)")
    parser.add_argument("--lr_alpha", "-la", type=float, default=0.2,
                        help="Alpha of the LeakyReLU layer (default=0.2)")
    parser.add_argument("--kernel_size", "-ks", type=int, default=5,
                        help="Size of the kernel of the convolutional layer (best if odd) (default=5)")
    parser.add_argument("--rn_stddev", "-rs", type=float, default=0.02,
                        help="Standard deviation of the initialization of the weights of each layers (default=0.02)")
    parser.add_argument("--min_weight", "-mw", type=int, default=5,
                        help="Minimum size pow(2, mw) of the first layer of convolutional layer (doubles each times) (default=5)")

    # Dataset parameters
    parser.add_argument("--type", "-t", type=str, default="digits",
                        choices=["custom", "digits", "fashion", "cifar10",
                                 "cifar100", "celebA_128", "LAG48", "LAG128",
                                 "cars64"],
                        help="Type of dataset to use (default='digits')")
    args = parser.parse_args()
    print(args)

    from wgan_gp import WGAN_GP
    from toolbox import extract_mnist
    from tensorflow.keras.datasets import mnist, fashion_mnist
    from tensorflow.keras.datasets import cifar10, cifar100

    if args.type == "custom":
        print("Custom type is not yet implemented !")
        return
    elif args.type in ["digits", "fashion"]:
        sample_shape = (7, 7)
        output_shape = (28, 28, 1)
        min_wh = 7
        tensor_to_img = False
        nb_layers = 3
        data_dir = "keras"
        X_train = extract_mnist((mnist, fashion_mnist)[args.type == "fashion"])
    elif args.type in ["cifar10", "cifar100"]:
        sample_shape = (7, 7)
        output_shape = (32, 32, 3)
        min_wh = 4
        tensor_to_img = False
        nb_layers = 4
        data_dir = "keras"
        X_train = (cifar10, cifar100)[args.type == "cifar100"]
        X_train = extract_mnist(X_train, img_shape=output_shape)  # , label=1)
    elif args.type == "celebA_128":
        sample_shape = (5, 5)
        output_shape = (128, 128, 3)
        min_wh = 4
        data_dir = './datasets/celebA_128'
        X_train = np.array(os.listdir(data_dir))
        tensor_to_img = True
        nb_layers = 6
    elif args.type == "LAG48":
        sample_shape = (5, 5)
        output_shape = (48, 48, 3)
        min_wh = 3
        data_dir = './datasets/_binarynumpy/normalized_LAGdataset_48.npy'
        X_train = np.load(data_dir)
        tensor_to_img = False
        nb_layers = 5
    elif args.type == "LAG128":
        sample_shape = (5, 5)
        output_shape = (128, 128, 3)
        min_wh = 4
        data_dir = './datasets/_binarynumpy/normalized_LAGdataset_128.npy'
        X_train = np.load(data_dir)
        tensor_to_img = False
        nb_layers = 6
    elif args.type == "cars64":
        sample_shape = (5, 5)
        output_shape = (64, 64, 3)
        min_wh = 4
        data_dir = './datasets/_binarynumpy/normalized_cars.npy'
        X_train = np.load(data_dir)
        tensor_to_img = False
        nb_layers = 5

    # TODO Add Imagenet 32 & 64
    '''
    sample_shape = (7, 7)
    output_shape = (32, 32, 3)
    min_wh = 4
    data_dir = './datasets/Imagenet/train_32x32'
    X_train = np.array(os.listdir(data_dir))
    name = f"wgan-gp_imagenet32_{int(time.time())}"
    tensor_to_img = True
    min_wei = 5  # 2 ** 5 => 32
    nb_layers = 4
    '''

    name = f"wgan-gp_{args.type}_{args.name}"
    weights = [pow(2, i) for i in range(args.min_weight, args.min_weight+nb_layers)]

    model = WGAN_GP(name, args.learning_rate_d, args.learning_rate_g,
                    args.beta_1, args.beta_2, args.training_ratio,
                    args.gradient_penalty_weight, args.z_size,
                    args.bn_momentum, args.lr_alpha, args.kernel_size,
                    args.rn_stddev)
    model.feed_data(X_train, data_dir, tensor_to_img, args.batch_size,
                    args.buffer_size, args.prefetch_size)
    model.set_output(sample_shape, output_shape)
    model.create_model(args.min_weight, min_wh, weights, nb_layers)
    model.print_desc(args.resume_from)
    model.train(args.epoch, args.checkpoint_interval, args.resume_from)


if __name__ == "__main__":
    main()
else:
    print("Launcher.py is to be used on its own")
