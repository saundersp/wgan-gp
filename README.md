# WGAN-GP with TensorFlow

Implementation of the Wasserstein Generative Adversarial Network with Gradient-Penalty with TensorFlow 2.0

## How to use

```bash
python launcher.py
```

You can customize using options specified in

```bash
python launcher.py --help
```

Output:

```bash
usage: launcher.py [-h] [--name NAME] [--resume_from RESUME_FROM]
                   [--checkpoint_interval CHECKPOINT_INTERVAL]
                   [--learning_rate_d LEARNING_RATE_D]
                   [--learning_rate_g LEARNING_RATE_G] [--beta_1 BETA_1]
                   [--beta_2 BETA_2] [--training_ratio TRAINING_RATIO]
                   [--gradient_penalty_weight GRADIENT_PENALTY_WEIGHT]
                   [--z_size Z_SIZE] [--epoch EPOCH] [--batch_size BATCH_SIZE]
                   [--buffer_size BUFFER_SIZE] [--prefetch_size PREFETCH_SIZE]
                   [--bn_momentum BN_MOMENTUM] [--lr_alpha LR_ALPHA]
                   [--kernel_size KERNEL_SIZE] [--rn_stddev RN_STDDEV]
                   [--min_weight MIN_WEIGHT]
                   [--type {custom,digits,fashion,cifar10,cifar100,celebA_128,LAG48,LAG128,cars64}]

WGAN-GP

optional arguments:
  -h, --help            show this help message and exit
  --name NAME, -n NAME, -id NAME
                        Name/ID of the current training model
  --resume_from RESUME_FROM, -rf RESUME_FROM
                        Number of epoch to resume from (if existing)
  --checkpoint_interval CHECKPOINT_INTERVAL, -ci CHECKPOINT_INTERVAL
                        Number of epoch before saving a checkpoint (0 to
                        disable checkpoints) (default = 20)
  --learning_rate_d LEARNING_RATE_D, -lrd LEARNING_RATE_D
                        Learning rate of the critic (default = 2e-4)
  --learning_rate_g LEARNING_RATE_G, -lrg LEARNING_RATE_G
                        Learning rate of the generator (default = 2e-4)
  --beta_1 BETA_1, -b1 BETA_1
                        BETA 1 of the optimizer (default = 0.5)
  --beta_2 BETA_2, -b2 BETA_2
                        BETA 2 of the optimizer (default = 0.9)
  --training_ratio TRAINING_RATIO, -tr TRAINING_RATIO
                        Training ratio of the critic (default = 5)
  --gradient_penalty_weight GRADIENT_PENALTY_WEIGHT, -gpd GRADIENT_PENALTY_WEIGHT
                        Gradient penalty weight applied to the critic
                        (default = 10)
  --z_size Z_SIZE       Size of the noise vector of the generator
                        (default = 128)
  --epoch EPOCH         Number of epoch to train (default = 10000)
  --batch_size BATCH_SIZE, -bs BATCH_SIZE
                        Size of the dataset mini-batch (default = 512)
  --buffer_size BUFFER_SIZE, -bus BUFFER_SIZE
                        Size of the buffer of the dataset iterator
                        (default = 2048)
  --prefetch_size PREFETCH_SIZE, -ps PREFETCH_SIZE
                        Size of prefetching of the dataset iterator
                        (default = 10)
  --bn_momentum BN_MOMENTUM, -bm BN_MOMENTUM
                        Momentum of the batch normalization layer
                        (default = 0.8)
  --lr_alpha LR_ALPHA, -la LR_ALPHA
                        Alpha of the LeakyReLU layer (default = 0.2)
  --kernel_size KERNEL_SIZE, -ks KERNEL_SIZE
                        Size of the kernel of the convolutional layer (best if
                        odd) (default = 5)
  --rn_stddev RN_STDDEV, -rs RN_STDDEV
                        Standard deviation of the initialization of the
                        weights of each layers (default = 0.02)
  --min_weight MIN_WEIGHT, -mw MIN_WEIGHT
                        Minimum size pow(2, mw) of the first layer of
                        convolutional layer (doubles each times) (default = 5)
  --type {custom,digits,fashion,cifar10,cifar100,celebA_128,LAG48,LAG128,cars64}, -t {custom,digits,fashion,cifar10,cifar100,celebA_128,LAG48,LAG128,cars64}
                        Type of dataset to use (default = 'digits')
```
