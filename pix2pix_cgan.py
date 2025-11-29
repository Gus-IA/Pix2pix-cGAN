import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
import os
import glob

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    LeakyReLU,
    ReLU,
    Dropout,
    Concatenate,
)
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam

# =============================
# RUTAS
# =============================
INPATH = "inputFlowers"
OUPATH = "targetFlowers"
CKPATH = "checkpoints"

# =============================
# OBTENER LISTA DE ARCHIVOS
# =============================
imgurls = sorted([os.path.basename(x) for x in glob.glob(INPATH + "/*.jpg")])

n = len(imgurls)
train_n = round(n * 0.80)

randurls = np.copy(imgurls)
np.random.seed(23)
np.random.shuffle(randurls)

tr_urls = randurls[:train_n]
ts_urls = randurls[train_n:n]

print(len(tr_urls), len(ts_urls))

IMG_WIDTH = 256
IMG_HEIGHT = 256


# =============================
# FUNCIONES DE PROCESAMIENTO
# =============================
def resize(inimg, tgimg, height, width):
    return (
        tf.image.resize(inimg, [height, width]),
        tf.image.resize(tgimg, [height, width]),
    )


def normalize(inimg, tgimg):
    return (inimg / 127.5) - 1, (tgimg / 127.5) - 1


def random_jitter(inimg, tgimg):
    inimg, tgimg = resize(inimg, tgimg, 286, 286)

    stacked = tf.stack([inimg, tgimg], axis=0)
    cropped = tf.image.random_crop(stacked, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

    inimg, tgimg = cropped[0], cropped[1]

    if tf.random.uniform(()) > 0.5:
        inimg = tf.image.flip_left_right(inimg)
        tgimg = tf.image.flip_left_right(tgimg)

    return inimg, tgimg


def load_image(filename, augment=True):
    inimg = tf.cast(
        tf.image.decode_jpeg(tf.io.read_file(INPATH + "/" + filename)), tf.float32
    )[..., :3]
    tgimg = tf.cast(
        tf.image.decode_jpeg(tf.io.read_file(OUPATH + "/" + filename)), tf.float32
    )[..., :3]

    inimg, tgimg = resize(inimg, tgimg, IMG_HEIGHT, IMG_WIDTH)

    if augment:
        inimg, tgimg = random_jitter(inimg, tgimg)

    return normalize(inimg, tgimg)


def load_train_image(filename):
    return load_image(filename, True)


def load_test_image(filename):
    return load_image(filename, False)


# Mostrar ejemplo
plt.imshow(((load_train_image(randurls[0])[1]) + 1) / 2)
plt.show()

# =============================
# DATASETS
# =============================
train_dataset = tf.data.Dataset.from_tensor_slices(tr_urls)
train_dataset = train_dataset.map(
    load_train_image, num_parallel_calls=tf.data.AUTOTUNE
).batch(1)

test_dataset = tf.data.Dataset.from_tensor_slices(ts_urls)
test_dataset = test_dataset.map(
    load_test_image, num_parallel_calls=tf.data.AUTOTUNE
).batch(1)


# =============================
# MODELOS
# =============================
def downsample(filters, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0, 0.02)

    result = Sequential()
    result.add(
        Conv2D(
            filters,
            4,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=not apply_batchnorm,
        )
    )
    if apply_batchnorm:
        result.add(BatchNormalization())

    result.add(LeakyReLU())
    return result


def upsample(filters, apply_dropout=False):
    initializer = tf.random_normal_initializer(0, 0.02)

    result = Sequential()
    result.add(
        Conv2DTranspose(
            filters,
            4,
            strides=2,
            padding="same",
            kernel_initializer=initializer,
            use_bias=False,
        )
    )
    result.add(BatchNormalization())

    if apply_dropout:
        result.add(Dropout(0.5))

    result.add(ReLU())
    return result


# =============================
# GENERATOR (CORREGIDO)
# =============================
def Generator():
    inputs = Input(shape=[256, 256, 3])

    down_stack = [
        downsample(64, apply_batchnorm=False),
        downsample(128),
        downsample(256),
        downsample(512),
        downsample(512),
        downsample(512),
        downsample(512),
        downsample(512),
    ]

    up_stack = [
        upsample(512, apply_dropout=True),
        upsample(512, apply_dropout=True),
        upsample(512, apply_dropout=True),
        upsample(512),
        upsample(256),
        upsample(128),
        upsample(64),
    ]

    initializer = tf.random_normal_initializer(0, 0.02)

    last = Conv2DTranspose(
        3,
        4,
        strides=2,
        padding="same",
        kernel_initializer=initializer,
        activation="tanh",
    )

    # U-NET
    x = inputs
    skips = []

    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = list(reversed(skips[:-1]))

    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = Concatenate()([x, skip])

    x = last(x)

    return Model(inputs=inputs, outputs=x)


# =============================
# DISCRIMINATOR
# =============================
def Discriminator():
    ini = Input(shape=[256, 256, 3])
    gen = Input(shape=[256, 256, 3])

    x = Concatenate()([ini, gen])

    initializer = tf.random_normal_initializer(0, 0.02)

    down1 = downsample(64, apply_batchnorm=False)(x)
    down2 = downsample(128)(down1)
    down3 = downsample(256)(down2)

    last = Conv2D(1, 4, strides=1, padding="same", kernel_initializer=initializer)(
        down3
    )

    return Model(inputs=[ini, gen], outputs=last)


# Instanciar modelos
generator = Generator()
discriminator = Discriminator()

# =============================
# LOSS Y OPTIMIZADORES
# =============================
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
LAMBDA = 100


def discriminator_loss(real, generated):
    real_loss = loss_object(tf.ones_like(real), real)
    fake_loss = loss_object(tf.zeros_like(generated), generated)
    return real_loss + fake_loss


def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    return gan_loss + (LAMBDA * l1_loss)


generator_optimizer = Adam(2e-4, beta_1=0.5)
discriminator_optimizer = Adam(2e-4, beta_1=0.5)

# =============================
# CHECKPOINTS
# =============================
checkpoint_prefix = os.path.join(CKPATH, "ckpt")
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,
    discriminator_optimizer=discriminator_optimizer,
    generator=generator,
    discriminator=discriminator,
)

if os.path.exists(CKPATH):
    checkpoint.restore(tf.train.latest_checkpoint(CKPATH))


# =============================
# GENERAR IMÁGENES
# =============================
def generate_images(model, test_input, tar):
    prediction = model(test_input, training=False)

    plt.figure(figsize=(12, 12))
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ["Input", "Target", "Generated"]

    for i in range(3):
        plt.subplot(1, 3, i + 1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis("off")
    plt.show()


# =============================
# TRAIN STEP
# =============================
@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:

        gen_output = generator(input_image, training=True)

        disc_real = discriminator([input_image, target], training=True)
        disc_fake = discriminator([input_image, gen_output], training=True)

        gen_loss = generator_loss(disc_fake, gen_output, target)
        disc_loss = discriminator_loss(disc_real, disc_fake)

    generator_grads = gen_tape.gradient(gen_loss, generator.trainable_variables)
    discriminator_grads = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(generator_grads, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(discriminator_grads, discriminator.trainable_variables)
    )


# =============================
# TRAIN LOOP
# =============================
def train(dataset, epochs):
    for epoch in range(epochs):
        print("Epoch:", epoch)

        for input_image, target in dataset:
            train_step(input_image, target)

        clear_output(wait=True)
        for inp, tar in test_dataset.take(1):
            generate_images(generator, inp, tar)

        if (epoch + 1) % 25 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)


train(train_dataset, 5)
