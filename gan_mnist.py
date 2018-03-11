import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for 
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)

lib.print_model_settings(locals().copy())

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear', 
        n_in, 
        n_out, 
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def Generator(n_samples, noise, idx, bijector_count=0, invert=False, noise_global=True):
    prefix = "Generator" + str(idx)

    output = lib.ops.linear.Linear(prefix + '.Input', 128, 4*4*4*DIM, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm(prefix + '.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 4*DIM, 4, 4])

    output = lib.ops.deconv2d.Deconv2D(prefix + '.2', 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm(prefix + '.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = output[:,:,:7,:7]

    output = lib.ops.deconv2d.Deconv2D(prefix + '.3', 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm(prefix + '.BN3', [0,2,3], output)
    output = tf.nn.relu(output)
   
    output_mean = tf.nn.sigmoid(lib.ops.deconv2d.Deconv2D(prefix + '.OutputMean', DIM, 1, 5, output))
    if not noise_global:
        output_scale = 1e-6 + tf.nn.softplus(lib.ops.deconv2d.Deconv2D(prefix + '.OutputStdDev', DIM, 1, 5, output))
    else:
        output_scale = tf.scalar_mul(1e-6 + tf.nn.softplus(lib.param(prefix + '.OutputStdDev', 0, dtype='float32')), tf.ones_like(output_mean))

    output_mean = tf.reshape(output_mean, [-1, OUTPUT_DIM])
    output_scale = tf.reshape(output_scale, [-1, OUTPUT_DIM])

    #output_distribution = tf.distributions.Normal(name=prefix + '.Noise', loc=output_mean, scale=output_scale)
    output_distribution = tf.contrib.distributions.MultivariateNormalDiag(name=prefix + '.Noise', loc=output_mean, scale_diag=output_scale)

    if bijector_count == 0:
        output = output_distribution.sample([])
        #eval_fun = lambda value: tf.reduce_sum(output_distribution.log_prob(value), 1)
        eval_fun = lambda value: output_distribution.log_prob(value)

        return output, output_mean, output_scale, eval_fun

    with tf.variable_scope(prefix, reuse=tf.AUTO_REUSE):
        bijectors = []

        for i in range(bijector_count):
            bijectors.append(
                    tf.contrib.distributions.bijectors.MaskedAutoregressiveFlow(
                            shift_and_log_scale_fn=tf.contrib.distributions.bijectors.masked_autoregressive_default_template(
                                hidden_layers=[OUTPUT_DIM, OUTPUT_DIM],
                                name=prefix + ".Bijector." + str(i) + ".Template"
                            )
                    )
                )
    
            random_permutation = tf.get_variable(
                                            prefix + ".Bijector." + str(i) + ".RandomPermutation", 
                                            initializer=np.random.permutation(OUTPUT_DIM), 
                                            trainable=False,
                                            dtype=tf.int64
                                        )
            bijectors.append(
                    tf.contrib.distributions.bijectors.Permute(
                        permutation=random_permutation,
                        name=prefix + ".PermuteBijector." + str(i), 
                    )
                )
                
        flow_bijector = tf.contrib.distributions.bijectors.Chain(list(reversed(bijectors[:-1])))
        distribution = tf.contrib.distributions.TransformedDistribution(
                            name=prefix + ".FlowBijector",
                            distribution=output_distribution,
                            bijector = flow_bijector if not invert else tf.contrib.distributions.bijectors.Invert(flow_bijector)
                        )

        return distribution.sample(), tf.zeros_like(output_mean), tf.zeros_like(output_mean), lambda value: distribution.log_prob(value)

def Encoder(output, idx, bijector_count=0, invert=False, noise_global=True):
    prefix = "Encoder" + str(idx)

    output = tf.reshape(output, [-1, 1, 28, 28])

    output = lib.ops.conv2d.Conv2D(prefix + '.1',1,DIM,5,output,stride=2)
    output = tf.nn.relu(output)

    output = lib.ops.conv2d.Conv2D(prefix + '.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm(prefix + '.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.conv2d.Conv2D(prefix + '.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm(prefix + '.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])

    output = lib.ops.linear.Linear(prefix + '.4', 4*4*4*DIM, DIM, output)
    ouptut = tf.nn.relu(output)

    output_mean = lib.ops.linear.Linear(prefix + '.OutputMean', DIM, 128, output)
    if not noise_global:
        output_scale = 1e-6 + tf.nn.softplus(lib.ops.linear.Linear(prefix + '.OutputMean', DIM, 128, output)) 
    else:
        output_scale = tf.scalar_mul(1e-6 + tf.nn.softplus(lib.param(prefix + ".OutputStdDev", 0, dtype='float32')), tf.ones_like(output_mean))
    output_distribution = tf.contrib.distributions.MultivariateNormalDiag(name=prefix + '.Noise', loc=output_mean, scale_diag=output_scale)

    if bijector_count == 0:
        output = output_distribution.sample([])
        eval_fun = lambda value: output_distribution.log_prob(value)

        return output, output_mean, output_scale, eval_fun

    with tf.variable_scope(prefix, reuse=tf.AUTO_REUSE):
        bijectors = []

        for i in range(bijector_count):
            bijectors.append(
                    tf.contrib.distributions.bijectors.MaskedAutoregressiveFlow(
                            shift_and_log_scale_fn=tf.contrib.distributions.bijectors.masked_autoregressive_default_template(
                                hidden_layers=[128, 128],
                                name=prefix + ".Bijector." + str(i) + ".Template"
                            )
                    )
                )
    
            random_permutation = tf.get_variable(
                                            prefix + ".Bijector." + str(i) + ".RandomPermutation", 
                                            initializer=np.random.permutation(128), 
                                            trainable=False,
                                            dtype=tf.int64
                                        )
            bijectors.append(
                    tf.contrib.distributions.bijectors.Permute(
                        permutation=random_permutation,
                        name=prefix + ".PermuteBijector." + str(i), 
                    )
                )
                
        flow_bijector = tf.contrib.distributions.bijectors.Chain(list(reversed(bijectors[:-1])))
        distribution = tf.contrib.distributions.TransformedDistribution(
                            name=prefix + ".FlowBijector",
                            distribution=output_distribution,
                            bijector = flow_bijector if not invert else tf.contrib.distributions.bijectors.Invert(flow_bijector)
                        )

        return distribution.sample(), tf.zeros_like(output_mean), tf.zeros_like(output_mean), lambda value: distribution.log_prob(value)

def LogRatio(output, noise, idx):
    prefix = 'LogRatio' + str(idx)

    output = tf.reshape(output, [-1, 1, 28, 28])

    output = lib.ops.conv2d.Conv2D(prefix + '.X.1',1,DIM,5,output,stride=2)
    output = tf.nn.relu(output)

    output = lib.ops.conv2d.Conv2D(prefix + '.X.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm(prefix + '.X.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = lib.ops.conv2d.Conv2D(prefix + '.X.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm(prefix + '.X.BN3', [0,2,3], output)
    output = tf.nn.relu(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])

    output = lib.ops.linear.Linear(prefix + '.X.4', 4*4*4*DIM, DIM, output)
    ouptut = tf.nn.relu(output)

    noise = lib.ops.linear.Linear(prefix + '.Z.1', 128, DIM, noise)
    noise = tf.nn.relu(noise)

    noise = lib.ops.linear.Linear(prefix + '.Z.2', DIM, DIM, noise)
    noise = tf.nn.relu(noise)

    joint = lib.ops.linear.Linear(prefix + '.XZ.1', 2*DIM, 2*DIM, tf.concat([output, noise],1))
    joint = tf.nn.relu(joint)

    joint = lib.ops.linear.Linear(prefix + '.XZ.2', 2*DIM, 2*DIM, joint)
    joint = tf.nn.relu(joint)

    joint = lib.ops.linear.Linear(prefix + '.XZ.2', 1, 2*DIM, joint)

    return tf.cast(tf.reshape(joint, [-1]),dtype=tf.float64)

def DiscriminatorBasic(inputs):
    output = tf.reshape(inputs, [-1, 1, 28, 28])

    output = lib.ops.conv2d.Conv2D('Discriminator.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4*4*4*DIM])
    
    output = lib.ops.linear.Linear('Discriminator.Output', 4*4*4*DIM, 1, output)
    
    return tf.reshape(output, [-1])

GeneratorFun = lambda data, noise, idx: Generator(data, noise, idx, bijector_count=0, invert=True, noise_global=True)
ModelGeneratorFun = lambda data, noise, idx: Generator(data, noise, idx, bijector_count=0, invert=False, noise_global=True)
EncoderFun = lambda data, idx: Encoder(data, idx, bijector_count=3, invert=False, noise_global=True) 
ModelEncoderFun = lambda data, idx: Encoder(data, idx, bijector_count=3, invert=True, noise_global=True) 
DiscriminatorFun = lambda data: DiscriminatorBasic(data)
LogRatioFun = lambda data, noise, idx: LogRatio(data, noise, idx)

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_noise_distribution = tf.distributions.Normal(loc=tf.zeros([128]), scale=tf.ones([128]))
fake_noise = fake_noise_distribution.sample([BATCH_SIZE])
fake_gen, fake_gen_mean, fake_gen_scale, fake_gen_eval_fun = GeneratorFun(BATCH_SIZE, fake_noise, 0)
fake_data = fake_gen

enc1_idx = 1
fake_enc1, fake_enc1_mean, fake_enc1_scale, fake_enc1_eval_fun = EncoderFun(fake_data, enc1_idx)
fake_enc1_log_ratio1 = LogRatioFun(fake_data, fake_noise, enc1_idx)
fake_enc1_log_ratio2 = LogRatioFun(fake_data, fake_enc1, enc1_idx)

#enc2_idx = 2
#fake_enc2, fake_enc2_mean, fake_enc2_scale, fake_enc2_eval_fun = ModelEncoderFun(fake_data, enc2_idx)
#fake_enc2_gen, fake_enc2_gen_mean, fake_enc2_gen_scale, fake_enc2_gen_eval_fun = ModelGeneratorFun(BATCH_SIZE, fake_enc2, 0)
#fake_enc2_log_ratio1 = LogRatioFun(fake_data, fake_noise, enc2_idx)
#fake_enc2_log_ratio2 = LogRatioFun(fake_data, fake_enc2, enc2_idx)

enc3_idx = 3
model_target = fake_data
model_noise = fake_noise_distribution.sample([BATCH_SIZE])
model_gen, model_gen_mean, model_gen_scale, model_gen_eval_fun = ModelGeneratorFun(BATCH_SIZE, model_noise, enc3_idx)
fake_enc3, fake_enc3_mean, fake_enc3_scale, fake_enc3_eval_fun = ModelEncoderFun(model_target, enc3_idx)
fake_enc3_gen, fake_enc3_gen_mean, fake_enc3_gen_scale, fake_enc3_gen_eval_fun = ModelGeneratorFun(BATCH_SIZE, fake_enc3, enc3_idx)
fake_enc3_log_ratio1 = LogRatioFun(model_gen, model_noise, enc3_idx)
fake_enc3_log_ratio2 = LogRatioFun(model_target, fake_enc3, enc3_idx)

disc_real = DiscriminatorFun(real_data)
disc_fake = DiscriminatorFun(fake_data)

gen_params = filter(lambda var: var.name.startswith('Generator0'), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
model_params = filter(lambda var: var.name.startswith('Generator' + str(enc3_idx)), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
enc1_params = filter(lambda var: var.name.startswith('Encoder' + str(enc1_idx)), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
enc1_log_ratio_params = filter(lambda var: var.name.startswith('LogRatio' + str(enc1_idx)), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
#enc2_params = filter(lambda var: var.name.startswith('Encoder' + str(enc2_idx)), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
#enc2_log_ratio_params = filter(lambda var: var.name.startswith('LogRatio' + str(enc2_idx)), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
gen3_params = filter(lambda var: var.name.startswith('Generator3'), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
enc3_params = filter(lambda var: var.name.startswith('Encoder' + str(enc3_idx)), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
enc3_log_ratio_params = filter(lambda var: var.name.startswith('LogRatio' + str(enc3_idx)), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
disc_params = filter(lambda var: var.name.startswith("Discriminator"), tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = -tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var, 
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1], 
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(DiscriminatorFun(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    #gradient_penalty = tf.reduce_mean((tf.maximum(tf.zeros_like(slopes),slopes-OUTPUT_DIM/10))**2)
    gradient_penalty = tf.reduce_mean((slopes-1)**2)
    disc_regularization_wgan = LAMBDA*gradient_penalty

    entropy1 = -tf.reduce_mean(
                            fake_gen_eval_fun(fake_data) 
                            + tf.reduce_sum(fake_noise_distribution.log_prob(fake_noise),1) #- 0.5 * tf.reduce_sum(fake_noise**2,1)
                            - fake_enc1_eval_fun(fake_noise)
                        )
    entropy1_adjustment = tf.reduce_mean(fake_enc1_log_ratio1)

    #entropy2 = -tf.reduce_mean(
    #                        fake_enc2_gen_eval_fun(fake_data) 
    #                        + tf.reduce_sum(fake_noise_distribution.log_prob(fake_enc2),1) #- 0.5 * tf.reduce_sum(fake_enc2**2,1)
    #                        - fake_enc2_eval_fun(fake_enc2)
    #                    )

    #entropy2_adjustment = tf.reduce_mean(fake_enc2_log_ratio2)

    entropy3 = -tf.reduce_mean(
                            fake_enc3_gen_eval_fun(model_target) 
                            + tf.reduce_sum(fake_noise_distribution.log_prob(fake_enc3),1) #- 0.5 * tf.reduce_sum(fake_enc3**2,1)
                            - fake_enc3_eval_fun(fake_enc3)
                        )

    entropy3_adjustment = tf.reduce_mean(fake_enc3_log_ratio2)

    wgan_steps = 0
    #CRITIC_ITERS=1
    
    global_step = tf.Variable(0, trainable=False, name='global_step')

    enc1_log_ratio_cost = -0.5 * (tf.reduce_mean(tf.log(tf.clip_by_value(tf.sigmoid(fake_enc1_log_ratio1), 1e-10, 1.0))) + tf.reduce_mean(tf.log(tf.clip_by_value(1 - tf.sigmoid(fake_enc1_log_ratio2), 1e-10, 1.0))))
    #enc2_log_ratio_cost = -0.5 * (tf.reduce_mean(tf.log(tf.clip_by_value(tf.sigmoid(fake_enc2_log_ratio1), 1e-10, 1.0))) + tf.reduce_mean(tf.log(tf.clip_by_value(1 - tf.sigmoid(fake_enc2_log_ratio2), 1e-10, 1.0))))
    enc3_log_ratio_cost = -0.5 * (tf.reduce_mean(tf.log(tf.clip_by_value(tf.sigmoid(fake_enc3_log_ratio1), 1e-10, 1.0))) + tf.reduce_mean(tf.log(tf.clip_by_value(1 - tf.sigmoid(fake_enc3_log_ratio2), 1e-10, 1.0))))

    #disc_regularization_final = tf.constant(0.0)
    #disc_regularization_final = .01*tf.reduce_mean(disc_fake**2) + tf.reduce_mean(disc_real**2)
    #disc_regularization_final = 10*tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1.0), [V for V in disc_params if len(V.shape) > 1])
    disc_regularization_final = disc_regularization_wgan
    disc_cost += tf.cond(tf.less(global_step, wgan_steps), lambda: disc_regularization_wgan, lambda: disc_regularization_final)

    gen_regularization_final = 0.0
    gen_regularization_final += tf.reduce_mean(-fake_enc1_eval_fun(fake_noise))
    #gen_regularization_final += -entropy3
    gen_cost += tf.cond(tf.less(global_step, wgan_steps), lambda: tf.constant(0.0), lambda: gen_regularization_final)

    model_cost = entropy3
   
    enc_cost = 0
    #enc_cost += entropy2
    enc_cost += -entropy1

    #enc_log_ratio_cost_regularization = 1e-10 * tf.contrib.layers.apply_regularization(tf.contrib.layers.l2_regularizer(1.0), [V for V in enc3_log_ratio_params if len(V.shape) > 1])
    #enc_log_ratio_cost_regularization = tf.reduce_mean(tf.abs(fake_enc3_log_ratio1)) + tf.reduce_mean(tf.abs(fake_enc3_log_ratio2))
    enc_log_ratio_cost = 0
    enc_log_ratio_cost += enc1_log_ratio_cost
    #enc_log_ratio_cost += enc2_log_ratio_cost
    enc_log_ratio_cost += enc3_log_ratio_cost

    fake_gen_scale_sum = tf.reduce_mean(tf.reduce_sum(fake_gen_scale, -1))
    #fake_enc1_scale_sum = tf.reduce_mean(tf.reduce_sum(fake_enc1_scale, -1))
    #fake_enc2_scale_sum = tf.reduce_mean(tf.reduce_sum(fake_enc2_scale, -1))
    #fake_enc2_gen_scale_sum = tf.reduce_mean(tf.reduce_sum(fake_enc2_gen_scale, -1))
    fake_enc3_gen_scale_sum = tf.reduce_mean(tf.reduce_sum(fake_enc3_gen_scale, -1))

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)
    model_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5,
        beta2=0.9
    ).minimize(model_cost, var_list=model_params + enc3_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5, 
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)
    enc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5,
        beta2=0.9
    #).minimize(enc_cost, var_list=enc1_params + enc2_params, global_step=global_step)
    ).minimize(enc_cost, var_list=enc1_params, global_step=global_step)
    enc_log_ratio_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, 
        beta1=0.5,
        beta2=0.9
    #).minimize(enc_log_ratio_cost, var_list=enc1_log_ratio_params + enc2_log_ratio_params + enc3_log_ratio_params, global_step=global_step)
    ).minimize(enc_log_ratio_cost, var_list=enc1_log_ratio_params + enc3_log_ratio_params, global_step=global_step)

    clip_disc_weights = None

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake, 
        tf.ones_like(disc_fake)
    ))

    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake, 
        tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_real, 
        tf.ones_like(disc_real)
    ))
    disc_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4, 
        beta1=0.5
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

# For saving samples
fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples = GeneratorFun(128, fixed_noise, 0)[1]
fixed_noise_samples_enc1_gen = GeneratorFun(128,EncoderFun(fixed_noise_samples, enc1_idx)[0], 0)[1]
#fixed_noise_samples_enc2_gen = ModelGeneratorFun(128,EncoderFun(fixed_noise_samples, enc2_idx)[0], 0)[1]
fixed_noise_samples_enc3_gen = ModelGeneratorFun(128,ModelEncoderFun(fixed_noise_samples, enc3_idx)[0], 3)[1]
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    lib.save_images.save_images(
        samples.reshape((128, 28, 28)), 
        'samples_{}.png'.format(frame)
    )
    samples_enc1_gen = session.run(fixed_noise_samples_enc1_gen)
    lib.save_images.save_images(
        samples_enc1_gen.reshape((128, 28, 28)), 
        'samples_enc1_gen{}.png'.format(frame)
    )
    #samples_enc2_gen = session.run(fixed_noise_samples_enc2_gen)
    #lib.save_images.save_images(
    #    samples_enc2_gen.reshape((128, 28, 28)), 
    #    'samples_enc2_gen{}.png'.format(frame)
    #)
    samples_enc3_gen = session.run(fixed_noise_samples_enc3_gen)
    lib.save_images.save_images(
        samples_enc3_gen.reshape((128, 28, 28)), 
        'samples_enc3_gen{}.png'.format(frame)
    )

# Dataset iterator
train_gen, dev_gen, test_gen = lib.mnist.load(BATCH_SIZE, BATCH_SIZE)
def inf_train_gen():
    while True:
        for images,targets in train_gen():
            yield images

# Train loop
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as session:

    session.run(tf.initialize_all_variables())

    gen = inf_train_gen()

    for iteration in xrange(ITERS):
        start_time = time.time()

        #if iteration > 0:
        #    _ = session.run(gen_train_op)

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in xrange(disc_iters):
            _data = gen.next()

            if i == 0:
                #_enc_cost, _ = session.run([enc_cost, enc_train_op])
                #_entropy1, _entropy1_adjustment, _entropy2, _entropy2_adjustment, _entropy3, _entropy3_adjustment, _fake_gen_scale_sum, _fake_enc1_scale_sum, _fake_enc2_scale_sum, _fake_enc2_gen_scale_sum, _gen_cost, _model_cost, _enc_cost, _enc1_log_ratio_cost, _enc2_log_ratio_cost, _enc3_log_ratio_cost, _disc_cost, _, _, _, _, _, _global_step = session.run([entropy1, entropy1_adjustment, entropy2, entropy2_adjustment, entropy3, entropy3_adjustment, fake_gen_scale_sum, fake_enc1_scale_sum, fake_enc2_scale_sum, fake_enc2_gen_scale_sum, gen_cost, model_cost, enc_cost, enc1_log_ratio_cost, enc2_log_ratio_cost, enc3_log_ratio_cost, disc_cost, gen_train_op, model_train_op, enc_train_op, enc_log_ratio_train_op, disc_train_op, global_step], feed_dict={real_data: _data})
                _entropy1, _entropy1_adjustment, _entropy3, _entropy3_adjustment, _fake_gen_scale_sum, _fake_enc3_gen_scale_sum, _gen_cost, _model_cost, _enc_cost, _enc1_log_ratio_cost, _enc3_log_ratio_cost, _disc_cost, _, _, _, _, _, _global_step = session.run([entropy1, entropy1_adjustment, entropy3, entropy3_adjustment, fake_gen_scale_sum, fake_enc3_gen_scale_sum, gen_cost, model_cost, enc_cost, enc1_log_ratio_cost, enc3_log_ratio_cost, disc_cost, gen_train_op, model_train_op, enc_train_op, enc_log_ratio_train_op, disc_train_op, global_step], feed_dict={real_data: _data})

            #_disc_cost, _ = session.run(
            #    [disc_cost, disc_train_op],
            #    feed_dict={real_data: _data}
            #)
            #if clip_disc_weights is not None:
            #    _ = session.run(clip_disc_weights)

        #print "gen_cost=%s model_cost=%s enc_cost=%s enc_log_ratio_cost=%s/%s/%s disc_cost=%s entropy1=%s (%s) entropy2=%s (%s) entropy3=%s (%s) time=%s" % (_gen_cost, _model_cost, _enc_cost, _enc1_log_ratio_cost, _enc2_log_ratio_cost, _enc3_log_ratio_cost, _disc_cost, _entropy1,  _entropy1_adjustment, _entropy2, _entropy2_adjustment, _entropy3, _entropy3_adjustment, time.time() - start_time)
        print "gen_cost=%s model_cost=%s enc_cost=%s enc_log_ratio_cost=%s/%s disc_cost=%s entropy1=%s (%s) entropy3=%s (%s) time=%s" % (_gen_cost, _model_cost, _enc_cost, _enc1_log_ratio_cost, _enc3_log_ratio_cost, _disc_cost, _entropy1,  _entropy1_adjustment, _entropy3, _entropy3_adjustment, time.time() - start_time)
        #print "train_gen_cost=%s time=%s" % (_gen_cost, time.time() - start_time)
        if iteration % 100 == 0:
            lib.plot.plot('train gen cost', _gen_cost)
            lib.plot.plot('train model cost', _model_cost)
            lib.plot.plot('train enc cost', _enc_cost)
            lib.plot.plot('train enc1 log ratio cost', _enc1_log_ratio_cost)
            #lib.plot.plot('train enc2 log ratio cost', _enc2_log_ratio_cost)
            lib.plot.plot('train disc cost', _disc_cost)
            lib.plot.plot('train entropy1', _entropy1)
            lib.plot.plot('train entropy1_ajdustment', _entropy1_adjustment)
            #lib.plot.plot('train entropy2', _entropy2)
            #lib.plot.plot('train entropy2_ajdustment', _entropy2_adjustment)
            lib.plot.plot('train entropy3', _entropy3)
            lib.plot.plot('train entropy3_ajdustment', _entropy3_adjustment)
            lib.plot.plot('train fake_gen_scale_sum', _fake_gen_scale_sum)
            #lib.plot.plot('train fake_enc1_scale_sum', _fake_enc1_scale_sum)
            #lib.plot.plot('train fake_enc2_scale_sum', _fake_enc2_scale_sum)
            #lib.plot.plot('train fake_enc2_gen_scale_sum', _fake_enc2_gen_scale_sum)
            lib.plot.plot('train fake_enc3_gen_scale_sum', _fake_enc3_gen_scale_sum)
            lib.plot.plot('global_step', _global_step)
            lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if iteration % 100 == 99:
            dev_disc_costs = []
            for images,_ in dev_gen():
                _dev_disc_cost = session.run(
                    disc_cost, 
                    feed_dict={real_data: images}
                )
                dev_disc_costs.append(_dev_disc_cost)
            lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            generate_image(iteration, _data)

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % 100 == 99) or True:
            lib.plot.flush()

        lib.plot.tick()
