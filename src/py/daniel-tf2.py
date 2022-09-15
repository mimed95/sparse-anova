# Single-level algorithm using the Milstein scheme
# For more detailed explanations of the training and model parameters
# see Gerstner et al. "Multilevel Monte Carlo learning." arXiv preprint arXiv:2102.08734 (2021).

# Packages
#%tensorflow_version 1.x
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import time
from tensorflow.python.ops import init_ops
from tensorflow.compat.v1.keras import initializers
from tensorflow.python.training.moving_averages import assign_moving_average

# Basic network framework according to Beck, Christian, et al. "Solving the Kolmogorov PDE by means of deep learning." Journal of Scientific Computing 88.3 (2021): 1-28.
def neural_net(
    x, xi_approx, neurons, is_training, name, mv_decay=0.9, dtype=tf.float32
):
    def approx_test():
        return xi_approx

    def approx_learn():
        return x

    x = tf.cond(pred=is_training, true_fn=approx_learn, false_fn=approx_test)

    def _batch_normalization(_x):
        beta = tf.compat.v1.get_variable(
            "beta", [_x.get_shape()[-1]], dtype, init_ops.zeros_initializer()
        )
        gamma = tf.compat.v1.get_variable(
            "gamma", [_x.get_shape()[-1]], dtype, init_ops.ones_initializer()
        )
        mv_mean = tf.compat.v1.get_variable(
            "mv_mean",
            [_x.get_shape()[-1]],
            dtype,
            init_ops.zeros_initializer(),
            trainable=False,
        )
        mv_variance = tf.compat.v1.get_variable(
            "mv_variance",
            [_x.get_shape()[-1]],
            dtype,
            init_ops.ones_initializer(),
            trainable=False,
        )
        mean, variance = tf.nn.moments(x=_x, axes=[0], name="moments")
        tf.compat.v1.add_to_collection(
            tf.compat.v1.GraphKeys.UPDATE_OPS,
            assign_moving_average(mv_mean, mean, mv_decay, True),
        )
        tf.compat.v1.add_to_collection(
            tf.compat.v1.GraphKeys.UPDATE_OPS,
            assign_moving_average(mv_variance, variance, mv_decay, False),
        )
        mean, variance = tf.cond(
            pred=is_training,
            true_fn=lambda: (mean, variance),
            false_fn=lambda: (mv_mean, mv_variance),
        )
        return tf.nn.batch_normalization(_x, mean, variance, beta, gamma, 1e-6)

    def _layer(_x, out_size, activation_fn):
        w = tf.compat.v1.get_variable(
            "weights",
            [_x.get_shape().as_list()[-1], out_size],
            dtype,
            initializers.glorot_normal(),
        )
        return activation_fn(_batch_normalization(tf.matmul(_x, w)))

    with tf.compat.v1.variable_scope(name):
        x = _batch_normalization(x)
        for i in range(len(neurons)):
            with tf.compat.v1.variable_scope("layer_%i_" % (i + 1)):
                x = _layer(
                    x, neurons[i], tf.nn.tanh if i < len(neurons) - 1 else tf.identity
                )
    return x


# Basic network framework according to Beck, Christian, et al. "Solving the Kolmogorov PDE by means of deep learning." Journal of Scientific Computing 88.3 (2021): 1-28.
# Minor adjustments to file output and in lines 108-115 changed to exponential decay


def train_and_test(
    xi,
    xi_approx,
    x_sde,
    phi,
    u_reference,
    neurons,
    train_steps,
    mc_rounds,
    mc_freq,
    file_name,
    dtype=tf.float32,
):
    def _approximate_errors():
        lr, gs = sess.run([learning_rate, global_step])
        l1_err, l2_err, li_err = 0.0, 0.0, 0.0
        rel_l1_err, rel_l2_err, rel_li_err = 0.0, 0.0, 0.0
        for _ in range(mc_rounds):
            (
                plot_xi,
                plot_approx,
                plot_ref,
                l1,
                l2,
                li,
                rl1,
                rl2,
                rli,
                appr,
                ref,
            ) = sess.run(
                [
                    xi_approx,
                    u_approx,
                    u_reference,
                    err_l_1,
                    err_l_2,
                    err_l_inf,
                    rel_err_l_1,
                    rel_err_l_2,
                    rel_err_l_inf,
                    approx,
                    reference,
                ],
                feed_dict={is_training: False},
            )
            l1_err, l2_err, li_err = (l1_err + l1, l2_err + l2, np.maximum(li_err, li))
            rel_l1_err, rel_l2_err, rel_li_err = (
                rel_l1_err + rl1,
                rel_l2_err + rl2,
                np.maximum(rel_li_err, rli),
            )
        l1_err, l2_err = l1_err / mc_rounds, np.sqrt(l2_err / mc_rounds)
        rel_l1_err, rel_l2_err = rel_l1_err / mc_rounds, np.sqrt(rel_l2_err / mc_rounds)
        t_mc = time.time()

        file_out.write(
            "%i, %f, %f, %f, %f \n"
            % (gs, li_err, lr, t1_train - t0_train, t_mc - t1_train)
        )
        file_out.flush()

    t0_train = time.time()
    is_training = tf.compat.v1.placeholder(tf.bool, [])
    u_approx = neural_net(xi, xi_approx, neurons, is_training, "u_approx", dtype=dtype)
    loss = tf.reduce_mean(input_tensor=tf.math.squared_difference(u_approx, phi))

    approx = tf.reduce_mean(input_tensor=u_approx)
    reference = tf.reduce_mean(input_tensor=u_reference)
    err = tf.abs(u_approx - u_reference)
    err_l_1 = tf.reduce_mean(input_tensor=err)
    err_l_2 = tf.reduce_mean(input_tensor=err**2)
    err_l_inf = tf.reduce_max(input_tensor=err)
    rel_err = err / tf.maximum(u_reference, 1e-4)
    rel_err_l_1 = tf.reduce_mean(input_tensor=rel_err)
    rel_err_l_2 = tf.reduce_mean(input_tensor=rel_err**2)
    rel_err_l_inf = tf.reduce_max(input_tensor=rel_err)

    lr = 0.01
    step_rate = 40000
    decay = 0.1

    global_step = tf.Variable(0, trainable=False)
    increment_global_step = tf.compat.v1.assign(global_step, global_step + 1)
    learning_rate = tf.compat.v1.train.exponential_decay(
        lr, global_step, step_rate, decay, staircase=True
    )
    optimizer = tf.compat.v1.train.AdamOptimizer(
        learning_rate=learning_rate, epsilon=0.01
    )

    update_ops = tf.compat.v1.get_collection(
        tf.compat.v1.GraphKeys.UPDATE_OPS, "u_approx"
    )
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step)

    file_out = open(file_name, "w")
    file_out.write("step, li_err, learning_rate, time_train, time_test \n ")

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        for step in range(train_steps):
            if step % mc_freq == 0:
                print(step)
                t1_train = time.time()
                _approximate_errors()
                t0_train = time.time()
            sess.run(train_op, feed_dict={is_training: True})
        t1_train = time.time()
        _approximate_errors()

    file_out.close()


# Model and training parameter specification
for i in range(1, 2):
    tf.compat.v1.reset_default_graph()
    tf.compat.v1.random.set_random_seed(i)
    with tf.compat.v1.Session() as sess:
        dtype = tf.float32

        # Set network and training parameter
        batch_size = 125000
        batch_size_approx = batch_size
        N, d = 128, 5
        neurons = [50, 50, 1]
        train_steps = 150000
        mc_rounds, mc_freq = 100, 5000
        mc_samples_ref, mc_rounds_ref = 1, 1
        N_l = 1

        # Define training and test interval
        s_0_l = 80.0
        s_0_r = 120.0
        sigma_l = 0.1
        sigma_r = 0.2
        mu_l = 0.02
        mu_r = 0.05
        T_l = 0.9
        T_r = 1.0
        K_l = 109.0
        K_r = 110.0
        s_0_l_approx = 80.4
        s_0_r_approx = 119.6
        sigma_l_approx = 0.11
        sigma_r_approx = 0.19
        mu_l_approx = 0.03
        mu_r_approx = 0.04
        T_l_approx = 0.91
        T_r_approx = 0.99
        K_l_approx = 109.1
        K_r_approx = 109.9
        s0 = tf.random.uniform((batch_size, 1), minval=s_0_l, maxval=s_0_r, dtype=dtype)
        sigma = tf.random.uniform(
            (batch_size, 1), minval=sigma_l, maxval=sigma_r, dtype=dtype
        )
        mu = tf.random.uniform((batch_size, 1), minval=mu_l, maxval=mu_r, dtype=dtype)
        T = tf.random.uniform((batch_size, 1), minval=T_l, maxval=T_r, dtype=dtype)
        K = tf.random.uniform((batch_size, 1), minval=K_l, maxval=K_r, dtype=dtype)
        s0_approx = tf.random.uniform(
            (batch_size_approx, 1),
            minval=s_0_l_approx,
            maxval=s_0_r_approx,
            dtype=dtype,
        )
        sigma_approx = tf.random.uniform(
            (batch_size_approx, 1),
            minval=sigma_l_approx,
            maxval=sigma_r_approx,
            dtype=dtype,
        )
        mu_approx = tf.random.uniform(
            (batch_size_approx, 1), minval=mu_l_approx, maxval=mu_r_approx, dtype=dtype
        )
        T_approx = tf.random.uniform(
            (batch_size_approx, 1), minval=T_l_approx, maxval=T_r_approx, dtype=dtype
        )
        K_approx = tf.random.uniform(
            (batch_size_approx, 1), minval=K_l_approx, maxval=K_r_approx, dtype=dtype
        )

        xi = tf.reshape(tf.stack([s0, sigma, mu, T, K], axis=2), (batch_size, d))
        xi_approx = tf.reshape(
            tf.stack([s0_approx, sigma_approx, mu_approx, T_approx, K_approx], axis=2),
            (batch_size_approx, d),
        )

        # Closed solution as reference
        tfd = tfp.distributions
        dist = tfd.Normal(loc=tf.cast(0.0, tf.float32), scale=tf.cast(1.0, tf.float32))
        d1 = tf.math.divide(
            (
                tf.math.log(tf.math.divide(s0_approx, K_approx))
                + (mu_approx + 0.5 * sigma_approx**2) * T_approx
            ),
            (sigma_approx * tf.sqrt(T_approx)),
        )
        d2 = tf.math.divide(
            (
                tf.math.log(tf.math.divide(s0_approx, K_approx))
                + (mu_approx - 0.5 * sigma_approx**2) * T_approx
            ),
            (sigma_approx * tf.sqrt(T_approx)),
        )
        u_reference = tf.multiply(s0_approx, (dist.cdf(d1))) - K_approx * tf.exp(
            -mu_approx * T_approx
        ) * (dist.cdf(d2))

    # European option
    def phi(x, sigma, mu, T, K, axis=1):
        payoffcoarse = tf.exp(-mu * T) * tf.maximum(x - K, 0.0)
        return payoffcoarse

    # Milstein scheme
    def sde_body(idx, s, sigma, mu, T, K, samples):
        h = T / N
        z = tf.random.normal(shape=(samples, batch_size, 1), stddev=1.0, dtype=dtype)
        s = (
            s
            + mu * s * h
            + sigma * s * tf.sqrt(h) * z
            + 0.5 * sigma * s * sigma * ((tf.sqrt(h) * z) ** 2 - h)
        )
        return tf.add(idx, 1), s, sigma, mu, T, K

    # Monte Carlo loop
    def mc_body(idx, p):
        _, _x, _sigma, _mu, _T, _K = tf.while_loop(
            cond=lambda _idx, s, sigma, mu, T, K: _idx < N,
            body=lambda _idx, s, sigma, mu, T, K: sde_body(
                _idx, s, sigma, mu, T, K, mc_samples_ref
            ),
            loop_vars=loop_var_mc,
        )
        return idx + 1, p + tf.reduce_mean(
            input_tensor=phi(_x, _sigma, _mu, _T, _K, 2), axis=0
        )

    loop_var_mc = (
        tf.constant(0),
        tf.ones((mc_samples_ref, batch_size, 1), dtype) * s0,
        tf.ones((mc_samples_ref, batch_size, 1), dtype) * sigma,
        tf.ones((mc_samples_ref, batch_size, 1), dtype) * mu,
        tf.ones((mc_samples_ref, batch_size, 1), dtype) * T,
        tf.ones((mc_samples_ref, batch_size, 1), dtype) * K,
    )
    _, u = tf.while_loop(
        cond=lambda idx, p: idx < N_l,
        body=mc_body,
        loop_vars=(tf.constant(0), tf.zeros((batch_size, 1), dtype)),
    )
    u_mc_test = u / tf.cast(N_l, tf.float32)

    # Start training and testing
    train_and_test(
        xi,
        xi_approx,
        xi,
        u_mc_test,
        u_reference,
        neurons,
        train_steps,
        mc_rounds,
        mc_freq,
        "single-introductory.csv",
        dtype,
    )
