import tensorflow as tf


# Parameters
learning_rate = 0.01


# tf Graph Input
x = tf.placeholder(tf.float32, [None, 784]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 10]) # 0-9 digits recognition => 10 classes


# Set model weights
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W)) # Softmax


# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))


W_grad =  - tf.matmul ( tf.transpose(x) , y - pred)
b_grad = - tf.reduce_mean( tf.matmul(tf.transpose(x), y - pred), reduction_indices=0)

new_W = W.assign(W - learning_rate * W_grad)
new_b = b.assign(b - learning_rate * b_grad)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            _, _, c = sess.run([new_W, new_b, cost], feed_dict={x: batch_xs, y: batch_ys})

            # Compute average loss
            avg_cost += c / total_batch
        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            print ("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    print ("Optimization Finished!")
