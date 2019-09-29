from base.base_model import BaseModel
import tensorflow as tf


class CnnMnistModel(BaseModel):
    def __init__(self, config):
        super(CnnMnistModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None, self.config.pixel_size, self.config.pixel_size, 1])
        self.y = tf.placeholder(tf.float32, shape=[None, 10])

        # network architecture
        # d1 = tf.layers.dense(self.x, 512, activation=tf.nn.relu, name="dense1")

        with tf.name_scope("hidden"):
            out = tf.layers.conv2d(self.x, filters=5, kernel_size=3, padding='same', name="conv1", activation=tf.nn.relu)
            # out = tf.nn.relu(out)
            out = tf.layers.max_pooling2d(out, 2, 2)

            assert out.shape[1:] == [14, 14, 5]
            out = tf.reshape(out, [-1, 14 * 14 * 5])

            d1 = tf.layers.dense(out, 128, activation=tf.nn.relu, name="dense1")
            d2 = tf.layers.dense(d1, 10, name="dense2")

        with tf.name_scope("loss"):
            self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=d2))
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.cross_entropy,
                                                                                         global_step=self.global_step_tensor)
            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_ckpt_to_keep)

