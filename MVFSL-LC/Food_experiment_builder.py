import tensorflow as tf
import tqdm
from one_shot_learning_relation_network import RelationNetwork
import pdb


class ExperimentBuilder:

    def __init__(self, dataTrain, dataTest, batch_size):
        self.dataTrain = dataTrain
        self.dataTest = dataTest
        self.batch_size = batch_size

    def build_experiment(self, batch_size, classes_per_set, samples_per_class, fce):
        self.support_set_images = tf.placeholder(tf.float32, [classes_per_set, samples_per_class, 14,14,1024], 'support_set_images')
        self.support_set_labels = tf.placeholder(tf.int32, [classes_per_set, samples_per_class], 'support_set_labels')
        self.target_image = tf.placeholder(tf.float32, [classes_per_set*1,14,14,1024], 'target_image')
        self.target_label = tf.placeholder(tf.int32, [classes_per_set*1], 'target_label')
        self.training_phase = tf.placeholder(tf.bool, name='training-flag')
        self.rotate_flag = tf.placeholder(tf.bool, name='rotate-flag')
        self.keep_prob = tf.placeholder(tf.float32, name='dropout-prob')
        self.current_learning_rate = 5e-05
        self.learning_rate = tf.placeholder(tf.float32, name='learning-rate-set')
        self.one_shot_omniglot = RelationNetwork(batch_size=batch_size, support_set_images=self.support_set_images,
                                                 support_set_labels=self.support_set_labels,
                                                 target_image=self.target_image, target_label=self.target_label,
                                                 keep_prob=self.keep_prob, 
                                                 is_training=self.training_phase, fce=fce, rotate_flag=self.rotate_flag,
                                                 num_classes_per_set=classes_per_set,
                                                 num_samples_per_class=samples_per_class,
                                                 learning_rate=self.learning_rate)

        summary, self.losses, self.c_error_opt_op = self.one_shot_omniglot.init_train()
        init = tf.global_variables_initializer()
        self.total_train_iter = 0
        return self.one_shot_omniglot, self.losses, self.c_error_opt_op, init, summary

    def run_training_epoch(self, total_train_batches, sess):
        total_c_loss = 0.
        total_accuracy = 0.
        # pdb.set_trace()
        with tqdm.tqdm(total=total_train_batches) as pbar:

            for i in range(total_train_batches):  # train epoch
                x_support_set, y_support_set, x_target, y_target = self.dataTrain.get_batch(self.batch_size,
                                                                                            shuffle=True)
                _, c_loss_value, acc = sess.run(
                    [self.c_error_opt_op, self.losses[self.one_shot_omniglot.classify],
                     self.losses[self.one_shot_omniglot.dn]],
                    feed_dict={self.keep_prob: 0.5, self.support_set_images: x_support_set,
                               self.support_set_labels: y_support_set, self.target_image: x_target,
                               self.target_label: y_target,
                               self.training_phase: True, self.rotate_flag: False,
                               self.learning_rate: self.current_learning_rate})

                iter_out = "train_loss: {}, train_accuracy: {}".format(c_loss_value, acc)
                pbar.set_description(iter_out)

                pbar.update(1)
                total_c_loss += c_loss_value
                total_accuracy += acc
                self.total_train_iter += 1
        total_c_loss = total_c_loss / total_train_batches
        total_accuracy = total_accuracy / total_train_batches
        return total_c_loss, total_accuracy

    def run_testing_epoch(self, total_test_batches, sess):
        total_test_c_loss = 0.
        total_test_accuracy = 0.
        with tqdm.tqdm(total=total_test_batches) as pbar:
            for i in range(total_test_batches):
                x_support_set, y_support_set, x_target, y_target = self.dataTest.get_batch(self.batch_size,
                                                                                           shuffle=False)
                c_loss_value, acc = sess.run(
                    [self.losses[self.one_shot_omniglot.classify], self.losses[self.one_shot_omniglot.dn]],
                    feed_dict={self.keep_prob: 1.0, self.support_set_images: x_support_set,
                               self.support_set_labels: y_support_set, self.target_image: x_target,
                               self.target_label: y_target,
                               self.training_phase: False, self.rotate_flag: False})

                iter_out = "test_loss: {}, test_accuracy: {}".format(c_loss_value, acc)
                pbar.set_description(iter_out)
                pbar.update(1)

                total_test_c_loss += c_loss_value
                total_test_accuracy += acc
            total_test_c_loss = total_test_c_loss / total_test_batches
            total_test_accuracy = total_test_accuracy / total_test_batches
        return total_test_c_loss, total_test_accuracy
