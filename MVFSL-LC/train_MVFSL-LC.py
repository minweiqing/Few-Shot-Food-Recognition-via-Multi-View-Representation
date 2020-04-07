from one_shot_learning_relation_network import *
from Food_experiment_builder import ExperimentBuilder
import tensorflow.contrib.slim as slim
from FoodOneShot_rn import FoodOneShotDataset
import tqdm
from storage import *
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
tf.reset_default_graph()
# Experiment Setup
batch_size = 1
fce = False
classes_per_set = 20
samples_per_class = 1
continue_from_epoch = -1# use -1 to start from scratch
epochs = 200
logs_path = "one_shot_outputs/"
experiment_name = "Food-one_shot_learning_embedding_{}_{}-test--concate-0.5-5e-5".format(samples_per_class, classes_per_set)
total_epochs = 10
total_train_batches = 10000
total_test_batches = 1000
root='/media/goodsrc/d3f87cd5-e8d1-4aa1-b51f-a46bb097e6bb/goodsrc1/food'
dataTrain = FoodOneShotDataset(type = 'train',dataroot=root,nEpisodes = total_epochs*batch_size*total_train_batches,classes_per_set = classes_per_set,samples_per_class = samples_per_class)
dataTest = FoodOneShotDataset(type = 'test',dataroot=root,nEpisodes = batch_size*total_test_batches,classes_per_set = classes_per_set,samples_per_class = samples_per_class)
experiment = ExperimentBuilder(dataTrain,  dataTest, batch_size)
one_shot_omniglot, losses, c_error_opt_op, init, out_summary = experiment.build_experiment(batch_size, classes_per_set, samples_per_class, fce)
writer = tf.summary.FileWriter(logs_path)

save_statistics(experiment_name, ["epoch", "train_c_loss", "train_c_accuracy", "val_loss", "val_accuracy","test_c_loss", "test_c_accuracy"])

# Experiment initialization and running
with tf.Session() as sess:
    sess.run(init)
    saver = tf.train.Saver(max_to_keep=5)
    writer.add_graph(sess.graph)
    if continue_from_epoch != -1: #load checkpoint if needed
        checkpoint = "saved_models/{}_{}.ckpt".format(experiment_name, continue_from_epoch)
        variables_to_restore = []
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
            print(var)
            variables_to_restore.append(var)

        tf.logging.info('Fine-tuning from %s' % checkpoint)

        fine_tune = slim.assign_from_checkpoint_fn(
            checkpoint,
            variables_to_restore,
            ignore_missing_vars=True)
        fine_tune(sess)

    best_val = 0.

    with tqdm.tqdm(total=total_epochs) as pbar_e:
        for e in range(0, total_epochs):
            #change learning rate
            if experiment.total_train_iter% 50000 == 0 and e != 0:
                experiment.current_learning_rate /= 2
                print("change learning rate", experiment.current_learning_rate)
            total_c_loss, total_accuracy = experiment.run_training_epoch(total_train_batches=total_train_batches,
                                                                                sess=sess)
            print("Epoch {}: train_loss: {}, train_accuracy: {}".format(e, total_c_loss, total_accuracy))

            # total_val_c_loss, total_val_accuracy = experiment.run_validation_epoch(
            #                                                                     total_val_batches=total_val_batches,
            #                                                                     sess=sess)
            # print("Epoch {}: val_loss: {}, val_accuracy: {}".format(e, total_val_c_loss, total_val_accuracy))

            # #if total_val_accuracy >= best_val: #if new best val accuracy -> produce test statistics
            # if True:
            #     best_val = total_val_accuracy
            total_test_c_loss, total_test_accuracy = experiment.run_testing_epoch(total_test_batches=total_test_batches, sess=sess)
            print("Epoch {}: test_loss: {}, test_accuracy: {}".format(e, total_test_c_loss, total_test_accuracy))
            # else:
            #     total_test_c_loss = -1
            #     total_test_accuracy = -1
            save_statistics(experiment_name,[e, total_c_loss, total_accuracy,  total_test_c_loss,total_test_accuracy])
            save_path = saver.save(sess, "saved_models/{}_{}.ckpt".format(experiment_name, e))
            pbar_e.update(1)
