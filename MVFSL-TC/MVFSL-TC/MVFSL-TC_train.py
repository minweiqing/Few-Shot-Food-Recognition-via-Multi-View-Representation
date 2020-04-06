from one_shot_learning_relation_network import *
from experiment_builder_mini import ExperimentBuilder
import tensorflow.contrib.slim as slim
#import data as dataset
from miniImagenetOneShot_rn import FoodOneShotDataset
import tqdm
from storage import *
import pdb
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'
config = tf.ConfigProto() 

tf.reset_default_graph()
def load_initial_weights(session, WEIGHTS_PATH):
    """
    As the weights from http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/ come
    as a dict of lists (e.g. weights['conv1'] is a list) and not as dict of
    dicts (e.g. weights['conv1'] is a dict with keys 'weights' & 'biases') we
    need a special load function
    """
    not_load_layers=['Adam','Adam_1']
    if WEIGHTS_PATH == 'None':
        raise ValueError('Please supply the path to a pre-trained model')
    print('Loading the weights of {}'.format(WEIGHTS_PATH))
    cp_vars = tf.contrib.framework.list_variables(WEIGHTS_PATH)
    load_layers = {}
    tf.get_variable_scope().reuse_variables()
    for var_name, _ in cp_vars:
        print(var_name)
        if var_name == 'Variable':
            continue
        if 'Adam' in var_name.split('/'):
          continue
        if 'Adam_1' in var_name.split('/'):
          continue
        if var_name == 'beta1_power':
            continue
        if var_name == 'beta2_power':
            continue
        load_layers[var_name] = tf.get_variable(var_name)
    init_fn = tf.contrib.framework.assign_from_checkpoint_fn(WEIGHTS_PATH,
                                                             load_layers,
                                                             ignore_missing_vars=True,
                                                             reshape_variables=False)
    init_fn(session)
    print('load pretrained models!')
def load_initial_weights2(session, WEIGHTS_PATH):
    not_load_layers=['fc6','fc7','fc8','global_step','total_loss','weight_loss',
    'weight_loss_1','weight_loss_2','weight_loss_3','weight_loss_4','weight_loss_5']
    anoth_layers=['conv4_1','conv4_2','conv4_3','conv5_1','conv5_2','conv5_3']
    if WEIGHTS_PATH == 'None':
        raise ValueError('Please supply the path to a pre-trained model')
    print('Loading the weights of {}'.format(WEIGHTS_PATH))
    cp_vars = tf.contrib.framework.list_variables(WEIGHTS_PATH)
    load_layers = {}
    tf.get_variable_scope().reuse_variables()
    for var_name, _ in cp_vars:
        print (var_name)
        if var_name == 'Variable':
            continue
        if var_name == 'cross_entropy/avg':
            continue
        if var_name.split('/')[0] in not_load_layers:
            continue
        if var_name.split('/')[0] in anoth_layers:
        	load_layers[var_name] = tf.get_variable('g_2/'+var_name)
        else:
            load_layers[var_name] = tf.get_variable('g/'+var_name)
    init_fn = tf.contrib.framework.assign_from_checkpoint_fn(WEIGHTS_PATH,
                                                             load_layers,
                                                             ignore_missing_vars=True,
                                                             reshape_variables=False)    
    init_fn(session)
    print('load pretrained models!')   
batch_size = 1#useless
fce = False
classes_per_set = 20# way
samples_per_class = 1#shot
continue_from_epoch = -1 # use -1 to start from scratch
root_path = 'MVFSL-TC'
logs_path = "one_shot_outputs/"#logs path
experiment_name = "one_shot_learning_embedding_{}_{}-new_test_list-2-64".format(samples_per_class, classes_per_set)# save_results
model_path = os.path.join(root_path,'one_shot_learning_embedding_1_20-test--concate-0.5-5e-5_3.ckpt')#path of relation_network pre-trained model
model_path2 = os.path.join(root_path,'fine_tune101.ckpt')#path of category pre-trained model(ingredient pertrained model see:one_shot_learning_relation_network.py)
total_epochs = 20
total_train_batches = 2000# iter number of one epoch
total_test_batches = 1000# iter number of one epoch
dataTrain = FoodOneShotDataset(type = 'train',nEpisodes =total_epochs*total_train_batches,classes_per_set = classes_per_set,samples_per_class = samples_per_class)#make train episodes
# dataVal = miniImagenetOneShotDataset(type = 'val',nEpisodes = total_epochs*total_val_batches,classes_per_set = 5,samples_per_class = 1)
dataTest = FoodOneShotDataset(type = 'test',nEpisodes = total_test_batches,classes_per_set = classes_per_set,samples_per_class = samples_per_class)#make test episodes

experiment = ExperimentBuilder(dataTrain, dataTest, batch_size)
one_shot_omniglot, losses, c_error_opt_op, init, out_summary = experiment.build_experiment(batch_size, classes_per_set, samples_per_class, fce)
#one_shot_omniglot, losses, c_error_opt_op, init = experiment.build_experiment(batch_size, classes_per_set, samples_per_class, fce)
writer = tf.summary.FileWriter(logs_path)
model=VGG16_ingre(num_classes=132)
save_statistics(experiment_name, ["epoch", "train_c_loss", "train_c_accuracy", "val_loss", "val_accuracy", "test_c_loss", "test_c_accuracy"])
with tf.Session(config = config) as sess:
    # 
    sess.run(tf.global_variables_initializer())
    load_initial_weights(sess,model_path)
    load_initial_weights2(sess,model_path2)
    model.load_initial_weights(sess) #ingredient model init
    print(sess.run(tf.contrib.framework.get_variables('g_ingre/conv5_3/weights')))
    saver = tf.train.Saver(max_to_keep=10)
    writer.add_graph(sess.graph)
    if continue_from_epoch != -1: #load checkpoint if needed
        checkpoint = "saved_models/{}_{}.ckpt".format(experiment_name, continue_from_epoch)
        saver.restore(sess,checkpoint)
        print(sess.run(tf.contrib.framework.get_variables('g/conv5_3/weights')))
    with tqdm.tqdm(total=total_epochs) as pbar_e:
        for e in range(0, total_epochs):
            if experiment.total_train_iter% 30000 == 0 and e != 0:
                experiment.current_learning_rate /= 2
                print("change learning rate", experiment.current_learning_rate)

            total_c_loss, total_accuracy = experiment.run_training_epoch(total_train_batches=total_train_batches,
                                                                                sess=sess)
            print("Epoch {}: train_loss: {}, train_accuracy: {}".format(e, total_c_loss, total_accuracy))
            total_test_c_loss, total_test_accuracy = experiment.run_testing_epoch(
                                                                    total_test_batches=total_test_batches, sess=sess)
            print("Epoch {}: test_loss: {}, test_accuracy: {}".format(e, total_test_c_loss, total_test_accuracy))
            save_statistics(experiment_name,
                            [e, total_c_loss, total_accuracy,  total_test_c_loss,total_test_accuracy])

            save_path = saver.save(sess, "saved_models/{}_{}.ckpt".format(experiment_name, e))
            pbar_e.update(1)
