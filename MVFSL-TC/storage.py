import csv
import tensorflow as tf
import pdb
def save_statistics(experiment_name, line_to_add):
    with open("{}.csv".format(experiment_name), 'a') as f:
        writer = csv.writer(f)
        writer.writerow(line_to_add)

def load_statistics(experiment_name):
    data_dict = dict()
    with open("{}.csv".format(experiment_name), 'r') as f:
        lines = f.readlines()
        data_labels = lines[0].replace("\n","").split(",")
        del lines[0]

        for label in data_labels:
            data_dict[label] = []

        for line in lines:
            data = line.replace("\n","").split(",")
            for key, item in zip(data_labels, data):
                data_dict[key].append(item)
    return data_dict
def init_from_ckpt(path):
    not_load_layers = ['fc8','global_step','total_loss','weight_loss',
    'weight_loss_1','cross_entropy','weight_loss_2','weight_loss_3','weight_loss_4','weight_loss_5']

    if path == 'None':
        raise ValueError('Please supply the path to a checkpoint of model')

    print('Loading the weights of {}'.format(path))

    cp_vars = tf.train.list_variables(path)
    load_layers = {}
    for var_name, _ in cp_vars:
        tmp_layer = var_name.split('/')[0]
        # print(tmp_layer)
        
        if tmp_layer not in not_load_layers:
            try:
                if var_name !='Variable':
                    tf.get_variable_scope().reuse_variables()
                    print(var_name)
                    load_layers[var_name] = 'g/'+var_name
            except:
                continue

    print('----------Alreadly loaded variables--------')
    # for k,j in load_layers.items():
    #     print(k)
    #     print(j)
    # pdb.set_trace()
    tf.train.init_from_checkpoint(path, load_layers)
    print('Loading the weights is Done.')



