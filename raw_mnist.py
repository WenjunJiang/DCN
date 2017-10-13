import gzip
import pickle
import numpy as np
import aelib.data
import matplotlib.cm as cm
import matplotlib.pyplot as plt

filename = 'mnist.pkl.gz'
path = 'data/MNIST/'
dataset = path+filename

file_pkl = 'mnist_dcn.pkl'
file_gz = 'mnist_dcn.pkl.gz'
data_pkl = path+file_pkl
data_gz = path+file_gz

with gzip.open(dataset, 'rb') as f:
    train_set, valid_set, test_set = pickle.load(f)
    train_x, train_y = train_set
    train_x = np.append(train_x, valid_set[0], axis=0)
    train_y = np.append(train_y, valid_set[1])
    train_x = np.append(train_x, test_set[0], axis=0)
    train_y = np.append(train_y, test_set[1])

# dump data to a pickle
with open(data_pkl, 'wb') as f:
    pickle.dump((train_x, train_y), f)

# dump another MNIST dataset
# X, Y = aelib.data.get_mnist()
# with open(data_pkl, 'wb') as f:
#     pickle.dump((X, Y), f)

# compress the data
f_in = open(data_pkl, 'rb')
f_out = gzip.open(data_gz, 'wb')
f_out.writelines(f_in)
f_out.close()
f_in.close()

# create a toy example with 20 data, number 0-3
toy_idx = np.argwhere(train_y==0)[0:5]
toy_idx = np.append(toy_idx, np.argwhere(train_y==1)[0:5])
toy_idx = np.append(toy_idx, np.argwhere(train_y==2)[0:5])
toy_idx = np.append(toy_idx, np.argwhere(train_y==3)[0:5])

toy_y = np.squeeze(train_y[toy_idx])
toy_x = train_x[toy_idx]

toy_pkl = path+'toy.pkl'
toy_gz = path+'toy.pkl.gz'
# dump data to a pickle
with open(toy_pkl, 'wb') as f:
    pickle.dump((toy_x, toy_y), f)

# compress the data
f_in = open(toy_pkl, 'rb')
f_out = gzip.open(toy_gz, 'wb')
f_out.writelines(f_in)
f_out.close()
f_in.close()