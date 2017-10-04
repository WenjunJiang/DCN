import gzip
import pickle
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
    train_x, train_y = test_set

# dump data to a pickle
with open(data_pkl, 'wb') as f:
    pickle.dump((train_x, train_y), f)

# compress the data
f_in = open(data_pkl, 'rb')
f_out = gzip.open(data_gz, 'wb')
f_out.writelines(f_in)
f_out.close()
f_in.close()

# # visualize
# plt.imshow(train_x[0].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()
