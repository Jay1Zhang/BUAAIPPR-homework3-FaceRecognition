import cv2
import numpy as np
from scipy.io import loadmat
import tensorflow as tf


class Dataset():
    path = '../dataset/PIE dataset/'

    def __init__(self, dataset_name=None):
        self.ds_name = dataset_name if dataset_name is not None else 'PoseAll'    # e.g. Pose05
        self.ds_path = '../dataset/PIE dataset/' + dataset_name + '_64x64.mat' if dataset_name is not None else None


    def load(self):
        if self.ds_path is None:
            print("ERROR, You didn't passed parameter 'dataset_name' when you built the Dataset(). Maybe you want to replace load() with load_all() ?")
            self.load_all()
            return 

        mat = loadmat(self.ds_path)
        self.size = mat['gnd'].shape[0]

        train_data = []
        train_label = []
        test_data = []
        test_label = []

        for i in range(self.size):
            gray = mat['fea'][i].reshape((64, 64))
            data = cv2.merge([gray])
            label = mat['gnd'][i][0]
            
            if mat['isTest'][i] == 0.0:
                train_data.append(data)
                train_label.append(label)
            else:
                test_data.append(data)
                test_label.append(label)
        
        self.train_data = np.array(train_data)
        self.train_label = np.array(train_label)
        self.test_data = np.array(test_data)
        self.test_label = np.array(test_label)


    def load_all(self):
        self.size = 0
        train_data = []
        train_label = []
        test_data = []
        test_label = []

        # Pose05
        mat = loadmat('../dataset/PIE dataset/Pose05_64x64.mat')
        size = mat['gnd'].shape[0]
        self.size += size
        for i in range(size):
            gray = mat['fea'][i].reshape((64, 64))
            data = cv2.merge([gray])
            label = mat['gnd'][i][0]
            
            if mat['isTest'][i] == 0.0:
                train_data.append(data)
                train_label.append(label)
            else:
                test_data.append(data)
                test_label.append(label)
        
        # Pose07
        mat = loadmat('../dataset/PIE dataset/Pose07_64x64.mat')
        size = mat['gnd'].shape[0]
        self.size += size
        for i in range(size):
            gray = mat['fea'][i].reshape((64, 64))
            data = cv2.merge([gray])
            label = mat['gnd'][i][0]
            
            if mat['isTest'][i] == 0.0:
                train_data.append(data)
                train_label.append(label)
            else:
                test_data.append(data)
                test_label.append(label)

        # Pose09
        mat = loadmat('../dataset/PIE dataset/Pose09_64x64.mat')
        size = mat['gnd'].shape[0]
        self.size += size
        for i in range(size):
            gray = mat['fea'][i].reshape((64, 64))
            data = cv2.merge([gray])
            label = mat['gnd'][i][0]
            
            if mat['isTest'][i] == 0.0:
                train_data.append(data)
                train_label.append(label)
            else:
                test_data.append(data)
                test_label.append(label)

        # Pose27
        mat = loadmat('../dataset/PIE dataset/Pose27_64x64.mat')
        size = mat['gnd'].shape[0]
        self.size += size
        for i in range(size):
            gray = mat['fea'][i].reshape((64, 64))
            data = cv2.merge([gray])
            label = mat['gnd'][i][0]
            
            if mat['isTest'][i] == 0.0:
                train_data.append(data)
                train_label.append(label)
            else:
                test_data.append(data)
                test_label.append(label)
        
        # Pose29
        mat = loadmat('../dataset/PIE dataset/Pose29_64x64.mat')
        size = mat['gnd'].shape[0]
        self.size += size
        for i in range(size):
            gray = mat['fea'][i].reshape((64, 64))
            data = cv2.merge([gray])
            label = mat['gnd'][i][0]
            
            if mat['isTest'][i] == 0.0:
                train_data.append(data)
                train_label.append(label)
            else:
                test_data.append(data)
                test_label.append(label)

        self.train_data = np.array(train_data)
        self.train_label = np.array(train_label)
        self.test_data = np.array(test_data)
        self.test_label = np.array(test_label)

    
    def gen_dataset(self):
        # ensure dataset type as int32 
        trainset = {
            'data': self.train_data.astype(int),
            'label': self.train_label.astype(int)
        }
        testset = {
            'data': self.test_data.astype(int),
            'label': self.test_label.astype(int)
        }
        return trainset, testset

    
    def gen_cnn_dataset(self):
        # trainset
        train_data = self.train_data.astype(int)
        train_data = train_data.reshape(train_data.shape[0], 64, 64, 1).astype(np.float32) / 255

        labels = self.train_label.astype(int) - 1
        batch_size = tf.size(labels)
        labels = tf.expand_dims(labels, 1)
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        concated = tf.concat([indices, labels], 1)
        onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, 68]), 1.0, 0.0)

        train_label = []
        with tf.Session() as sess:
            val = sess.run(onehot_labels)
            train_label.extend(val)
        train_label = np.array(train_label)

        trainset = {
            'data': train_data,
            'label': train_label
        }

        # testset
        test_data = self.test_data.astype(int)
        test_data = test_data.reshape(test_data.shape[0], 64, 64, 1).astype(np.float32) / 255

        labels = self.test_label.astype(int) - 1
        batch_size = tf.size(labels)
        labels = tf.expand_dims(labels, 1)
        indices = tf.expand_dims(tf.range(0, batch_size, 1), 1)
        concated = tf.concat([indices, labels], 1)
        onehot_labels = tf.sparse_to_dense(concated, tf.stack([batch_size, 68]), 1.0, 0.0)

        test_label = []
        with tf.Session() as sess:
            val = sess.run(onehot_labels)
            test_label.extend(val)
        test_label = np.array(test_label)
        
        testset = {
            'data': test_data,
            'label': test_label
        }

        return trainset, testset
    

    def describe(self):
        # describe dataset and generate sample image to ../image/sample/
        print('Dataset Name: ' + self.ds_name)
        print('Dataset Size: ' + str(self.size))

        """
        if self.ds_name != 'PoseAll':
            img = self.train_data[0]
            cv2.imshow(self.ds_name, img)
            cv2.waitKey(0)
            cv2.destroyWindow(self.ds_name)
            cv2.imwrite('../image/sample/' + self.ds_name + '_sample.png', img)
        """
        

if __name__ == "__main__":
    #dataset = Dataset('Pose29')
    #dataset.load()
    dataset = Dataset()
    dataset.load_all()
    dataset.gen_cnn_dataset()

    dataset.describe()
