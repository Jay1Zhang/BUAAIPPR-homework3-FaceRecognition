import cv2
import numpy as np
from scipy.io import loadmat


def read_data(filepath):
    mat = loadmat(filepath)
    
    train_data = []
    train_label = []
    test_data = []
    test_label = []
    n = mat['gnd'].shape[0]
    
    for i in range(n):
        gray = mat['fea'][i].reshape((64, 64))
        data = cv2.merge([gray])
        label = mat['gnd'][i][0]
        
        if mat['isTest'][i] == 0.0:
            train_data.append(data)
            train_label.append(label)
        else:
            test_data.append(data)
            test_label.append(label)
    
    return np.array(train_data).astype(int), np.array(train_label).astype(int), \
            np.array(test_data).astype(int), np.array(test_label).astype(int)


"""
    model_type = 'PCA' or 'LDA' or 'LBPH'
"""
def build_model(train_data, train_label, model_type='LBPH'):
    if model_type == 'PCA':
        # 特征脸方法
        model = cv2.face.EigenFaceRecognizer_create()
    elif model_type == 'LDA':
        # 线性判别分析
        model = cv2.face.FisherFaceRecognizer_create()
    elif model_type == 'LBPH':
        # LBP 局部二值模式直方图
        model = cv2.face.LBPHFaceRecognizer_create()
    else:
        print("ERROR,non support model type...")
        return None
    
    print('training...')
    model.train(train_data, train_label)

    return model


def predict(model, test_data):
    pred_label = []
    pred_confidence = []

    print('predicting...')
    for i in range(len(test_label)):
        label, confidence = model.predict(test_data[i])
        
        pred_label.append(label)
        pred_confidence.append(confidence)

    return pred_label, pred_confidence


def evaluate(pred_label, label):
    n = len(test_label)
    correct_num = 0

    for i in range(n):
        #print(pred_confidence[i])
        if test_label[i] == pred_label[i]:
            correct_num += 1
            
    print(1.0 * correct_num / n)


if __name__ == "__main__":
    filepath = "../dataset/PIE dataset/Pose29_64x64.mat"
    train_data, train_label, test_data, test_label = read_data(filepath)

    model = build_model(train_data, train_label, model_type='PCA')
    pred_label, pred_confidence = predict(model, test_data)
    evaluate(pred_label, test_label)
