import cv2
import time
from dataset import Dataset


class Model():
    def __init__(self, model_name='LBPH'):
        self.model_name = model_name

        if model_name == 'PCA':
            # 特征脸方法
            self.model = cv2.face.EigenFaceRecognizer_create()
        elif model_name == 'LDA':
            # 线性判别分析
            self.model = cv2.face.FisherFaceRecognizer_create()
        elif model_name == 'LBPH':
            # LBP 局部二值模式直方图
            self.model = cv2.face.LBPHFaceRecognizer_create()
        else:
            print("ERROR,non support model type...")
            self.model = None
    
    def train(self, train_data, train_label):
        print(self.model_name + ' is training...')
        startime = time.time()
        self.model.train(train_data, train_label)
        endtime = time.time()
        costime = round(endtime - startime, 0)
        print(self.model_name + ' 训练耗时: ' + str(costime) + 's')

    def predict(self, test_data):
        pred_label = []
        pred_confidence = []

        print(self.model_name + ' is predicting...')
        startime = time.time()
        for i in range(len(test_data)):
            label, confidence = self.model.predict(test_data[i])
            
            pred_label.append(label)
            pred_confidence.append(confidence)

        endtime = time.time()
        costime = round(endtime - startime, 0)
        print(self.model_name + ' 预测耗时: ' + str(costime) + 's')

        return pred_label, pred_confidence


    def evaluate(self, pred_label, label):
        n = len(label)
        correct_num = 0

        for i in range(n):
            if label[i] == pred_label[i]:
                correct_num += 1
                
        print(self.model_name + " acc: " +  str(1.0 * correct_num / n))


if __name__ == "__main__":
    ds_name = 'Pose29'

    dataset = Dataset(ds_name)
    dataset.load()
    dataset.describe()
    trainset, testset = dataset.gen_dataset()
    """
    # PCA
    pca_model = Model('PCA')
    pca_model.train(trainset['data'], trainset['label'])
    pred_label, pred_confidence = pca_model.predict(testset['data'])
    pca_model.evaluate(pred_label, testset['label'])
    """
    # LDA
    lda_model = Model('LDA')
    lda_model.train(trainset['data'], trainset['label'])
    pred_label, pred_confidence = lda_model.predict(testset['data'])
    lda_model.evaluate(pred_label, testset['label'])
    """
    # LBPH
    lbph_model = Model('LBPH')
    lbph_model.train(trainset['data'], trainset['label'])
    pred_label, pred_confidence = lbph_model.predict(testset['data'])
    lbph_model.evaluate(pred_label, testset['label'])
    """
