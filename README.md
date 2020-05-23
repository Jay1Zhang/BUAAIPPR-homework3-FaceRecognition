# 图像处理第三次大作业 - 人脸识别

17373489 张佳一



特征脸算法对光照十分敏感。



（1）主成分分析（PCA）——Eigenfaces（特征脸）——函数：cv2.face.EigenFaceRecognizer_create（）

PCA：低维子空间是使用主元分析找到的，找具有最大方差的哪个轴。

缺点：若变化基于外部（光照），最大方差轴不一定包括鉴别信息，不能实行分类。



（2）线性判别分析（LDA）——Fisherfaces（特征脸）——函数： cv2.face.FisherFaceRecognizer_create()

LDA:线性鉴别的特定类投影方法，目标：实现类内方差最小，类间方差最大。

（3）局部二值模式（LBP）——LocalBinary Patterns Histograms——函数：cv2.face.LBPHFaceRecognizer_create()

PCA和LDA采用整体方法进行人脸辨别，LBP采用局部特征提取，除此之外，还有的局部特征提取方法为：

盖伯小波（Gabor Waelets）和离散傅里叶变换（DCT）。




## 参考文献

[1] http://openbio.sourceforge.net/resources/eigenfaces/eigenfaces-html/facesOptions.html



