
# Grad-CAM(Gradient-wegihted Class Activation Mapping)
 - Model Visualization explanations 기법
 - CAM을 통해 모델이 무엇을 보고 예측/분류하는지에 대해 시각적으로 보여줌으로써 학습이 잘 되었는지, 한쪽으로 치우쳐 학습되지는 않았는지, General한 특성을 학습했는지에 대해 확인할 수 있다.

 - 구현 원리
  : 일반적으로 CNN모델 분류 문제의 경우에는 Softmax 전에 Spatial 구조를 가지는 형태에서 분류하는 개수로 FCN을 구성하게 되는데, 이 Spatial 구조를 가지는 Feature map에서의 Gradient 값들을 Global average pooling을 이용하여 모델이 참고하는 Feature들의 Importance 값을 얻는다.

- **Global average pooling**
![gaap](/assets/gaap.PNG)
![grad-cam Relu](/assets/grad-cam%20Relu.PNG)
A: Featrue map activation(before softmax)
 i,j :pixels
a:weights of classes
![grad-cam](/assets/grad-cam.PNG)![grad-](/assets/grad-.PNG)
