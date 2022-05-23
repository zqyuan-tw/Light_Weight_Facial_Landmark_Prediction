# Light-Weight-Facial-Landmark-Prediction-Challenge
## [FaceAlignment Library](https://github.com/1adrianb/face-alignment "游標顯示")
### [Training Code in .lua](https://github.com/1adrianb/face-alignment-training )
1. 需要使用`Torch7`這個module進行訓練，不知道有沒有相關的規定？
2. model size很大，最小的也有`183 Mb`
## [PFLD-pytorch](https://github.com/polarisZhao/PFLD-pytorch)
### [Training Code in PyTorch](https://github.com/polarisZhao/PFLD-pytorch/blob/master/train.py)
1. Ground truth 是每一張照片都有`98 landmarks`
2. 照他的demo來看應該是可以處理另外一邊的臉看不到的問題
[PFLD: A Practical Facial Landmark Detector](https://arxiv.org/pdf/1902.10859.pdf)
3. Paper中提出兩種架構，`PFLD 0.25X`及`PFLD 1X`，model size分別是`2.1 Mb`和`12.5 Mb`
4. `PFLD 1X`的NME可以達到1.88
5. source code中仍有face detector
