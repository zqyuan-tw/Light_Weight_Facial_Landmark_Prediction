# Light-Weight-Facial-Landmark-Prediction-Challenge
## Datasets
### Training Data
`99756` images with ground truth
### Validation Data
`199` images with ground truth
## [FaceAlignment Library](https://github.com/1adrianb/face-alignment "游標顯示")
### [Training Code in .lua](https://github.com/1adrianb/face-alignment-training )
1. 需要使用`Torch7`這個module進行訓練，不知道有沒有相關的規定？
2. model size很大，最小的也有`183 Mb`
## [PFLD-pytorch](https://github.com/polarisZhao/PFLD-pytorch)
### [polarisZhao/PFLD-pytorch](https://github.com/polarisZhao/PFLD-pytorch)
### [samuelyu2002/PFLD](https://github.com/samuelyu2002/PFLD)
1. Employ a branch of network to estimate the geometric information for each face sample (Auxiliary  Network)
2. Build the backbone network of our PFLD using MobileNet blocks to gurantee small model size
3. 在paper中提出PFLD Loss，目的在於解決data imbalance的問題 \
Penalize more on errors corresponding to rare training samples than on those to rich ones
5. Ground truth 是每一張照片都有`98 landmarks`
6. 照他的demo來看應該是可以處理另外一邊的臉看不到的問題
[PFLD: A Practical Facial Landmark Detector](https://arxiv.org/pdf/1902.10859.pdf)
3. Paper中提出兩種架構，`PFLD 0.25X`及`PFLD 1X`，model size分別是`2.1 Mb`和`12.5 Mb`
4. `PFLD 1X`的NME可以達到1.88
5. source code中仍有face detector
