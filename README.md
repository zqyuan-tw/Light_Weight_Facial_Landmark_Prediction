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

## [sicxu/Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch)
### [Paper](https://arxiv.org/pdf/1903.08527.pdf)
1. Hybrid-level weakly-supervised training for CNN-based 3D face reconstruction
2. Fast, accurate, and robust to pose and occlusions
3. Using [Arcface](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) to compute perceptual loss. Need to download the pre-trained model in advance
4. 不需要Ground Truth，只需要把image丟到dlib或是MTCNN就可以detect出初始的5個landmark
5. R-Net pre-trained on ImageNet
## [Jiahao-UTS/SLPT-master](https://github.com/Jiahao-UTS/SLPT-master)
1. Heatmap regression method ignore the inherent relation between different landmarks
2. Using attention mechanism to learn this inherent relation
3. Model size: SLPT-6-layers `51 Mb` and SLPT-12-layers `75 Mb`
4. **Coarse-to-fine locating** 每一個stage會以前一個stage的facial landmarks為中心crop出patches，再將這些patches分別resize成K x K，以這個為input丟入SPLT並計算出local landmark的位置
## ADNet
1. Network employs stacked 4 Hourglasses as backbone, and in each HG structure, three heatmaps are generated, respectively corresponding to a landmarks heatmap, edge heatmap and a point heatmap
2. An anisotropic attention mask is generated from the point and edge heatmaps. The attention mask can then impose anisotropic supervsion for landmark training
3. **Anisotropic Direction Loss** disentangles the error into two mutually orthogonal directions, namely normal error into and tangent error, and put anisotropic loss weights to them
## [choyingw/SynergyNet](https://github.com/choyingw/SynergyNet)
1. Backbone learns to regress 3DMM parameters and reconstruct 3D face meshes from monocular face images
2. Muti-Attribute feature aggregation gathers underlying 3DMM semantics and the latent image code to refine landmarks further
3. Landmark-to-3DMM regresses 3DMM from refined landmark  to reveale the embedded facial geometry in 3D landmark
4. **Self-Constraint** is applied to 3DMM parameters regressed from different sources
