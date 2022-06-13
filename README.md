# Light-Weight-Facial-Landmark-Prediction-Challenge
## Datasets
#### Training Data
`99756` images with ground truth
#### Validation Data
`199` images with ground truth
### Data Pre-processing
#### Gamma Correction
Gamma Correction is also known as Power Law Transform. First, our image pixel intensities must be scaled from the range [0, 255] to [0, 1.0]. From there, we obtain our output gamma corrected image by applying the following equation:

$O = I^{1/G}$\
where $I$ is the input image, $G$ is the gamma value and $O$ is the gamma corrected image 

To map the input image pixel to gamma corrected output values, we have to build a lookup table and thus the function can quickly determine the output value for the input pixel (in $O(1)$ time)
![](https://i.imgur.com/r1Ll0M0.png)

![](https://i.imgur.com/aq0QHgQ.png)
![](https://i.imgur.com/5FLVrN6.png)
![](https://i.imgur.com/xLzsQd5.png)
![](https://i.imgur.com/pUmR4YO.png)

<center class="half">
     <img src="https://i.imgur.com/r1Ll0M0.png" width="200"/><img src="![](https://i.imgur.com/5FLVrN6.png)" width="200"/>
</center>

<details>
    <summary>FaceAlignment Library</summary>
    
## [FaceAlignment Library](https://github.com/1adrianb/face-alignment "游標顯示")
### [Training Code in .lua](https://github.com/1adrianb/face-alignment-training )
1. 需要使用`Torch7`這個module進行訓練，不知道有沒有相關的規定？
2. model size很大，最小的也有`183 Mb`
</details>

<details>
    <summary>PFLD-pytorch</summary>
    
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
</details>

<details>
    <summary>Deep3DFaceRecon_pytorch</summary>
    
## [sicxu/Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch)
### [Paper](https://arxiv.org/pdf/1903.08527.pdf)
1. Hybrid-level weakly-supervised training for CNN-based 3D face reconstruction
2. Fast, accurate, and robust to pose and occlusions
3. Using [Arcface](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch) to compute perceptual loss. Need to download the pre-trained model in advance
4. 不需要Ground Truth，只需要把image丟到dlib或是MTCNN就可以detect出初始的5個landmark
5. R-Net pre-trained on ImageNet
</details>

<details>
    <summary>SLPT</summary>
    
## [Jiahao-UTS/SLPT-master](https://github.com/Jiahao-UTS/SLPT-master)
1. Heatmap regression method ignore the inherent relation between different landmarks
2. Using attention mechanism to learn this inherent relation
3. Model size: SLPT-6-layers `51 Mb` and SLPT-12-layers `75 Mb`
4. **Coarse-to-fine locating** 每一個stage會以前一個stage的facial landmarks為中心crop出patches，再將這些patches分別resize成K x K，以這個為input丟入SPLT並計算出local landmark的位置
</details>

<details>
    <summary>ADNet</summary>
    
## ADNet
1. Network employs stacked 4 Hourglasses as backbone, and in each HG structure, three heatmaps are generated, respectively corresponding to a landmarks heatmap, edge heatmap and a point heatmap
2. An anisotropic attention mask is generated from the point and edge heatmaps. The attention mask can then impose anisotropic supervsion for landmark training
3. **Anisotropic Direction Loss** disentangles the error into two mutually orthogonal directions, namely normal error into and tangent error, and put anisotropic loss weights to them
</details>

<details>
    <summary>SynergyNet</summary>
    
## [choyingw/SynergyNet](https://github.com/choyingw/SynergyNet)
1. Backbone learns to regress 3DMM parameters and reconstruct 3D face meshes from monocular face images
2. Muti-Attribute feature aggregation gathers underlying 3DMM semantics and the latent image code to refine landmarks further
3. Landmark-to-3DMM regresses 3DMM from refined landmark  to reveale the embedded facial geometry in 3D landmark
4. **Self-Constraint** is applied to 3DMM parameters regressed from different sources
</details>
<details>
    <summary>ASMNet</summary>
### [ASMNet](https://github.com/aliprf/ASMNet)
![Framework of ASMNet](https://i.imgur.com/KIBgYZm.png)
1. CNN lower layers have features such as edges, and corners which are more suitable for tasks like *landmark localization* and pose estimation
2. Training a network for correlated tasks simultaneously builds a synergy that can improve the preformance of each task
3. The framework is a multi-task CNN that detect facial landmark as well as estimating the pose of faces (pitch, row, and yaw) simultaneously
4. Create short cuts from multiple batch normalization layer
5. Connect each of these shortcuts to the output of block 15 of MobileNetV2, block-15-add, using global average pooling layers
6. Apply PCA on the training set and calculate the eigen-vectors and eigenvalues, and using ASM operator to transform original ground truth to set A, called ASM points
7. ASM-Loss contains two part, mean square error computed from landmark prediction and error between predicted landmarks and ASM points
</details>


