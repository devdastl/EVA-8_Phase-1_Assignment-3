# EVA-8_Phase-1_Assignment-3
## Part-I
## Part-II
### Introduction
In the part-2 of assignment-3 we need to write a CNN based neural architecture which will need to achieve validating accuracy of `99.4%` on MNIST digit dataset. Following are the constraints to achieve this results
  - Model needs to have less then 20k parameters.
  - Training needs to be done under 20 epoch.
  - Model architecture need to use Batch normalization, Dropout, a Fully connected layer and Global Average Pooling.
  
### Data representation
As mentioned we are using MNIST digit dataset which basically has a 28x28 image of a digit and an integer representing that digit between 0 & 9.
Below is an image showing a single datapoint of the used dataset.
![Alt text](img_data.JPG?raw=true "model architecture")

### Model architecture
As mentioned in the introduction, model arhitecture make use of mentioned layers. Below is the image representing model architecture
![Alt text](img_arch.JPG?raw=true "model architecture")

Below log represent output shape of each layers as well as total number of parameters which are <20k.
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 14, 26, 26]             140
       BatchNorm2d-2           [-1, 14, 26, 26]              28
           Dropout-3           [-1, 14, 26, 26]               0
            Conv2d-4           [-1, 16, 24, 24]           2,032
       BatchNorm2d-5           [-1, 16, 24, 24]              32
           Dropout-6           [-1, 16, 24, 24]               0
         MaxPool2d-7           [-1, 16, 12, 12]               0
            Conv2d-8           [-1, 32, 10, 10]           4,640
       BatchNorm2d-9           [-1, 32, 10, 10]              64
          Dropout-10           [-1, 32, 10, 10]               0
           Conv2d-11             [-1, 16, 8, 8]           4,624
      BatchNorm2d-12             [-1, 16, 8, 8]              32
          Dropout-13             [-1, 16, 8, 8]               0
           Conv2d-14             [-1, 14, 6, 6]           2,030
      BatchNorm2d-15             [-1, 14, 6, 6]              28
          Dropout-16             [-1, 14, 6, 6]               0
           Conv2d-17             [-1, 10, 2, 2]           3,510
      BatchNorm2d-18             [-1, 10, 2, 2]              20
          Dropout-19             [-1, 10, 2, 2]               0
AdaptiveAvgPool2d-20             [-1, 10, 1, 1]               0
           Linear-21                   [-1, 10]             110
================================================================
Total params: 17,290
Trainable params: 17,290
Non-trainable params: 0
----------------------------------------------------------------
```
### Training and Evaluation log
Below are the logs generated while training this model architecture. It almost achieved accuracy ~99.4% but also the accuracy is fluctuating 
```
loss=0.12086963653564453 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.78it/s]

Test set: Average loss: 0.0660, Accuracy: 9842/10000 (98.4200%)

loss=0.047105640172958374 batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 27.31it/s]

Test set: Average loss: 0.0348, Accuracy: 9896/10000 (98.9600%)

loss=0.05986561253666878 batch_id=468: 100%|██████████| 469/469 [00:18<00:00, 25.06it/s]

Test set: Average loss: 0.0317, Accuracy: 9899/10000 (98.9900%)

loss=0.021554915234446526 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.75it/s]

Test set: Average loss: 0.0310, Accuracy: 9913/10000 (99.1300%)

loss=0.008834722451865673 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.17it/s]

Test set: Average loss: 0.0291, Accuracy: 9918/10000 (99.1800%)

loss=0.01571638323366642 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.25it/s]

Test set: Average loss: 0.0231, Accuracy: 9933/10000 (99.3300%)

loss=0.0705643892288208 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.11it/s]

Test set: Average loss: 0.0228, Accuracy: 9928/10000 (99.2800%)

loss=0.02423100732266903 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.52it/s]

Test set: Average loss: 0.0212, Accuracy: 9933/10000 (99.3300%)

loss=0.02119695208966732 batch_id=468: 100%|██████████| 469/469 [00:20<00:00, 22.53it/s]

Test set: Average loss: 0.0195, Accuracy: 9939/10000 (99.3900%)

loss=0.033961083739995956 batch_id=468: 100%|██████████| 469/469 [00:22<00:00, 21.11it/s]

Test set: Average loss: 0.0200, Accuracy: 9937/10000 (99.3700%)

loss=0.039219122380018234 batch_id=468: 100%|██████████| 469/469 [00:23<00:00, 19.70it/s]

Test set: Average loss: 0.0239, Accuracy: 9930/10000 (99.3000%)

loss=0.006649520248174667 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.13it/s]

Test set: Average loss: 0.0188, Accuracy: 9941/10000 (99.4100%)

loss=0.023747803643345833 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.22it/s]

Test set: Average loss: 0.0207, Accuracy: 9937/10000 (99.3700%)

loss=0.019061079248785973 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.32it/s]

Test set: Average loss: 0.0193, Accuracy: 9931/10000 (99.3100%)

loss=0.009254010394215584 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 28.80it/s]

Test set: Average loss: 0.0210, Accuracy: 9932/10000 (99.3200%)

loss=0.05322651565074921 batch_id=468: 100%|██████████| 469/469 [00:17<00:00, 26.29it/s]

Test set: Average loss: 0.0190, Accuracy: 9943/10000 (99.4300%)

loss=0.051516756415367126 batch_id=468: 100%|██████████| 469/469 [00:16<00:00, 27.88it/s]

Test set: Average loss: 0.0181, Accuracy: 9941/10000 (99.4100%)
```
### Conclusion
We almost achieved accuracy of 99.4% under 20k parameters but it could be further improved.
