# Age-Regression

## Overview
The basic objective of the code is to be able to capture the analogy between an image and a filtered version of that image and then take that analogy and apply it on someother input image. This is then extended to be able to age a person if the analogy between the two input images is an age regression effect.

## Part 1 - Simple Analogy Transfer
Consider an 3 input images A, A', and B. We take each pixel of B and find the operation that must be taken on it so that pixel-by-pixel we are able to construct the output B'. We attempt to find a best match for each pixel in B with a pixel in A. This is done using two techniques. The **Best Approximate Match** will go through each pixel in A and compare the feature vectors of each pixel and the pixel selected in B using L2 norm. The feature vector is a list of attributes associated with each pixel such as luminosity and texture which make it a basis of comparison between pixels. Another technique is to find the **Best Coherent Match** which picks out a pixel from A which best matches with the neighborhood around the selected pixel in B by comparing the feature vectors of the pixels in the neighborhood. A parameter kappa is set to decide whether to use the Best approximate match or the Best Coherent Match. After a pixel is match the filter operation of the pixel in A to A' is matched to the pixel in B to obtain B'. 

## Examples of Simple Analogy Transfer
![A](https://github.com/NikhilGupta1997/Age-Regression/tree/master/Images/blurA1.jpg)
