# EE-399-HW-1

## Abstract
This report analyzes a grayscale image dataset of faces captured under varying lighting conditions. The report explores the computation of correlation matrices of varying sizes and examines the most and least correlated image pairs. Additionally, matrix factorization techniques are employed to compute the first six eigenvectors and principal component directions. Finally, the percentage of variance captured by the top six singular value decomposition modes is computed and the modes are plotted.

## Introduction and Overview
Machine learning techniques such as correlation analysis and Principal Component Analysis (PCA) can be used to explore and understand relationships within datasets, such as the "yalefaces" dataset. Finding the correlation matrix or applying the Singular Value Decomposition (SVD) are two methods of determining the relationship between given images in the dataset.

A correlation matrix gives the relative relationship between images through the dot product of the vectorization of the images, while the SVD determines the principal component vectors of the given data, the feature space which best defines a face. Both techniques can be used to illustrate the most and least correlated images, as well as the most shared features of a face. This can provide valuable insights into the underlying structure of the dataset and can be used to inform further analysis and modeling.

## Theoretical Background
Machine learning techniques such as correlation analysis and Singular Value Decomposition (SVD) can be used to explore and understand relationships within datasets. To find the correlation between two objects, they can be vectorized and then input into the dot product. Larger values in the resulting correlation matrix represent images that were evaluated by the dot product to share more features, while smaller values indicate less shared features, as explained by the equation:

<p align="left">
  $c_{jk} = x^{T}_{j}x_{k}$
</p>
The dot product can be applied to a set of images, resulting in a 2D array where each index is the correlation between the two corresponding images for that index.
To find the shared features among the faces in the dataset, the SVD can be used. The SVD will factorize a 2D matrix, returning the eigenvectors as rows in U, the singular values in the diagonal matrix S, and the eigenvectors as columns in V, as shown by the equation:

<p align="left">
 $X = U \Sigma V^{T}$
</p>
Plotting the eigenvectors for the largest found eigenvalues will visualize the feature space for the most prominent elements of each of the given images. This can provide valuable insights into the underlying structure of the dataset and can be used to inform further analysis and modeling.

## Algorithm Implementation and Development
As mentioned prior, this project implements the “yalefaces” dataset which is comprised of 2414 images with 1024 elements each.. The images were stored in matrix ‘X’, which was a 1024 x 2414 array.

```
results=loadmat('yalefaces.mat')
X=results['X']
print(X.shape)
```

### Part A
Part A aims to compute a 100x100 correlation matrix of the dot products of the first 100 images in the yalefaces dataset. I first separated out the first 100 images, then took the dot product of that array with its transpose.

```
# first 100 images into matrix X
data_100 = X[:, :100]
C = np.ndarray((100, 100))


# correlation matrix C, dot product (correlation) between first 100 images in X
C = np.dot(data_100.T, data_100)

```

### Part B
The goal of part B was to determine which two images from part A have the maximum and minimum correlations. Finding the first two pairs of images was fairly straight forward.

```
# most correlated
most = np.argwhere(C == np.max(C))[0]


# least correlated
least = np.argwhere(C == np.min(C))[0]
```
The first two pairs of images were transposes of the same image. ie: image1 dotted with the transpose of image1. To mitigate this, I looked for the next most/least correlated pairs of images.
```
# images are the same, look for next most/ least correlated
most = np.argwhere(C == np.sort(C.flatten())[-3])[0]
least = np.argwhere(C == np.sort(C.flatten())[1])[0]
```

### Part C
For part C, we were tasked with repeating part A but with a 10x10 correlation matrix and only for a subset of the images.

I first pulled out the subset of images from the yalefaces data.
```
# sample set
img_list = [1, 313, 512, 5, 2400, 113, 1024, 87, 314, 2005]


# sample images from data set
get_img = X[:, np.subtract(img_list, 1)]
```
Then I calculated the 10x10 correlation matrices.
```
# correlation matrix
C = np.ndarray((10, 10))
C = np.dot(get_img.T, get_img)
```

### Part D
Part D tasked us with using a correlation matrix to find the first six eigenvectors with the largest magnitude eigenvalue. I started by creating the matrix.

```
# create matrix Y
Y = np.dot(X, X.T)
```
I then used `scipy.sparse.linalg.eigs` to find the eigenvalues and eigenvectors.
```
w, v = scipy.sparse.linalg.eigs(Y, k=6, which="LM")
```

### Part E
For part E, I used `np.linalg.svd` to determine the first six principal component directions from the SVD.

```
# perform SVD on X
U, s, Vt = np.linalg.svd(X)


# find the first six principal component directions
pc_directions = U[:, :6]
```

### Part F
In part F, I compared the first eigenvector from part D with the first SVD Mode from part E. I did this comparison both visually and numerically.

First, I separated out the eigenvector and the SVD Mode.
```
v1 = np.real(v[:,0])
u1 = pc_directions[:,0]
```
Then I plotted them in order to compare them visually.
```
# plot the first eigenvector and SVD Mode
plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
img = v1.reshape(32,32)
plt.title('First Eigenvector v1')
plt.imshow(img)
plt.subplot(1,2,2)
img = u1.reshape(32,32)
plt.title('First SVD Mode u1')
plt.imshow(img)
plt.tight_layout()
plt.show()
```
Finally, I calculated the norm of the difference of their absolute values in order to compare them numerically.
```
norm = np.linalg.norm(np.abs(v1) - np.abs(u1))
print("Norm of difference of their absolute values:", norm)
```

### Part G
In part G, examined the first 6 SVD Modes by looking at the percentage of the variance between them.
```
var = s[:6] ** 2
var_ratio = var/np.sum(var) * 100
```
To compare them visually, I plotted them.
```
# plot the first 6 SVD modes
fig = plt.figure()
for i in range(6):
    mode = pc_directions[:, i].reshape((32, 32), order='F').T
    ax = fig.add_subplot(2, 3, i+1)
    ax.imshow(mode)
    plt.title("Mode " + str(i+1))
    plt.axis('off')
fig.suptitle("First 6 SVD Modes")
```

## Computational Results
### Part A

The results of the computation of the correlation between the first 100 images is best shown in the plot below.

![download](https://user-images.githubusercontent.com/129991497/232694211-07ba9999-053f-47b6-afbf-0ff4d93050eb.png "Fig 1. Plot of the Correlation between the first 100 images and their transposes")

### Part B
```
First Pair:
Most: [88 88] Least: [64 64]
Final Findings:
Most: [86 88] Least: [54 64]
```
![download](https://user-images.githubusercontent.com/129991497/232695105-52398645-acd0-4d70-81df-8d7f2ae1ef09.png "Fig 2. Two Most Correlated Images")

![download](https://user-images.githubusercontent.com/129991497/232695120-1e5611a7-d8dd-4874-9564-66a3dc89a716.png "Fig 3. Two Least Correlated Images")

### Part C

![download](https://user-images.githubusercontent.com/129991497/232695342-26f259c5-e0b5-4af7-9fb1-db8301edbe4e.png "Fig 4. Correlation between 10 images in the sample set")

### Part D

The first 6 eigenvectors with the lowest eigenvalues and their corresponding eigenvalues.
```
Eigenvalue: 234020.4548538858
Eigenvector: [0.02384327 0.02576146 0.02728448 ... 0.02082937 0.0193902  0.0166019 ]
Eigenvalue: 49038.31530059216
Eigenvector: [ 0.04535378  0.04567536  0.04474528 ... -0.03737158 -0.03557383
 -0.02965746]
Eigenvalue: 8236.539897013148
Eigenvector: [-0.05653196 -0.04709124 -0.0362807  ... -0.06455006 -0.06196898
 -0.05241684]
Eigenvalue: 6024.871457930157
Eigenvector: [-0.04441826 -0.05057969 -0.05522219 ...  0.01006919  0.00355905
 -0.00040934]
Eigenvalue: 2051.496432691054
Eigenvector: [-0.03378603 -0.01791442 -0.00462854 ...  0.06172201  0.05796353
  0.05757412]
Eigenvalue: 1901.079114823662
Eigenvector: [0.02207542 0.03378819 0.04487476 ... 0.03025485 0.02850199 0.00941028]
```

### Part E

The first 6 principal component vectors of the SVD.
```
[[-0.02384327 -0.04535378 -0.05653196  0.04441826 -0.03378603  0.02207542]
 [-0.02576146 -0.04567536 -0.04709124  0.05057969 -0.01791442  0.03378819]
 [-0.02728448 -0.04474528 -0.0362807   0.05522219 -0.00462854  0.04487476]
 ...
 [-0.02082937  0.03737158 -0.06455006 -0.01006919  0.06172201  0.03025485]
 [-0.0193902   0.03557383 -0.06196898 -0.00355905  0.05796353  0.02850199]
 [-0.0166019   0.02965746 -0.05241684  0.00040934  0.05757412  0.00941028]]
 ```

### Part F

![download](https://user-images.githubusercontent.com/129991497/232695923-26c4e8b6-7fe6-47b7-b2ae-aab5d6270f51.png "Fig 5. Comparison of the First Eigenventor and the first SVD Mode")

Numerical comparison of the First Eigenventor and the first SVD Mode.
```
v1:
 [0.02384327 0.02576146 0.02728448 ... 0.02082937 0.0193902  0.0166019 ]
u1:
 [-0.02384327 -0.02576146 -0.02728448 ... -0.02082937 -0.0193902
 -0.0166019 ]
Norm of difference of their absolute values: 6.742051385122386e-16
```

### Part G
The variance of the first 6 SVD Modes.
```
Percentage of variance captured by each of the first 6 SVD modes:
SVD Mode 1: 77.677% Var
SVD Mode 2: 16.277% Var
SVD Mode 3: 2.734% Var
SVD Mode 4: 2.0% Var
SVD Mode 5: 0.681% Var
SVD Mode 6: 0.631% Var
```

![download](https://user-images.githubusercontent.com/129991497/232696261-4d515db1-decb-4b36-b28f-160af6cdef3e.png "Fig 6. Plots of the First 6 SVD Modes")

## Summary and Conclusions
Through variations of correlation matrices and SVDs, we were able to model the variations and the correlations of the images contained withing the yalefaces dataset.
