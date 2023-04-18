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



### Part B



### Part C



### Part D



### Part E



### Part F



### Part G


