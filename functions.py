import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os

RED = '\033[91m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'


def read_images(path):
    """
    function to read all images inside a directory.
    |~~~~~~~~~~|
    |DISCLAIMER|, the images that will be used as input, will need to have the same size. Also they need to be RGB and not in 
    |~~~~~~~~~~|  any other colourscale


    input: Folder name (path)
    output: np.array() of images. The array will have the size of how many images exist inside the folder directory. E.g.
            if inside the folder reside 3 images, then our images_array will have a size of 3.
    """
    
    dir_list = os.listdir(path)
    n_img = len(dir_list)

    img_dir = []
    full_path = []
    for i in range(n_img):
        suffix = dir_list[i][:-3]
        if "nb" not in suffix:
            img_dir.append(dir_list[i])
            full_path.append(path + "/" + dir_list[i])
        
    print(f"Found {n_img}, images in folder: {img_dir}")
    print(f"Full path {full_path}")
    #print(f"Images found: {dir_list}")
    images_array = np.array([imageio.imread(f) for f in full_path])

    return images_array

def SIFT_feats(images_array):
    """
    function to calculate SIFT keypoints and Descriptors using openCV's built-in functions

    input: numpy array of images
    output: list containing keypoints and descriptors for every image
    """
    
    siftDetector= cv2.SIFT_create()
    n_imgs = len(images_array)
    
    print(44*"~")
    print(f"{BLUE}Disclaimer! SIFT works better with Grayscale Images. Use RGB at your own caution{RESET}")
    k_points = []
    descriptors = []
    for i in range(n_imgs):
    
        kp, des = siftDetector.detectAndCompute(images_array[i], None)

        k_points.append(kp)
        descriptors.append(des)
        print(f"For Image no.{i+1}, we found {len(kp)} keypoints and {len(des)} descriptors")
    print(44*"~")
    return k_points, descriptors

def matcher_(descriptor1,descriptor2):

    """
    function to find matches between 2 images

    input: numpy arrays of descriptors
    output: matches, a opencv type of object
    """

    
    # using Brute-Force Matcher as seen on opencv: https://docs.opencv.org/3.4/dc/dc3/tutorial_py_matcher.html
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptor1,descriptor2, k=2)

    good = []
    for m,n in matches:
        if m.distance < 0.75*n.distance:  # Using Lowes treshold
            good.append([m])

    return(good)

def coordinates_tr(matches,keypoints_list):
    
    """
    function to return keypoints as coordinates and not as objects

    input: a list of matches and a list of 2 images keypoints
    output: numpy array of coordinates, in order to create Homography Matrix
    """
    
    
    
    n_imgs = len(keypoints_list)
    if n_imgs!=2:
        print(f"{RED}We need 2 images and not more, ERROR INCOMING{RESET}")

    keypoints1 = keypoints_list[0]
    keypoints2 = keypoints_list[1]

    kp1 = []
    kp2 = []
    
    for match in matches:
        points1 = list(keypoints1[match[0].queryIdx].pt)
        points2 = list(keypoints2[match[0].trainIdx].pt)
        
        kp2.append(points2)
        kp1.append(points1)
    
    kp1 = np.array(kp1)
    kp2 = np.array(kp2)
    
    return kp1,kp2

def con_ls(equation):
    """
    function to solve the equation using Constrained Least Squares

    input: numpy array, consisting of Matrix A
    output: numpy array of non-normalized Homography matrix
    """


    
    transposed_eq = equation.T
    matrix = np.dot(transposed_eq,equation) 

    eig_vals, eig_vecs = np.linalg.eigh(matrix)
    H = eig_vecs[:,np.argmin(eig_vals)]

    return H


def homography_mat(coordinates1,coordinates2):
    
    """
    function to compute the Homography Matrix
    
    input: a list of matches and a list of 2 images keypoints 
    output: numpy array of the Homography matrix
    """    
    #here our keypoints are in a opencv dtype, we need to transform them to actual coordinates!

    n_points = len(coordinates1) # we have the same number of points, so no problem will occur

    """
    Homography formula:

    [x2]   [h11 h12 h13]   [x1]
    [y2] = [h21 h22 h23] * [y1]
    [ 1]   [h31 h32 h33]   [ 1]

    
    System to be solved:

    [x1, y1, 1,   0,   0,  0,  -x2*x1,  -x2*y1,  -x2] = 0
    [  0,   0,  0, x1, y1, 1,  -y2*x1,  -y2*y1,  -y2]
    """
    equation = []
    for i in range(n_points):
        x1,y1 = coordinates1[i]
        x2,y2 = coordinates2[i]

        z1 = 1
        z2 = 1

        eq1 =  [x1, y1, z1, 0, 0, 0, -x2*x1, -x2*y1, -x2]
        eq2 =  [0, 0, 0, x1, y1, z2, -y2*x1, -y2*y1, -y2]
        equation.append(eq1)
        equation.append(eq2)

    equation = np.array(equation)
    H = con_ls(equation)
    H = H.reshape(3,3)
    H = H/H[2, 2]

    
    return H

def error_func(cords1,cords2, h_matrix):
    """
    Function to compute RANSAC ERROR. It is a simple RMSE error, utilizing the extended points nad homography matrix

    inputs: Cordinates of keypoints & homography matrix
    output: RMSE (numpy array of errors)
    """
    
    n_points = len(cords1)

    
    cords1 = np.hstack((cords1, np.ones((n_points, 1))))
    cords2 = np.hstack((cords2, np.ones((n_points, 1))))
    estimated_points = np.zeros_like(cords2)

    
    for i in range(len(cords1)):
        estimated_points[i] = np.dot(h_matrix, cords1[i].T)

    # RMSE
    error_ = np.linalg.norm(cords2 - estimated_points , axis=1) ** 2

    return error_

def RANSAC(threshold, matches, keypoints_list,epochs):
    """
    Function to apply RANSAC algorithm    

    inputs: 1. Threshold, in order to get all the points where the error is less than the treshold (acceptable points)
            2. List of matches based on keypoints
            3. Keypoints list
            4. Epochs, for how many iterations we shall run the algorithm

    outputs: The best Homography matrix corresponding to these points
    """

    n_best_inliers = 0

    # get keypoints coordinates
    coordinates1,coordinates2 = coordinates_tr(matches, keypoints_list)
    n_points = len(coordinates1)
    print(f"Number of points {n_points}")
    
    for i in range(epochs):
        # grab 4 random points to use as input in Homography matrix
        idx = np.random.choice(n_points, 4, replace=False)
        points_1 = coordinates1[idx]
        points_2 = coordinates2[idx]
        
        H = homography_mat(points_1,points_2)

        error = error_func(coordinates1,coordinates2, H)

        # The Error array has the same size of the filtered keypoints (our coordinates_tr function if noticed carefully)
        # returns the coordinates of the matched points. So what we need, is to find the index of the points that have via the 
        # error function
        
        inliers_idx = np.where(error < threshold)[0]
        inliers_1 = coordinates1[inliers_idx]
        inliers_2 = coordinates2[inliers_idx]
        total_inliers = np.concatenate((inliers_1,inliers_2))
        
        # Total inliers is an array of shape: )N,2). The keypoints of the first image are stored in the first N//2 values
        # and in the last N//2 values the keypoints for the second image are stored

        if len(total_inliers) > n_best_inliers:
            best_inliers = total_inliers
            n_best_inliers = len(total_inliers)
            best_H = H
    
    print(f"Out of {n_points}, we deemed as inliers {n_best_inliers}")
    return best_H


def translation_mat_resize(hom_matrix, image_sample):
    """
    function to compute translation matrix (translation matrix along x,y-axis) to apply homography matrix to image and 
    to resize the images correctly

    T = [1, 0, dx]
        [0, 1, dy]
        [0, 0, 1]

    dx,dy denote how much we want to resize our image
    """
    height, width, _ = image_sample.shape

    # Make new margins-resize image
    margins = np.array([[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]])
    corners = [np.dot(hom_matrix, margin) for margin in margins]
    corners = np.array(corners).T 
    x_new = corners[0] / corners[2]
    y_new = corners[1] / corners[2]
    y_min = min(y_new)
    x_min = min(x_new)

    #size = (x_new, y_new)
    # Translated Homography Matrix
    t_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    h_new = np.dot(t_mat,hom_matrix)

    return x_min,y_min,t_mat,h_new
    
def wrap(image1,image2,H):
    x_1,y_1,t_mat,H = translation_mat_resize(H,image1)
    
    size = (int(round(abs(x_1) + image1.shape[1])) ,int(round(abs(y_1) + image1.shape[0]))  )
    
    wraped1 = cv2.warpPerspective(src=image1, M=H, dsize=size)
    wraped2 = cv2.warpPerspective(src=image2, M=t_mat, dsize=size)
    
    return wraped1,wraped2

def simple_stitching(image1,image2):
    foreground = image2.copy()
    background = image1.copy()
    
    overlay_img = foreground[:, :, :3]  # Grab the BRG planes

    res = background

    only_right = np.nonzero((np.sum(overlay_img, 2) != 0) * (np.sum(background,2) == 0))
    left_and_right = np.nonzero((np.sum(overlay_img, 2) != 0) * (np.sum(background,2) != 0))

    res[only_right] = overlay_img[only_right]
    res[left_and_right] = res[left_and_right]*0.5 + overlay_img[left_and_right]*0.5

    return res,res


def gaussian_laplacian_stitching(image1, image2):
    A = image1.copy()
    B = image2.copy()

    # Generate Gaussian pyramid for A
    gpA = [A]
    for i in range(6):
        G = cv2.pyrDown(gpA[-1])
        gpA.append(G)

    # Generate Gaussian pyramid for B
    gpB = [B]
    for i in range(6):
        G = cv2.pyrDown(gpB[-1])
        gpB.append(G)

    # Generate Laplacian pyramid for A
    lpA = [gpA[-1]]
    for i in range(5, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        # Resize GE to match the shape of gpA[i-1] (to handle rounding differences)
        GE = cv2.resize(GE, (gpA[i - 1].shape[1], gpA[i - 1].shape[0]))
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)

    # Generate Laplacian pyramid for B
    lpB = [gpB[-1]]
    for i in range(5, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        # Resize GE to match the shape of gpB[i-1]
        GE = cv2.resize(GE, (gpB[i - 1].shape[1], gpB[i - 1].shape[0]))
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)

    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, :cols // 2], lb[:, cols // 2:]))
        LS.append(ls)

    # Reconstruct the image
    ls_ = LS[0]
    for i in range(1, 6):
        ls_ = cv2.pyrUp(ls_)
        # Resize ls_ to match the current level's shape
        ls_ = cv2.resize(ls_, (LS[i].shape[1], LS[i].shape[0]))
        ls_ = cv2.add(ls_, LS[i])

    # Image with direct connecting each half
    cols = A.shape[1]
    real = np.hstack((A[:, :cols // 2], B[:, cols // 2:]))

    return real, ls_