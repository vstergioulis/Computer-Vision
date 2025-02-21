import cv2
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import os
import argparse
from functions import *

def choose_blending_method(method_):
    choosing = 0
    
    if method_ == 'simple':
        methodman = simple_stitching
        choosing = 0
    else :
        methodman = gaussian_laplacian_stitching
        if 'l' in method_:
            choosing = 1
    return methodman, choosing



def main():
    
    GREEN = '\033[92m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    
    
    parser = argparse.ArgumentParser(description="Image Stitching")
    
    parser.add_argument("--folder", type=str, default="images", help="Path to the folder containing images.")
    parser.add_argument("--RANSAC_thres", type=float, default= 0.5,help="Threshold to use for RANSAC algorithm.")
    parser.add_argument("--RANSAC_epochs", type=int, default= 2000,help="Epochs to use for RANSAC algorithm.")
    parser.add_argument("--blending", type=str,choices=["simple", "gaussian", "laplacian"], default="gaussian",
                        help="Blending technique to use ('simple','gaussian' or 'laplacian).")

    args = parser.parse_args()

    #output_dir = "result"
    output_dir = os.path.join("result", os.path.basename(args.folder))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    print(f"{BLUE}Results may vary{RESET}")
    images = read_images(os.path.join(args.folder, ""))
    technique, choice = choose_blending_method(args.blending)
    # Initialization, take the first two images
    points_init,descriptor_init = SIFT_feats(images[:2])
    matches_init = matcher_(descriptor_init[0],descriptor_init[1])
    
    H_mat = RANSAC(args.RANSAC_thres, matches_init, points_init, args.RANSAC_epochs)
    image_left , image_right =  wrap(images[0], images[1], H_mat)
    
    stiched_img, laplacian_stiched = technique(image_left,image_right)
    
    if choice == 1:
        stiched_img = laplacian_stiched
    if len(images) > 2:
        
        print(44*"~")
        
        for i in range(2,len(images)):
            print(f"Stiching for the remaining {len(images) - i} images")
            images2 = [stiched_img]
            images2.extend(images[2:])
            
            points,descriptor = SIFT_feats(images2[:2])
            matches_ = matcher_(descriptor[0],descriptor[1])
            H_mat = RANSAC(args.RANSAC_thres, matches_, points, args.RANSAC_epochs)
            image_left , image_right =  wrap(images2[0], images2[1], H_mat)
            
            stiched_img, laplacian_stiched = technique(image_left,image_right)
    
            if choice == 1:
                stiched_img = laplacian_stiched
        
        print(44*"~")
    print(f"{GREEN}Panorama is Ready{RESET}")
    t = os.path.join(output_dir, f"panorama_{args.blending}_rgb.jpg")
    cv2.imwrite(t,cv2.cvtColor(stiched_img, cv2.COLOR_RGB2BGR))
    
if __name__ == "__main__":

    main()