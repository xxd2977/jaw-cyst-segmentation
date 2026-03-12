import torchvision
import os
import torch
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.imgname import read_img_name
import seaborn as sns


def visual_segmentation(seg, image_filename, opt):
    img_ori = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                              [144, 255, 144], [0, 215, 255], [96, 164, 244], [128, 128, 240], [250, 206, 135]])
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        # img_r[seg0 == i] = table[i - 1, 0]
        # img_g[seg0 == i] = table[i - 1, 1]
        # img_b[seg0 == i] = table[i - 1, 2]
        img_r[seg0 == i] = table[i + 1 - 1, 0]
        img_g[seg0 == i] = table[i + 1 - 1, 1]
        img_b[seg0 == i] = table[i + 1 - 1, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
    #img = cv2.addWeighted(img_ori0, 0.6, overlay, 0.4, 0) 
    img = cv2.addWeighted(img_ori0, 0.5, overlay, 0.5, 0) 
    #img = np.uint8(0.3 * overlay + 0.7 * img_ori)
          
    fulldir = opt.visual_result_path + "/" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, img)


def visual_segmentation_sets(seg, image_filename, opt):
    img_path = os.path.join(opt.data_subpath + '/img', image_filename)
    img_ori = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_subpath + '/img', image_filename))
    img_ori = cv2.resize(img_ori, dsize=(256, 256))
    img_ori0 = cv2.resize(img_ori0, dsize=(256, 256))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[96, 164, 244], [193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                              [144, 255, 144], [0, 215, 255], [128, 128, 240], [250, 206, 135]])
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        img_r[seg0 == i] = table[i - 1, 0]
        img_g[seg0 == i] = table[i - 1, 1]
        img_b[seg0 == i] = table[i - 1, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
 
    img = cv2.addWeighted(img_ori0, 0.4, overlay, 0.6, 0) 
    #img = img_ori0
          
    fulldir = opt.result_path + "/" + opt.modelname + "/"
    #fulldir = opt.result_path + "/" + "GT" + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, img)

def visual_segmentation_sets_with_pt1(seg, image_filename, opt, pt):

    img_path = os.path.join(opt.data_subpath + '/img', image_filename)
    img_ori = cv2.imread(img_path)
    img_ori0 = cv2.imread(img_path)

    img_ori = cv2.resize(img_ori, dsize=(256, 256))
    img_ori0 = cv2.resize(img_ori0, dsize=(256, 256))

 
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[96, 164, 244], [193, 182, 255], [219, 112, 147], [237, 149, 100], 
                      [211, 85, 186], [204, 209, 72], [144, 255, 144], [0, 215, 255], 
                      [128, 128, 240], [250, 206, 135]])
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        img_r[seg0 == i] = table[i - 1, 0]
        img_g[seg0 == i] = table[i - 1, 1]
        img_b[seg0 == i] = table[i - 1, 2]
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)

  
    img = cv2.addWeighted(img_ori0, 0.4, overlay, 0.6, 0) 
    fulldir = opt.result_path + "/PT10-" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, img)

    
    binary_result = (seg0 > 0).astype(np.uint8) * 255  # 二值图像，0表示背景，255表示分割区域
    binary_dir = opt.result_path + "/PT10-" + opt.modelname + "-binary/"
    if not os.path.isdir(binary_dir):
        os.makedirs(binary_dir)
    cv2.imwrite(os.path.join(binary_dir, image_filename), binary_result)

def visual_segmentation_sets_with_pt(seg, image_filename, opt, pt, heatmap):

    img_path = os.path.join(opt.data_subpath + '/img', image_filename)
    img_ori = cv2.imread(img_path)
    img_ori0 = cv2.imread(img_path)
    img_ori = cv2.resize(img_ori, dsize=(256, 256))
    img_ori0 = cv2.resize(img_ori0, dsize=(256, 256))

   
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    table = np.array([[96, 164, 244], [193, 182, 255], [219, 112, 147], [237, 149, 100], [211, 85, 186], [204, 209, 72],
                              [144, 255, 144], [0, 215, 255], [128, 128, 240], [250, 206, 135]])
    seg0 = seg[0, :, :]
    
    for i in range(1, opt.classes):
        img_r[seg0 == i] = table[i - 1, 0]
        img_g[seg0 == i] = table[i - 1, 1]
        img_b[seg0 == i] = table[i - 1, 2]
    
    overlay[:, :, 0], overlay[:, :, 1], overlay[:, :, 2] = img_r, img_g, img_b
    overlay = np.uint8(overlay)
    img = cv2.addWeighted(img_ori0, 0.4, overlay, 0.6, 0)


    fulldir = opt.result_path + "/PT10-" + opt.modelname + "/"
    os.makedirs(fulldir, exist_ok=True)
    cv2.imwrite(fulldir + image_filename, img)

 
    binary_result = (seg0 > 0).astype(np.uint8) * 255
    binary_dir = opt.result_path + "/PT10-" + opt.modelname + "-binary/"
    os.makedirs(binary_dir, exist_ok=True)
    cv2.imwrite(os.path.join(binary_dir, image_filename), binary_result)

    
    if heatmap is not None:
      
        heatmap = cv2.resize(heatmap, (img_ori.shape[1], img_ori.shape[0]))
        
       
        heatmap_normalized = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap_normalized), cv2.COLORMAP_JET)
        
       
        heatmap_overlay = cv2.addWeighted(img_ori, 0.5, heatmap_colored, 0.5, 0)
       
        heatmap_dir = opt.result_path + "/PT10-" + opt.modelname + "-heatmap/"
        os.makedirs(heatmap_dir, exist_ok=True)
        
       
        cv2.imwrite(os.path.join(heatmap_dir, f"heatmap_{image_filename}"), heatmap_colored)
  
        cv2.imwrite(os.path.join(heatmap_dir, f"overlay_heatmap_{image_filename}"), heatmap_overlay)
def visual_segmentation_binary(seg, image_filename, opt):
    img_ori = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    img_ori0 = cv2.imread(os.path.join(opt.data_path + '/img', image_filename))
    overlay = img_ori * 0
    img_r = img_ori[:, :, 0]
    img_g = img_ori[:, :, 1]
    img_b = img_ori[:, :, 2]
    seg0 = seg[0, :, :]
            
    for i in range(1, opt.classes):
        img_r[seg0 == i] = 255
        img_g[seg0 == i] = 255
        img_b[seg0 == i] = 255
            
    overlay[:, :, 0] = img_r
    overlay[:, :, 1] = img_g
    overlay[:, :, 2] = img_b
    overlay = np.uint8(overlay)
          
    fulldir = opt.visual_result_path + "/" + opt.modelname + "/"
    if not os.path.isdir(fulldir):
        os.makedirs(fulldir)
    cv2.imwrite(fulldir + image_filename, overlay)
