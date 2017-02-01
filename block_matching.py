# -*- coding: utf-8 -*-
"""
Created on Fri Jan 27 17:53:22 2017

@author: Group 3
"""

import cv2
import configuration as conf
import numpy as np
import sys
sys.path.append('../')

def camera_motion(real_x, real_y, curr_img):
    x_size = curr_img.shape[0]
    y_size = curr_img.shape[1]
    block_size = conf.block_size
    area_size = conf.area_size

    x_blocks = x_size/block_size
    y_blocks = y_size/block_size

    if curr_img.shape.__len__() == 2:
        new_curr_img = np.zeros([x_size + 2 * area_size, y_size + 2 * area_size])
        new_curr_img[area_size:area_size + x_size, area_size:area_size + y_size] = curr_img
        comp_img = np.zeros([x_size, y_size])
    elif curr_img.shape.__len__() == 3:
        new_curr_img = np.zeros([x_size + 2 * area_size, y_size + 2 * area_size, 3])
        auxImg = np.pad(curr_img[:, :, 0], ((area_size, area_size), (area_size, area_size)), 'symmetric')
        new_curr_img[:,:,0] = auxImg
        auxImg = np.pad(curr_img[:, :, 1], ((area_size, area_size), (area_size, area_size)), 'symmetric')
        new_curr_img[:, :, 1] = auxImg
        auxImg = np.pad(curr_img[:, :, 2], ((area_size, area_size), (area_size, area_size)), 'symmetric')
        new_curr_img[:, :, 2] = auxImg
        comp_img = np.zeros([x_size, y_size, 3])
    else:
        print 'ERROR dimension'
        return curr_img

    for x_pos in range(x_blocks):
        for y_pos in range(y_blocks):
            # +y or -y depending on compensation mode
            if conf.compensation == 'backward':
                if curr_img.shape.__len__() == 2:
                    comp_img[x_pos * block_size:x_pos * block_size + block_size,y_pos * block_size:y_pos * block_size + block_size] = new_curr_img[x_pos * block_size + real_x + area_size:x_pos * block_size + block_size + real_x + area_size,y_pos * block_size - real_y + area_size:y_pos * block_size + block_size - real_y + area_size]

                elif curr_img.shape.__len__() == 3:
                    comp_img[x_pos * block_size:x_pos * block_size + block_size,
                    y_pos * block_size:y_pos * block_size + block_size, 0] = new_curr_img[
                                                                             x_pos * block_size + real_x + area_size:x_pos * block_size + block_size + real_x + area_size,
                                                                             y_pos * block_size - real_y + area_size:y_pos * block_size + block_size - real_y + area_size, 0]
                    comp_img[x_pos * block_size:x_pos * block_size + block_size,
                    y_pos * block_size:y_pos * block_size + block_size, 1] = new_curr_img[
                                                                             x_pos * block_size + real_x + area_size:x_pos * block_size + block_size + real_x + area_size,
                                                                             y_pos * block_size - real_y + area_size:y_pos * block_size + block_size - real_y + area_size, 1]
                    comp_img[x_pos * block_size:x_pos * block_size + block_size,
                    y_pos * block_size:y_pos * block_size + block_size, 2] = new_curr_img[
                                                                             x_pos * block_size + real_x + area_size:x_pos * block_size + block_size + real_x + area_size,
                                                                             y_pos * block_size - real_y + area_size:y_pos * block_size + block_size - real_y + area_size, 2]

            else:
                if curr_img.shape.__len__() == 2:
                    comp_img[x_pos * block_size:x_pos * block_size + block_size,
                    y_pos * block_size:y_pos * block_size + block_size] = new_curr_img[
                                                                          x_pos * block_size + real_x + area_size:x_pos * block_size + block_size + real_x + area_size,
                                                                          y_pos * block_size + real_y + area_size:y_pos * block_size + block_size + real_y + area_size]
                elif curr_img.shape.__len__() == 3:
                    comp_img[x_pos * block_size:x_pos * block_size + block_size,
                    y_pos * block_size:y_pos * block_size + block_size, 0] = new_curr_img[
                                                                          x_pos * block_size + real_x + area_size:x_pos * block_size + block_size + real_x + area_size,
                                                                          y_pos * block_size + real_y + area_size:y_pos * block_size + block_size + real_y + area_size, 0]

                    comp_img[x_pos * block_size:x_pos * block_size + block_size,
                    y_pos * block_size:y_pos * block_size + block_size, 1] = new_curr_img[
                                                                          x_pos * block_size + real_x + area_size:x_pos * block_size + block_size + real_x + area_size,
                                                                          y_pos * block_size + real_y + area_size:y_pos * block_size + block_size + real_y + area_size, 1]

                    comp_img[x_pos * block_size:x_pos * block_size + block_size,
                    y_pos * block_size:y_pos * block_size + block_size, 2] = new_curr_img[
                                                                          x_pos * block_size + real_x + area_size:x_pos * block_size + block_size + real_x + area_size,
                                                                          y_pos * block_size + real_y + area_size:y_pos * block_size + block_size + real_y + area_size, 2]

    return comp_img
    
def create_compensated_image(prev_img, motion_matrix, block_size, x_blocks, y_blocks):
    x_size = prev_img.shape[0]
    y_size = prev_img.shape[1]

    comp_img = np.zeros([x_size, y_size])

    for x_pos in range(x_blocks):
        for y_pos in range(y_blocks):
            comp_img[x_pos*block_size:x_pos*block_size+block_size,y_pos*block_size:y_pos*block_size+block_size]=prev_img[x_pos*block_size-motion_matrix[x_pos,y_pos,0]:x_pos*block_size+block_size-motion_matrix[x_pos,y_pos,0],y_pos*block_size+motion_matrix[x_pos,y_pos,1]:y_pos*block_size+block_size+motion_matrix[x_pos,y_pos,1]]
    return comp_img

def compute_error(block1, block2):
    return sum(sum(abs(block1-block2)**2))

def block_search(region_to_explore, block_to_search):
    block_size = conf.block_size
    area_size = conf.area_size
    x_size = region_to_explore.shape[0]
    y_size = region_to_explore.shape[1]

    min_diff = sys.float_info.max

    for row in range(x_size-area_size):
        for column in range(y_size-area_size):
            block2analyse = region_to_explore[row:row+block_size, column:column+block_size]
            diff = compute_error(block2analyse, block_to_search)
            if diff < min_diff:
                min_diff = diff
                x_mot = - row + area_size
                y_mot = column - area_size
    return x_mot, y_mot
    
def compute_block_matching(prev_img, curr_img):
    block_size = conf.block_size
    area_size = conf.area_size
    compensation = conf.compensation
    #We will apply backward compensation
    if compensation=='backward':
        img2xplore = curr_img
        searchimg = prev_img
    else:
        img2xplore = prev_img
        searchimg = curr_img
    
    x_blocks = img2xplore.shape[0]/block_size
    y_blocks = img2xplore.shape[1]/block_size

    #Add padding in the search image
    pad_searchimg = np.zeros([img2xplore.shape[0]+2*area_size,img2xplore.shape[1]+2*area_size])
    pad_searchimg[area_size:area_size+img2xplore.shape[0],area_size:area_size+img2xplore.shape[1]] = searchimg[:,:]

    motion_matrix = np.zeros([x_blocks, y_blocks, 2])
    
    for row in range(x_blocks):
        for column in range(y_blocks):
            # print "Computing block " + str(column)
            block_to_search = img2xplore[row*block_size:row*block_size+block_size, column*block_size:column*block_size+block_size]
            region_to_explore = pad_searchimg[row*block_size:row*block_size+block_size+2*area_size, column*block_size:column*block_size+block_size+2*area_size]
            x_mot, y_mot = block_search(region_to_explore, block_to_search)
            
            motion_matrix[row,column,0] = x_mot
            motion_matrix[row,column,1] = y_mot
            
    return motion_matrix
    
if __name__ == "__main__":
    print 'block matching method'
    
