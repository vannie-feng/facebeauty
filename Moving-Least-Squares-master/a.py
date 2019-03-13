#! /usr/bin/env python
# -*- coding: utf-8 -*-

from sys import exit
import os
import sys
import numpy as np
from PIL import Image, ImageDraw
from img_utils import (mls_affine_deformation, mls_affine_deformation_inv,
                       mls_similarity_deformation, mls_similarity_deformation_inv,
                       mls_rigid_deformation, mls_rigid_deformation_inv)
import face_recognition as fr
from copy import deepcopy

FNAME = 'b.jpg'


def demo(fun, fun_inv, name):
    p = np.array([
        [30, 155], [125, 155], [225, 155],
        [100, 235], [160, 235], [85, 295], [180, 293]
    ])
    q = np.array([
        [42, 211], [125, 155], [235, 100],
        [80, 235], [140, 235], [85, 295], [180, 295]
    ])
    image = Image.open(os.path.join(sys.path[0], "mr_big_ori.jpg"))
    image = np.array(image) / 255.0

    if fun is not None:
        transformed_image = fun(image, p, q, alpha=1, density=1)
        transformed_image = (transformed_image * 255).astype(np.uint8)
        Image.fromarray(transformed_image).save('11.jpg')
        transformed_image = fun(image, p, q, alpha=1, density=0.7)
        transformed_image = (transformed_image * 255).astype(np.uint8)
        Image.fromarray(transformed_image).save('12.jpg')
    if fun_inv is not None:
        transformed_image = fun_inv(image, p, q, alpha=1, density=1)
        transformed_image = (transformed_image * 255).astype(np.uint8)
        Image.fromarray(transformed_image).save('13.jpg')
        transformed_image = fun_inv(image, p, q, alpha=1, density=0.7)
        transformed_image = (transformed_image * 255).astype(np.uint8)
        Image.fromarray(transformed_image).save('14.jpg')

def demo2(fun):
    ''' 
        Smiled Monalisa  
    '''
    
    p = np.array([
        [186, 140], [295, 135], [208, 181], [261, 181], [184, 203], [304, 202], [213, 225], 
        [243, 225], [211, 244], [253, 244], [195, 254], [232, 281], [285, 252]
    ])
    q = np.array([
        [186, 140], [295, 135], [208, 181], [261, 181], [184, 203], [304, 202], [213, 225], 
        [243, 225], [207, 238], [261, 237], [199, 253], [232, 281], [279, 249]
    ])
    image = Image.open(os.path.join(sys.path[0], "monalisa.jpg"))
    image = np.array(image) / 255.0
    transformed_image = fun(image, p, q, alpha=1, density=1)
    transformed_image = (transformed_image * 255).astype(np.uint8)
    Image.fromarray(transformed_image).save('2.jpg')


if __name__ == "__main__":

    # affine deformation
    # demo(mls_affine_deformation, mls_affine_deformation_inv, "Affine")
    # demo2(mls_affine_deformation_inv)

    # similarity deformation
    # demo(mls_similarity_deformation, mls_similarity_deformation_inv, "Similarity")
    # demo2(mls_similarity_deformation_inv)

    # rigid deformation
    # demo(mls_rigid_deformation, mls_rigid_deformation_inv, "Rigid")
    # demo2(mls_rigid_deformation_inv)

    image = fr.load_image_file(FNAME)
    face_locations = fr.face_locations(image)
    fls = [(x[3], x[0], x[1], x[2]) for x in face_locations]
    face_landmarks_list = fr.face_landmarks(image)
    # print face_landmarks_list
    t_image = image / 255.0

    for face_landmarks in face_landmarks_list:

        nx, ny = face_landmarks['nose_bridge'][-1]
        chin = face_landmarks['chin']
        s_chin = []
        for i, (x, y) in enumerate(chin):

            k = 0.1
            if i in (3, 4, 5, 12, 13, 14):
                k = 0.12
            sx = x + (nx - x) * k
            sy = y + (ny - y) * k
            s_chin.append((int(sx), int(sy)))

        face_landmarks['s_chin'] = s_chin

        left_eye = face_landmarks['left_eye']
        left_eye_center = [tuple(np.array(left_eye).mean(axis=0).astype(np.uint).tolist())]
        lx, ly = left_eye_center[-1]
        face_landmarks['left_eye_center'] = left_eye_center
        b_left_eye = []
        for i, (x, y) in enumerate(left_eye):

            k = 0.005
            bx = x + (x - lx) * k
            by = y + (y - ly) * k
            b_left_eye.append((int(bx), int(by)))

        face_landmarks['b_left_eye'] = b_left_eye

        right_eye = face_landmarks['right_eye']
        right_eye_center = [tuple(np.array(right_eye).mean(axis=0).astype(np.uint).tolist())]
        rx, ry = right_eye_center[-1]
        face_landmarks['right_eye_center'] = right_eye_center
        b_right_eye = []
        for i, (x, y) in enumerate(right_eye):

            k = 0.005
            bx = x + (x - rx) * k
            by = y + (y - ry) * k
            b_right_eye.append((int(bx), int(by)))

        face_landmarks['b_right_eye'] = b_right_eye

        t_image = mls_rigid_deformation_inv(t_image,
                np.array(chin + left_eye + right_eye),
                np.array(s_chin + b_left_eye + b_right_eye),
                alpha=1, density=1)

    image = (t_image * 255).astype(np.uint8)
    pil_image = Image.fromarray(image)
    draw = ImageDraw.Draw(pil_image)
    ## draw.rectangle(fls[0], outline=(255, 0, 0))
    for l, t, r, d, in fls:

        draw.ellipse((l - 2, t - 2, l + 2, t + 2), fill=(255, 255, 0))
        draw.ellipse((r - 2, t - 2, r + 2, t + 2), fill=(255, 255, 0))
        draw.ellipse((l - 2, d - 2, l + 2, d + 2), fill=(255, 255, 0))
        draw.ellipse((r - 2, d - 2, r + 2, d + 2), fill=(255, 255, 0))

    for face_landmarks in face_landmarks_list:

        for key, values in face_landmarks.iteritems():

            for x, y in values:

                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(255, 255, 255))

        nx, ny = face_landmarks['nose_bridge'][-1]
        draw.ellipse((nx - 2, ny - 2, nx + 2, ny + 2), fill=(255, 0, 0))
        nx, ny = face_landmarks['left_eye_center'][-1]
        draw.ellipse((nx - 2, ny - 2, nx + 2, ny + 2), fill=(255, 0, 0))
        nx, ny = face_landmarks['right_eye_center'][-1]
        draw.ellipse((nx - 2, ny - 2, nx + 2, ny + 2), fill=(255, 0, 0))
        for x, y in face_landmarks['b_left_eye']:

            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 0, 255))

        for x, y in face_landmarks['b_right_eye']:

            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(0, 0, 255))

    pil_image.save('c1.jpg')


    exit(0)
