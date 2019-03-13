#! /usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image deformation using moving least squares

@author: Jarvis ZHANG
@date: 2017/8/8
@editor: VS Code
"""

from sys import exit
import os
import sys
import numpy as np
from PIL import Image
from img_utils import (mls_affine_deformation, mls_affine_deformation_inv,
                       mls_similarity_deformation, mls_similarity_deformation_inv,
                       mls_rigid_deformation, mls_rigid_deformation_inv)


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
    demo(mls_rigid_deformation, mls_rigid_deformation_inv, "Rigid")
    demo2(mls_rigid_deformation_inv)

    exit(0)
