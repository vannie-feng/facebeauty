
from PIL import Image, ImageDraw
import face_recognition
import numpy as np
import json
import sys
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt


def process(): 
 #   # Load the jpg file into a numpy array
 #   imagep = face_recognition.load_image_file("slp.jpg")
 #   imageq = face_recognition.load_image_file("slq.jpg")
    
    if len(sys.argv)<3:
        imagep = face_recognition.load_image_file("slp.jpg")
        imageq = face_recognition.load_image_file("slq.jpg")
    else:
        imagep=face_recognition.load_image_file(str(sys.argv[1]))
        imageq=face_recognition.load_image_file(str(sys.argv[2]))
    
    # Find all facial features in all the faces in the image
    face_landmarks_listp = face_recognition.face_landmarks(imagep)
    face_landmarks_listq = face_recognition.face_landmarks(imageq)
    
    
    print("I found {} face(s) in this photograph.".format(len(face_landmarks_listp)))
    
    # Create a PIL imagedraw object so we can draw on the picture
    pil_imagep = Image.fromarray(imagep)
    pil_imageq = Image.fromarray(imageq)
    dp = ImageDraw.Draw(pil_imagep)
    dq = ImageDraw.Draw(pil_imageq)
    featurename=face_landmarks_listp[0].keys()
    cnt=1
    for face_landmarks in face_landmarks_listp:
        
        Alipoint=[]
        # Print the location of each facial feature in this image
        for facial_feature in face_landmarks.keys():
    #        print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
            if facial_feature=='chin':
                print 'here'
                print type(face_landmarks['chin'])
    
                array1=np.array(face_landmarks['chin'])
                np.save('./pchin.npy',array1)
        Alipoint.append(face_landmarks['left_eye'][2])
        Alipoint.append(face_landmarks['right_eye'][2])
        Alipoint.append(face_landmarks['bottom_lip'][6])
        # Let's trace out each facial feature in the image with a line!
        for facial_feature in face_landmarks.keys():
            dp.line(face_landmarks[facial_feature], width=5)
        arrayAli=np.array(Alipoint)
        np.save('./Alipointp'+str(cnt)+'.npy',arrayAli)
        cnt+=1
    cnt=1
    
    print "have %d faces in picq"%len(face_landmarks_listq)
    for face_landmarks in face_landmarks_listq:
        Alipoint=[]
        # Print the location of each facial feature in this image
        for facial_feature in face_landmarks.keys():
            print("The {} in this face has the following points: {}".format(facial_feature, face_landmarks[facial_feature]))
            if facial_feature=='chin':
                array1=np.array(face_landmarks['chin'])
                print array1
                np.save('./qchin.npy',array1)
        Alipoint.append(face_landmarks['left_eye'][2])
        Alipoint.append(face_landmarks['right_eye'][2])
        Alipoint.append(face_landmarks['bottom_lip'][6])
        arrayAli=np.array(Alipoint)
        np.save('./Alipointq'+str(cnt)+'.npy',arrayAli)
        cnt+=1
        # Let's trace out each facial feature in the image with a line!
        for facial_feature in face_landmarks.keys():
           # dq.line(face_landmarks[facial_feature], width=5)
            for x,y in face_landmarks[facial_feature]:
               # print ("keypoint",x,y)
                dq.text((x,y),u'p')
    # Show the picture
    pil_imageq.save("./kpointq.jpg")
    # Show the picture
    pil_imagep.save("./kpointp.jpg")
    #pil_image.show()
if __name__=='__main__':
   # set_arg()
    process()

