import numpy as np
import pandas as pd
import cv2
import imutils

class load_image:
    def __init__(self,fold='data',filename = 'image003.png',area_size = 0.25,plot_image = True,threshold1=30, threshold2=100):
        self.img = cv2.imread(filename)
        grayImage = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        

        edges = cv2.Canny(grayImage, threshold1=threshold1, threshold2=threshold2)

        contours, hierarchy = cv2.findContours(edges,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        areas = []
        areas_ind = []
        countours_filter = []
        #filter for countours that isn't rocks'
        for k in range(0,len(contours)):
            if cv2.contourArea(contours[k])> area_size:
                areas.append(cv2.contourArea(contours[k]))
                areas_ind.append(k)
                countours_filter.append(contours[k])

        print("Significant countours = " + str(len(countours_filter)))

        pos = 1
        for c in countours_filter:
            if cv2.contourArea(c)> area_size:
                # compute the center of the contour
                M = cv2.moments(c)
                try:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                except:
                    continue
                # draw the contour and put ids rotules
                cv2.drawContours(self.img, c, -1, (0, 255, 0), 2)
                cv2.putText(self.img, "#id_"+str(pos), (cX - 40, cY - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 255), 1)
                pos = pos+1
        cv2.imwrite('./Output/counter_'+filename,self.img)
        if plot_image:
            
            cv2.imshow('Counter Image', self.img)
            
            cv2.waitKey(0)
            cv2.destroyAllWindows()

if __name__ == '__main__':
    #Reading of images 003 and 002
    img_1 = load_image(fold='./data/',filename = 'image003.png',area_size = 2.5)
    img_2 = load_image(fold='./data/',filename = 'image002.jpg',area_size = 2.5,threshold1=60, threshold2=200)
    
    

    