import cv2
import numpy as np

def getTxtHeight(img, heightPrior=100):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobel16s = cv2.Sobel(gray,cv2.CV_16S, 1, 0, 3)
    sobel8u = cv2.convertScaleAbs(sobel16s)
    avgGrayValPerRow = np.squeeze(np.mean(sobel8u, axis=1)) # 对于m行图片数据， gray_val即对m行中的每一行求平均值


    maxVal = np.max(avgGrayValPerRow[:-heightPrior]) # 后100行之前的最大行灰度值

    maxDiff = -1
    rowIdx = -1

    for i, curVal in enumerate(avgGrayValPerRow[-heightPrior:]): # 对后100行数据求两两之间的最大差值。存储差值最大的行索引
        if curVal > maxVal:
            maxVal = curVal
        
        diff = curVal - maxVal
        if diff > maxDiff:
            maxDiff = diff
            rowIdx = i
     
    return (heightPrior - rowIdx)
        

if __name__ == '__main__':
    ori = cv2.imread('3300_TB9C3614AV_TBAOLDC0_14_-65.550_-1133.180__S_20191225_010138.JPG')
    txtHeight = getTxtHeight(ori)
    crop = ori[:-txtHeight,:]
    cv2.imwrite("./crop.jpg", crop)
    # cv2.namedWindow("ori", cv2.WINDOW_NORMAL)
    # cv2.namedWindow("crop", cv2.WINDOW_NORMAL)
    # cv2.imshow("ori", ori)
    # cv2.imshow("crop", crop)
    # cv2.waitKey(0)
