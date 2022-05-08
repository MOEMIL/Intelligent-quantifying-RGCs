import os

import numpy as np
import glob
import torch
import torch.nn as nn

from models.common import Conv
from utils.base import non_max_suppression
import cv2
import time


class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        y = torch.stack(y).mean(0)  # mean ensemble
        return y, None  # inference, train output


def getOneChannel(imgPath, idx):
    img = cv2.imdecode(np.fromfile(imgPath, dtype=np.uint8), -1)
    return img[:, :, idx]


class Detect:
    def __init__(self, weights_file):

        # 设置输入输出参数
        self.weights = weights_file
        self.input_size = 512
        self.num_classes = 1
        self.score_threshold = 0.1
        self.iou_threshold = 0.05
        
        self.model = self.attempt_load(map_location=torch.device('cpu'))

    def attempt_load(self, map_location=None):
        # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
        model = Ensemble()
        model.append(
            torch.load(self.weights, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

        # Compatibility updates
        for m in model.modules():
            if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
                m.inplace = True  # pytorch 1.7.0 compatibility
            elif type(m) is Conv:
                m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility
        return model[-1]  # return model
        

    def _py_cpu_nms(self, dets):
        print(dets.shape)

        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        areas = (y2 - y1 + 1) * (x2 - x1 + 1)
        scores = dets[:, 4]
        keep = []
        index = scores.argsort()[::-1]
        while index.size > 0:
            i = index[0]  # every time the first is the biggst, and add it directly
            keep.append(i)

            x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
            y11 = np.maximum(y1[i], y1[index[1:]])
            x22 = np.minimum(x2[i], x2[index[1:]])
            y22 = np.minimum(y2[i], y2[index[1:]])

            w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
            h = np.maximum(0, y22 - y11 + 1)  # the height of overlap

            overlaps = w * h
            ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)

            idx = np.where(ious <= self.iou_threshold)[0]
            index = index[idx + 1]  # because index start from 1
        return keep

    def _predict(self, image):
        org_image = np.copy(image)
        _, org_h, org_w = org_image.shape[:3]
        assert org_h == self.input_size and org_w == self.input_size

        image_data = image / 255.
        image_data = image_data[np.newaxis, ...]
        # image_data = float64(image_data)
        image_data = torch.from_numpy(image_data).float()

        y, _ = self.model(image_data)
        output = non_max_suppression(y, conf_thres=self.score_threshold, iou_thres=self.iou_threshold)

        return output[0].detach().numpy()

    def GetImg(self, image_path, image_name, exten):

        # Args:
        #    image_path: 图片路径
        #    exten: 图片后缀

        # Returns: 得到的图片 (channel, w, h)
        
        ext = "*." + exten
        filenames = sorted(glob.glob(os.path.join(image_path, ext)))
        filenames = [f for f in filenames if os.path.isfile(f)]
        filenames.sort()
        
        
        if len(filenames)<5:
            img1 = getOneChannel(os.path.join(image_path, image_name), 2)
            img2 = getOneChannel(os.path.join(image_path, image_name), 2)
            img3 = getOneChannel(os.path.join(image_path, image_name), 2)
            img4 = getOneChannel(os.path.join(image_path, image_name), 2)
            img5 = getOneChannel(os.path.join(image_path, image_name), 2)
        else:
            img1 = getOneChannel(filenames[0], 2)
            img2 = getOneChannel(filenames[1], 2)
            img3 = getOneChannel(filenames[2], 2)
            img4 = getOneChannel(filenames[3], 2)
            img5 = getOneChannel(filenames[4], 2)
        



        img = cv2.merge([img1, img2, img3, img4, img5])
        # img = cv2.merge([getOneChannel(image_path + '/1.' +exten,2)img1[:, :, 2], img2[:, :, 2], img3[:, :, 2], img4[:, :, 2], img5[:, :, 2]])
        img = img.transpose(2, 0, 1)
        return img

    def testBigImg(self, image_path, image_name, exten, csv_path):
        
        stride = 462
        intput_w = 512
        intput_h = 512
        
        original_image = self.GetImg(image_path, image_name, exten)

        BoxList = []
        for x in list(range(0, original_image.shape[2] - intput_w, stride)) + [original_image.shape[2] - intput_w]:
            for y in list(range(0, original_image.shape[1] - intput_h, stride)) + [original_image.shape[1] - intput_h]:
                img = original_image[:, y:y + intput_h, x:x + intput_w]
                # 
                bboxes = self._predict(img)
                

                for num in bboxes:
                    w = num[2] - num[0]
                    h = num[3] - num[1]
                    if (w < 5) or (h < 5):  # (w > 80) or (h > 80) or (w < 5) or (h < 5):
                        continue
                    if (w > 35) or (h > 35):  # (w > 80) or (h > 80) or (w < 5) or (h < 5):
                        continue
                    if num[4]<0.2:
                        continue

                    temp = np.array([num[0] + x, num[1] + y, num[2] + x, num[3] + y, num[4], num[5]])
                    BoxList.append(temp)

        # print('Begin Summary.... (This may take some time)')

        BoxNums = np.array(BoxList)
        keep = self._py_cpu_nms(BoxNums)
        BoxNums = BoxNums[keep]
        BoxConf = BoxNums[:, 4]
        BoxNums = BoxNums[:, 0:4]
        BoxNums = np.round(BoxNums)
        BoxNums[:, 2] = BoxNums[:, 2] - BoxNums[:, 0] + 1
        BoxNums[:, 3] = BoxNums[:, 3] - BoxNums[:, 1] + 1
        BoxNums = BoxNums.astype(np.int32)
        res = np.column_stack((BoxNums, BoxConf))
        np.savetxt(csv_path, res, delimiter=",")


def interface(image_path,csv_path):
    
    image_path, image_name=os.path.split(image_path[0])
    exten = image_name[-3:] #tif  jpg  png

    yolov5.testBigImg(image_path, image_name, exten, csv_path[0])
    
    return 1


weights = "C:/pmse_plus.pt"
yolov5 = Detect(weights)

testimg = np.zeros((5, 512, 512), dtype=np.uint8)
yolov5._predict(testimg)
print("it is ok")
