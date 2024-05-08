import cv2
import numpy as np
#import os
#import dlib


class FaceAugmentation:
    def __init__(self, face_detector=None, landmarks_predictor=None, unt=1):
        self.detector = face_detector
        self.predictor = landmarks_predictor
        self.unt = unt

    def crop_face(self, image):
        try:
            rect = self.detector(image, upsample_num_times=self.unt)[0]
            startX = rect.left()
            startY = rect.top()
            endX = rect.right()
            endY = rect.bottom()
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(endX, image.shape[1])
            endY = min(endY, image.shape[0])
            return image[startY:endY, startX:endX]
        except:
            return None

    @staticmethod
    def rotate_image(image, angle):
        image_center = tuple(np.array(image.shape[1::-1]) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        return cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)

    def make_face_vertical(self, image):
        try:
            rectangle = self.detector(image, upsample_num_times=self.unt)[0]
            shape = self.predictor(image, rectangle)

            re_x = (shape.part(36).y + shape.part(39).y) // 2
            re_y = (shape.part(41).x + shape.part(38).x) // 2
            le_x = (shape.part(42).y + shape.part(45).y) // 2
            le_y = (shape.part(47).x + shape.part(44).x) // 2
            dx = re_x - le_x
            dy = re_y - le_y
            angle = np.degrees(np.arctan2(dx, dy)) - 180

            return self.rotate_image(image, angle)
        except:
            return None

    def crop_eyes_and_mouth(self, image, res_shape=(64, 32)):
        try:
            rectangle = self.detector(image, upsample_num_times=self.unt)[0]
            shape = self.predictor(image, rectangle)

            x1 = shape.part(17).x
            x2 = shape.part(26).x
            y1 = np.min([shape.part(19).y, shape.part(24).y])
            y2 = shape.part(29).y
            eyes = image[y1:y2, x1:x2]

            x1 = shape.part(17).x
            x2 = shape.part(26).x
            y1 = shape.part(33).y
            y2 = shape.part(57).y
            mouth = image[y1:y2, x1:x2]

            return cv2.resize(eyes, res_shape), cv2.resize(mouth, res_shape)
        except:
            return None, None