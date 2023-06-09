from tensorflow.keras.models import load_model
import cv2
import os
import numpy as np
import tensorflow as tf
import logging
# import pretrain_models.retinaface as retinaface
# import pretrain_models.retinaface.modules.utils as utils
from models.localize.retinaface.modules.models import RetinaFaceModel
from models.localize.retinaface.modules.utils import (
    set_memory_growth, load_yaml, draw_bbox_landm, pad_input_image, recover_pad_output)
from tensorflow.keras.preprocessing.image import img_to_array    

logger = logging.getLogger(__name__)


class ModelHandler:
    """ This class supports basic functionality of a model. 
    Functions of the class include: 
        __init__(self) -> None
        async __get_retinaface(self) -> RetinaFaceModel, Dict
        async run_detection(self) -> frame: DON'T draw on it, result_list:[[box[i] of frames, label[i]] 
    """

    def __init__(self):
        logger.info('Loading models')
        self.face_localize_model, self.cfg = self.__get_retinaface(
            'retinaface_res50')
        self.mask_classifier = load_model('./models/classify/MyResNet50')

    def __get_retinaface(self, model_type, iou_thres=0.4, score_thres=0.5):
        cfg = load_yaml(
            './models/localize/retinaface/configs/'+model_type+'.yaml')
        model = RetinaFaceModel(cfg, training=False,
                                iou_th=iou_thres, score_th=score_thres)
        # load checkpoint
        checkpoint_dir = './models/localize/retinaface/checkpoint/' + model_type
        checkpoint = tf.train.Checkpoint(model=model)
        checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))
        logger.info("Loading checkpoint of ", model_type)
        return model, cfg

    def __draw_box(self, image, box, color, label_str):
        # print(box)

        cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 1)
        cv2.putText(image, label_str,
                    (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        return image

    def __draw_target(self, image, box, label):  # visualize the confidence here
        masked_color = (0, 255, 0)
        unmasked_color = (255, 0, 0)
        wrongmasked_color = (0, 0, 255)

        if (label == 1):
            new_img = self.__draw_box(image, box, masked_color, 'mask')

        elif (label == 0):
            new_img = self.__draw_box(image, box, wrongmasked_color, 'no mask')

        else:
            new_img = self.__draw_box(image, box, unmasked_color, 'incorrect')

    def run_detection(self, video_frame):
        if not self.face_localize_model or not self.mask_classifier:
            raise Exception('Models not found')

        img_height_raw, img_width_raw, _ = video_frame.shape
        img = np.float32(video_frame.copy())
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # pad input image to avoid unmatched shape problem
        img, pad_params = pad_input_image(
            img, max_steps=max(self.cfg['steps']))

        # run model
        outputs = self.face_localize_model(img[np.newaxis, ...]).numpy()

        # recover padding effect
        outputs = recover_pad_output(outputs, pad_params)
        result = []
        temp = {}

        # classify and draw and save results
        for idx in range(len(outputs)):
            detected_face = outputs[idx]
            x_min = int(detected_face[0] * img_width_raw)
            y_min = int(detected_face[1] * img_height_raw)
            x_max = int(detected_face[2] * img_width_raw)
            y_max = int(detected_face[3] * img_height_raw)
            box = (x_min, y_min, x_max, y_max)
            face = video_frame[y_min:y_max, x_min:x_max]
            try:
                face = cv2.resize(face, (256, 256))
            except:
                break
            face = img_to_array(face)
            face = np.expand_dims(face, axis=0)
            scores = self.mask_classifier.predict(face)[0].tolist()

            # print(result)
            score = max(scores)
            label=scores.index(max(scores))
            # 0: without, 1: with mask, 2: incorrect
            self.__draw_target(video_frame, box, label)
            label = scores.index(max(scores))
            result.append({'box': box, 'label': label})
        temp = self.count_faces(result)
        return video_frame, temp
    
    def count_faces(self, result):
        total = len(result)
        mask=0
        no_mask=0 
        incorrect=0
        for e in result:
            if e['label']==0:
                no_mask+=1
            elif (e['label']==1):
                mask+=1
            else: 
                incorrect+=1
        return {'total': total,
                'mask': mask,
                'no_mask': no_mask,
                'incorrect': incorrect}
