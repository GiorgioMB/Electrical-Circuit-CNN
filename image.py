import torch
import torchvision.models as models
import ssl
from PIL import Image, ImageDraw
from torchvision.transforms import functional as F
import json
import cv2
import numpy as np
ssl._create_default_https_context = ssl._create_unverified_context

dictionary = {'diode': 1, 'transistor.bjt': 2, 'nor': 3, 'operational_amplifier': 4, 'speaker': 5, 'integrated_circuit': 6, 'voltage.battery': 7, 'voltage.ac': 8, 'resistor': 9, 'optical': 10, 'crystal': 11, 'transistor.photo': 12, 'capacitor.polarized': 13, 'relay': 14, 'crossover': 15, 'switch': 16, 'microphone': 17, 'probe': 18, 'terminal': 19, 'nand': 20, 'mechanical': 21, 'xor': 22, 'vss': 23, 'probe.current': 24, 'diode.zener': 25, 'transistor.fet': 26, 'and': 27, 'text': 28, 'capacitor.adjustable': 29, 'triac': 30, 'capacitor.unpolarized': 31, 'resistor.photo': 32, 'gnd': 33, 'diode.light_emitting': 34, 'voltage.dc': 35, 'integrated_circuit.ne555': 36, 'operational_amplifier.schmitt_trigger': 37, 'magnetic': 38, 'resistor.adjustable': 39, 'not': 40, 'integrated_circuit.voltage_regulator': 41, 'lamp': 42, 'varistor': 43, 'socket': 44, 'unknown': 45, 'thyristor': 46, 'junction': 47, 'optocoupler': 48, 'probe.voltage': 49, 'or': 50, 'antenna': 51, 'transformer': 52, 'inductor': 53, 'diode.thyrector': 54, 'fuse': 55, 'inductor.ferrite': 56, 'diac': 57, 'motor': 58}
classes = {str(v): k for k, v in dictionary.items()}

class ModelOutput:
    def __init__(self, model_path, num_classes : int = 59, resolution : tuple[int, int] = (1024, 1024)) -> None:
        # PATH = './faster_rcnn_model_20.pth'
        self.h = resolution[0]
        self.w = resolution[1]
        self.resolution = resolution
        
        self.model = models.detection.fasterrcnn_resnet50_fpn_v2(pretrained=True)

        in_features = self.model.roi_heads.box_predictor.cls_score.in_features

        self.model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)

        state_dict = torch.load(model_path, map_location='cpu' if not torch.cuda.is_available() else None)
        new_state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
        self.model.load_state_dict(new_state_dict)
        self.model.eval()

    def get_boxes_image(self, in_path: str, out_path: str) -> None:
        image = Image.open(in_path)
        image_pil = F.resize(image, self.resolution)
        image = F.to_tensor(image_pil)
        
        pred = self.model([image]) 
        pred = pred[0]
        box_ids = pred['labels']
        scores = pred['scores']
        bboxes = pred['boxes']
        image_np = np.array(image_pil)
        
        for box, label, score in zip(bboxes, box_ids, scores):
            if score.item() > 0.5:
                x1, y1, x2, y2 = box.tolist()  # Convert tensor to Python list
                p1 = np.array([x1,y1]).astype(np.int16) 
                p2 = np.array([x2,y2]).astype(np.int16) 
                cv2.rectangle(image_np, p1, p2, (36,255,12), thickness=2)
                cv2.putText(image_np, classes[str(label.item())], p1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
        cv2.imwrite(out_path, image_np)


if __name__ == '__main__':
    mod = ModelOutput('faster_rcnn_model.pth')
    mod.get_boxes_image('TEST.jpg', 'out.jpg')
