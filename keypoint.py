import torchvision
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import transforms as transforms

def draw_keypoints_and_boxes(outputs, image):
    # the `outputs` is list which in-turn contains the dictionary
    for i in range(len(outputs[0]['keypoints'])):
        # get the detected keypoints
        keypoints = outputs[0]['keypoints'][i].cpu().detach().numpy()
        # get the detected bounding boxes
        boxes = outputs[0]['boxes'][i].cpu().detach().numpy()
        # proceed to draw the lines and bounding boxes
        if outputs[0]['scores'][i] > 0.9: # proceed if confidence is above 0.9
            keypoints = keypoints[:, :].reshape(-1, 3)
            for p in range(keypoints.shape[0]):
                # draw the keypoints
                cv2.circle(image, (int(keypoints[p, 0]), int(keypoints[p, 1])),
                            3, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            # draw the bounding boxes around the objects
            cv2.rectangle(image, (int(boxes[0]), int(boxes[1])), (int(boxes[2]), int(boxes[3])),
                          color=(0, 255, 0),
                          thickness=2)
        else:
            continue
    return image


def get_model():
    # initialize the model
    model = torchvision.models.detection.keypointrcnn_resnet50_fpn(pretrained=True,num_keypoints=17,min_size=800)
    return model

transform = transforms.Compose([
    transforms.ToTensor()
])
# set the computation device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load the modle on to the computation device and set to eval mode
model = get_model().to(device).eval()
cap = cv2.VideoCapture(0)
logo = cv2.imread('image.png')
size = 100
logo = cv2.resize(logo, (size, size))
img2gray = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 1, 255, cv2.THRESH_BINARY)
# get the video frames' width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
while (cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:

        pil_image = Image.fromarray(frame).convert('RGB')
        orig_frame = frame
        # transform the image
        image = transform(pil_image)
        # add a batch dimension
        image = image.unsqueeze(0).to(device)
        # get the detections, forward pass the frame through the model
        with torch.no_grad():
            outputs = model(image)
        output_image = draw_keypoints_and_boxes(outputs, orig_frame)
        cv2.imshow('Pose detection frame', output_image)
        cv2.waitKey(1)
    else:
        break
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()