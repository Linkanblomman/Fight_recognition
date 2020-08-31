import torch
import numpy as np
import joblib
import cv2
import time
import math
 
from PIL import Image

import resnet
from spatial_transforms import (Compose, Normalize, Resize, CenterCrop,
                                CornerCrop, MultiScaleCornerCrop,
                                RandomResizedCrop, RandomHorizontalFlip,
                                ToTensor, ScaleValue, ColorJitter,
                                PickFirstChannels)

# load the trained model and label binarizer from disk
print('Loading model and label binarizer...')
lb = joblib.load("./outputs/lb.pkl")

model = resnet.generate_model(model_depth=50, n_classes=700)

fc = torch.nn.Linear(2048, 2) # requires_grad=True by deafault
model.fc = fc
print('Model Loaded...')

model.load_state_dict(torch.load("./outputs/fight_reco_3DCNNmodel.pth"))
print('Loaded model state_dict...')

device = torch.device('cuda:0')
model.to(device)

value_scale = 1
mean = [0.4345, 0.4051, 0.3775]
std = [0.2768, 0.2713, 0.2737]

sample_size = 112 # resolution of frame

spatial_transform =  Compose([Resize(sample_size),
                                        CenterCrop(sample_size),
                                        ToTensor(),
                                        ScaleValue(value_scale),
                                        Normalize(mean, std)])

VIDEO_PATH = "./input/test_data/fi038.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)

if (cap.isOpened() == False):
    print('Error while trying to read video. Plese check again...')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

collected_frames = []

# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read() # capturing the frame 
    if ret == True:
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        collected_frames.append(pil_image)     
    else:
        model.eval()
        with torch.no_grad():
          duration = len(collected_frames)
          frames = 16
          steps = math.floor(duration/frames)
          start_frame = 0
          stop_frame = steps * frames
          
          frame_id_list = range(start_frame, stop_frame, steps)
          collected_frames = [collected_frames[id] for id in frame_id_list]

          video_snippet = [spatial_transform(frame) for frame in collected_frames]
          video_snippet = torch.stack(video_snippet, 0).permute(1,0,2,3) # [Channel, Depth, Height, Width]

          batch = video_snippet.unsqueeze(0).cuda() # [Batch, Channel, Depth, Height, Width]
          #print("Batch shape:", batch.shape)
          
          outputs = model(batch)
          _, preds = torch.max(outputs.data, 1)
          
          prediction = lb.classes_[preds]
          print("Model predict:",prediction)
        break  

# release VideoCapture()
cap.release()

"""If no video output. Run last 2 cells again"""

cap = cv2.VideoCapture(VIDEO_PATH)

if (cap.isOpened() == False):
    print('Error while trying to read video. Plese check again...')

# get the frame width and height
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

image_text_pos_X = int(frame_width/2.5)
image_text_pos_Y = int(frame_height/10)

# define codec and create VideoWriter object (specify the format for saving the video)
out = cv2.VideoWriter(str("./outputs/model_prediction_on_video/" + prediction + ".mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 30, (frame_width,frame_height))

while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read() # capturing the frame 
    if ret == True:
        cv2.rectangle(frame, (int(image_text_pos_X - 50), int(image_text_pos_Y - 30)) , (int(image_text_pos_X + 160), int(image_text_pos_Y + 10)), (96,96,96), -1)
        #image, text, pos, font, fontSize, fontColor, fontThickness
        if(prediction == "fight"):
          cv2.putText(frame, lb.classes_[preds], (int(image_text_pos_X + 20), image_text_pos_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2) 
        else:
          cv2.putText(frame, lb.classes_[preds], (int(image_text_pos_X), image_text_pos_Y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        
        cv2.imshow('image', frame)
        out.write(frame)

        # press `q` to exit
        if cv2.waitKey(27) & 0xFF == ord('q'):
            break
    else: 
        break

# release VideoCapture()
cap.release()

# close all frames and video windows
cv2.destroyAllWindows()

print("Model prediction COMPLETE")
