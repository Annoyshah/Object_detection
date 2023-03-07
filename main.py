import cv2
import numpy as np
from gui_buttons import Buttons 

button = Buttons()
button.add_button("person",20,20)
button.add_button("cell phone" , 20 , 90 )
button.add_button("mouse" , 20 , 160)
button.add_button("remote",20 , 230)
button.add_button("keyboard",20,300)
button.add_button("clock",20,370)
button.add_button("toothbrush",20,440)
button.add_button("cup",20,510)
button.add_button("spoon",20,580)
button.add_button("bottle",20,650)
#initalising network  once we initalize the network we need not define the model , 
# we have used here a coco dataset , so a coco dataset is 
net = cv2.dnn.readNet("dnn_model/yolov4-tiny.weights","dnn_model/yolov4-tiny.cfg") #on readnet we need two files , configuration of the model and then the weights file of the model and the file that is trained on 1000 of images based on objects
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320), scale=1/255)# bigger image , better resolution , slower processing
# when we process in deep learning even a 4k frame or even a very big image , the deep learning model doesnot process the bigger image but its going 
# to shrink the image , doesnt matter the shape of the image even if its a rectangle 69 resolution , its going to resize that into 
#square , usyually very small square like in this case 320 * 320 , bigger size better precision but slower speed 
# 320 is good detection as well as good speed , sweet spot 
# scale = 1/255 because on opencv pixels go from 0 t0 255 but in neural networks they go from 0 to 1

#Load class lists
classes = []
with open("dnn_model/classes.txt","r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)
print(classes)


cap = cv2.VideoCapture(0) #0 will take the first webcam , 1 will take the second web cam 
cap.set(cv2.CAP_PROP_FRAME_WIDTH ,1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
#FULL HD 1920 * 1080

def click_button(event , x , y , flags , params):
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        button.button_click(x,y)

            
            




#Create window
cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame",click_button)

while True: #by keeping this in a loop we get a new frame everytime when we press a key as cv2.waitkey() is 0 , 0 freezes the frame and resumes when we press a key , 1 will take 1 mllisecond to resume between thw frames
    ret, frame = cap.read() #ret is to check if the frame is null or not 
    
    # set active button list
    active_buttons = button.active_buttons_list()
    print("active_buttons",active_buttons)
    # Object detection
    (class_ids , scores , bboxes) = model.detect(frame) #bboxes ( bounding boxes is for that specific object where the object is located so a rectangle surrounding the object , we get mostly two coordinates )
                                                                                            #zip is the function to extract at the same time in the loop arrays 
    for class_id,score,bbox in zip(class_ids,scores,bboxes):
        (x,y,w,h) = bbox
        class_name = classes[class_id]
        
        if class_name in active_buttons:
            cv2.putText(frame , str(class_name), (x,y-5),cv2.FONT_HERSHEY_PLAIN,1,(200,0,50),2)
            cv2.rectangle(frame,(x,y),(x+w,y+h),(200,0,50),3)      
                                                             
                                                                                       
    # score is how confident is the object about detection
    # class_id is the id of the class , we have 80 classes in coco dataset , so we have 80 classes in the class_id
    # cv2.rectangle(frame , (20 ,20) , (150 , 70) , (0,0,200), -1)
    # polygon = np.array([[(20,20),(220,20),(220,70),(20,70)]])
    # cv2.fillPoly(frame , polygon , (0,0,200))
    # cv2.putText(frame , "PERSON" , (30,60) , cv2.FONT_HERSHEY_PLAIN , 1 , (255,255,255) , 3)
    button.display_buttons(frame)
    cv2.imshow("Frame", frame) #its open webcam window but closes it very fast 
    key = cv2.waitKey(1) #doing this the frame is stready.
    if key == 27: #27 is the ascii value of escape key
        break
    




