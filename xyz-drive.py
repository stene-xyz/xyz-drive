global dataset, neuralNet, sct, gamepad, screen, mode

# Used for screen capture
import numpy as np
import cv2
from mss import mss
from PIL import Image

# Used for input/output
import vgamepad as vg
import pygame

from datetime import datetime
import os, sys
from ML import ML
dataset = ML.Dataset("data.json")
neuralNet = ML.NeuralNet(dataset)

# Init screen capture
sct = mss()

# Init input/output
gamepad = vg.VX360Gamepad()
pygame.init()
screen = pygame.display.set_mode((400, 400))

# 0 - do nothing
# 1 - capture training data
# 2 - run model
mode = 0

def run():
    global dataset, neuralNet, sct, gamepad, screen, mode
    while True:
        for event in pygame.event.get():
            if(event.type == pygame.QUIT):
                cv2.destroyAllWindows()
                pygame.quit()
                return
        
        screen.fill("white")

        keys = pygame.key.get_pressed()
        throttle = 0
        brake = 0
        if keys[pygame.K_w]:
            throttle = 255
            gamepad.right_trigger(value=255)

        if keys[pygame.K_s]:
            brake = 255

        steer = 0
        if keys[pygame.K_a]:
            steer -= 23768
        if keys[pygame.K_d]:
            steer += 23768

        if(keys[pygame.K_z]):
            if(mode == 1):
                print("Saving dataset...")
                dataset.save()
                print("Dataset saved.")
            mode = 0
        elif(keys[pygame.K_x]):
            mode = 1
        elif(keys[pygame.K_c]):
            mode = 2

        sct_img = sct.grab({'top': 0, 'left': 0, 'width': 1920, 'height': 1080})
        img = np.array(sct_img)
        img = cv2.resize(img, (960, 540))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow('screen', img)

        if(mode == 1): # train mode
            imageFilename = datetime.now().strftime("%b-%d-%j-%I-%M-%p-%S-%f.png")
            cv2.imwrite("data-img/" + imageFilename, img)
            neuralNetData = [False, False, False]
            neuralNetData[0] = throttle
            neuralNetData[1] = brake
            neuralNetData[2] = steer
            neuralNet.dataset.data[imageFilename] = neuralNetData

            pygame.display.flip()
        if(mode == 2):
            netImg = cv2.merge([img, img, img])
            netImg = np.reshape(netImg, (1, 1, 540, 960, 3))
            neuralNetData = neuralNet.predict(netImg)
            #print(neuralNetData)
            throttle = int(neuralNetData[0][0].item()) * 100
            brake = int(neuralNetData[0][1].item()) * 100
            steer = int(neuralNetData[0][2].item()) * 100
        
        gamepad.left_trigger(value=brake)
        gamepad.right_trigger(value=throttle)
        gamepad.left_joystick(x_value=steer, y_value=0)
        gamepad.update()

if("--train" in sys.argv):
    neuralNet.train()
    neuralNet.model.save()
    neuralNet.dataset.save()
else:
    run()