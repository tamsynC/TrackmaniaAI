import time
import keyboard
import random as rand

outputs = ['left', 'right', 'up', 'down']
outputweights = [0.5, 0.5, 0.95, 0.1] #threshold values

def RandomOutputs():

    randOutputs = []

    for i in range(len(outputs)):
        if rand.random() <= outputweights[i]:
            keyboard.press(outputs[i])
        else:
            keyboard.release(outputs[i])

    #If in list -> keep pressing else release

if __name__ == "__main__":
    keyboard.press_and_release('delete')
    while True:
        RandomOutputs()
        time.sleep(1/60)