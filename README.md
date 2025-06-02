# TrackmaniaAI

## Resources
The Game:
https://store.ubisoft.com/us/trackmania/5e8b58345cdf9a12c868c878.html?lang=en_US

## Group information
Group 5: Trackmania

## Group composition:
Jet Webb: 24502825

Tamsyn Crangle: 24439287

Nicolas Yao: 24563633

Connor Williams: 24459594

## Install info
Dependencies:
Use pip install -r thingsToInstall.txt to install dependencies.
Trackmania 2020 with the following settings: Additionally, if training set max FPS ingame to 15 and resolution to 640*480.
![image](https://github.com/user-attachments/assets/c90e8cd6-a327-40e7-956b-91a1ab6d9471)

Adjust the give up key bind to be Backspace.
Download the track BasicAIMapV5.Gbx map and put it in the folder Documents\Trackmania\Maps\Downloaded. This will allow you to open the track under local play a track, downloaded.

## Run commands

Training:
1.	Open the Trackmania map.
2.  Run: python JetsModelV12-12.py
3.	Click on the Trackmania window, the model will then train.


![image](https://github.com/user-attachments/assets/32af277e-dcd4-4146-9be8-85c9705e3bf6)
Running the Code:
1.	Open the Trackmania map.
2. If required, change model paths on lines 17 & 19:

```python
MODEL_PATH = 'unet_model.pth'  # Segmentation model path
DQN_MODEL_PATH = 'trackmania_dqn_interrupted.pth'  # Your trained DQN model

4.	Run: python ModelRunJetV6.py
5.	Click on the Trackmania window, the model will then run.
