# import time
# import keyboard
# import random as rand

# outputs = ['a', 'd', 'w', 's']
# outputweights = [0.5, 0.5, 0.95, 0.1] #threshold values

# def RandomOutputs():

#     randOutputs = []

#     for i in range(len(outputs)):
#         if rand.random() <= outputweights[i]:
#             keyboard.press(outputs[i])
#             print(f"Pressed: {outputs[i]}")
#         else:
#             keyboard.release(outputs[i])
#             print(f"Release: {outputs[i]}")

#     #If in list -> keep pressing else release

# if __name__ == "__main__":
#     # Skeyboard.press_and_release('delete')
#     while True:
#         RandomOutputs()
#         time.sleep(1/60)

import keyboard

def main():
    print("Press keys (Ctrl+C to exit):")
    while True:
        try:
            key_event = keyboard.read_event()
            if key_event.event_type == keyboard.KEY_DOWN:
                print(f"Pressed: {key_event.name}")
        except KeyboardInterrupt:
            print("\nExiting...")
            break

if __name__ == "__main__":
    main()
