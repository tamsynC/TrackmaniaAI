# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# # class SimpleCNN(nn.Module):
# #     """
# #     CNN model for behavioral cloning on driving data.
# #     Input: RGB image (3 x 120 x 160)
# #     Output: 4 logits corresponding to key presses [a, d, w, s]
# #     """
# #     class SimpleCNN(nn.Module):
# #         """
# #         CNN model for behavioral cloning on driving data.
# #         Input: RGB image (3 x 120 x 160)
# #         Output: 4 logits corresponding to key presses [a, d, w, s]
# #         """
# #         def __init__(self):
# #             super().__init__()
# #             self.features = nn.Sequential(
# #                 nn.Conv2d(3, 16, 3, stride=2), nn.ReLU(),
# #                 nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
# #                 nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
# #             )
# #             self.classifier = nn.Sequential(
# #                 nn.Flatten(),
# #                 nn.Linear(17024, 128, bias=True),
# #                 nn.ReLU(),
# #                 nn.Linear(128, 4, bias=True),
# #                 nn.Sigmoid()
# #             )

# #         def forward(self, x):
# #             x = self.features(x)
# #             x = self.classifier(x)
# #             return x

# class SimpleCNN(nn.Module):
#     def __init__(self):
#         super(SimpleCNN, self).__init__()
#         print("Initializing SimpleCNN model")
#         self.features = nn.Sequential(
#             nn.Conv2d(3, 16, 3, stride=2), nn.ReLU(),
#             nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
#             nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
#         )
#         self.classifier = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(17024, 128, bias=True),
#             nn.ReLU(),
#             nn.Linear(128, 4, bias=True),
#             nn.Sigmoid()
#         )
#         print("Model initialized")

import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        print("Initializing SimpleCNN model")
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2), nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2), nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(17024, 128, bias=True),
            nn.ReLU(),
            nn.Linear(128, 4, bias=True),
            nn.Sigmoid()
        )
        print("Model initialized")

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
