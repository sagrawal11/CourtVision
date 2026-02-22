"""
TrackNet ball detector that utilizes the 3-frame PyTorch model for maximum quality.
This replaces the old ensemble detector to provide robust, SOTA small object tracking.
"""

import sys
import os
from pathlib import Path
from typing import List, Optional, Tuple, Deque
from collections import deque
import numpy as np
import cv2
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pad=1, stride=1, bias=True):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=bias),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        return self.block(x)

class BallTrackerNet(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.out_channels = out_channels

        self.conv1 = ConvBlock(in_channels=9, out_channels=64)
        self.conv2 = ConvBlock(in_channels=64, out_channels=64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = ConvBlock(in_channels=64, out_channels=128)
        self.conv4 = ConvBlock(in_channels=128, out_channels=128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv5 = ConvBlock(in_channels=128, out_channels=256)
        self.conv6 = ConvBlock(in_channels=256, out_channels=256)
        self.conv7 = ConvBlock(in_channels=256, out_channels=256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv8 = ConvBlock(in_channels=256, out_channels=512)
        self.conv9 = ConvBlock(in_channels=512, out_channels=512)
        self.conv10 = ConvBlock(in_channels=512, out_channels=512)
        self.ups1 = nn.Upsample(scale_factor=2)
        self.conv11 = ConvBlock(in_channels=512, out_channels=256)
        self.conv12 = ConvBlock(in_channels=256, out_channels=256)
        self.conv13 = ConvBlock(in_channels=256, out_channels=256)
        self.ups2 = nn.Upsample(scale_factor=2)
        self.conv14 = ConvBlock(in_channels=256, out_channels=128)
        self.conv15 = ConvBlock(in_channels=128, out_channels=128)
        self.ups3 = nn.Upsample(scale_factor=2)
        self.conv16 = ConvBlock(in_channels=128, out_channels=64)
        self.conv17 = ConvBlock(in_channels=64, out_channels=64)
        self.conv18 = ConvBlock(in_channels=64, out_channels=self.out_channels)

        self.softmax = nn.Softmax(dim=1)
        self._init_weights()
                  
    def forward(self, x, testing=False): 
        batch_size = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)    
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.pool3(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.ups1(x)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.ups2(x)
        x = self.conv14(x)
        x = self.conv15(x)
        x = self.ups3(x)
        x = self.conv16(x)
        x = self.conv17(x)
        x = self.conv18(x)
        out = x.reshape(batch_size, self.out_channels, -1)
        if testing:
            out = self.softmax(out)
        return out                       
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.uniform_(module.weight, -0.05, 0.05)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)



class EnsembleBallDetector:
    """Combines consecutive frames through TrackNet for robust tennis ball detection.
       Retaining the class name 'EnsembleBallDetector' to avoid breaking upstream code.
    """
    
    def __init__(self, device: Optional[str] = None):
        """Initialize the TrackNet ball detector."""
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_input_width = 640
        self.model_input_height = 360
        
        # Keep track of the last up to 3 frames
        self.frame_history: Deque[np.ndarray] = deque(maxlen=3)
        self.video_width = None
        self.video_height = None
        
        # Load TrackNet model
        try:
            self._init_tracknet()
            self.model_loaded = True
            print("✓ TrackNet ball detector loaded successfully")
        except Exception as e:
            print(f"⚠ TrackNet detector failed to load: {e}")
            self.model_loaded = False
            
    def _init_tracknet(self):
        """Initialize TrackNet PyTorch detector and load pretrained weights."""
        model_path = PROJECT_ROOT / "models" / "ball" / "pretrained_ball_detection.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"TrackNet model weights not found at {model_path}")
        
        self.model = BallTrackerNet(out_channels=256)
        torch.serialization.add_safe_globals([np.dtype, np.generic])
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        
        if "model_state" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state"])
        elif "model" in checkpoint:
            self.model.load_state_dict(checkpoint["model"])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()

    def _combine_three_frames(self, frame1, frame2, frame3):
        """
        Resize, convert, and stack 3 frames for the TrackNet input.
        Returns a single numpy array of shape (9, height, width).
        """
        # Resize and type converting for each frame
        img1 = cv2.resize(frame1, (self.model_input_width, self.model_input_height))
        img1 = img1.astype(np.float32)

        img2 = cv2.resize(frame2, (self.model_input_width, self.model_input_height))
        img2 = img2.astype(np.float32)

        img3 = cv2.resize(frame3, (self.model_input_width, self.model_input_height))
        img3 = img3.astype(np.float32)

        # concatenate three imgs to  (height, width, rgb*3)
        imgs = np.concatenate((img1, img2, img3), axis=2)

        # TrackNet ordering is 'channels_first' (C, H, W)
        imgs = np.rollaxis(imgs, 2, 0)
        return np.array(imgs)


    def _get_center_ball(self, output: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Detect the center of the ball using Hough circle transform on the heatmap output.
        """
        output = output.reshape((self.model_input_height, self.model_input_width))
        output = output.astype(np.uint8)

        # heatmap is converted into a binary image by threshold method.
        ret, heatmap = cv2.threshold(output, 127, 255, cv2.THRESH_BINARY)

        # find the circle in image with 2<=radius<=7
        circles = cv2.HoughCircles(
            heatmap, cv2.HOUGH_GRADIENT, dp=1, minDist=1, 
            param1=50, param2=8, minRadius=2, maxRadius=7
        )
        
        if circles is not None and len(circles) > 0:
            # return coordinates of the most prominent circle
            x = int(circles[0][0][0])
            y = int(circles[0][0][1])
            return x, y
            
        return None

    def detect_ball(
        self,
        frame: np.ndarray,
        text_prompt: str = "tennis ball",
        threshold: float = 0.3
    ) -> Optional[Tuple[Tuple[int, int], float, np.ndarray]]:
        """
        Detect ball using the 3-frame TrackNet model.
        Returns best detection (center, confidence, mask) where mask may be None.
        
        Note: text_prompt and threshold args are kept for backwards compatibility
        with the visualizer's expectations.
        """
        if not self.model_loaded:
            return None
            
        # Initialize video dimensions on first frame
        if self.video_width is None:
            self.video_width = frame.shape[1]
            self.video_height = frame.shape[0]
            
        # Append current frame to history buffer
        self.frame_history.append(frame)
        
        # TrackNet requires 3 consecutive frames
        if len(self.frame_history) < 3:
            return None
            
        # Formulate input tensor
        # TrackNet convention: frame1=current, frame2=previous, frame3=oldest
        # Our deque: [oldest, previous, current]
        frames_combined = self._combine_three_frames(
            self.frame_history[2], # current
            self.frame_history[1], # previous
            self.frame_history[0]  # oldest
        )
        
        frames_tensor = (torch.from_numpy(frames_combined) / 255.0).float().to(self.device)
        frames_tensor = frames_tensor.unsqueeze(0) # add batch dimension
        
        with torch.no_grad():
            output = self.model(frames_tensor, testing=True)
            output = output.argmax(dim=1).detach().cpu().numpy()
            output *= 255
            
        # Calculate coordinates in 640x360 space
        center = self._get_center_ball(output)
        
        if center is not None:
            x_model, y_model = center
            # Scale back to original video dimensions
            x_orig = int(x_model * (self.video_width / self.model_input_width))
            y_orig = int(y_model * (self.video_height / self.model_input_height))
            
            # Confidence is synthesized since HoughCircles doesn't provide a direct prob percentage
            # Mask is None
            return ((x_orig, y_orig), 0.95, None)
            
        return None
