import time
import cv2
import numpy as np
import torch

from PIL import Image
from transformers import (
    VideoMAEImageProcessor,
    VideoMAEForVideoClassification,
)


MODEL_NAME = "Vansh180/VideoMae-ffc23-deepfake-detector"
NUM_FRAMES = 16


class DeepfakeDetector:

    def __init__(self):

        print("Loading VideoMAE deepfake detection model...")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        print(f"Using device: {self.device}")

        self.processor = VideoMAEImageProcessor.from_pretrained(
            MODEL_NAME
        )

        self.model = VideoMAEForVideoClassification.from_pretrained(
            MODEL_NAME
        )

        self.model.to(self.device)
        self.model.eval()

        print("VideoMAE model loaded successfully.")

    def extract_frames(self, video_path):

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError("Could not open the video file.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        duration = (
            total_frames / fps
            if fps and fps > 0
            else 0
        )

        if total_frames <= 0:
            cap.release()
            raise ValueError("Video contains no readable frames.")

        # Select 16 frames uniformly across the entire video
        indices = np.linspace(
            0,
            total_frames - 1,
            NUM_FRAMES
        ).astype(int)

        frames = []
        display_frames = []

        for index in indices:

            cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))

            success, frame = cap.read()

            if success:

                # OpenCV BGR -> RGB
                frame_rgb = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB
                )

                pil_frame = Image.fromarray(frame_rgb)

                frames.append(pil_frame)

                # Keep original RGB frame for Streamlit display
                display_frames.append(frame_rgb)

        cap.release()

        if len(frames) == 0:
            raise ValueError("Could not extract frames from video.")

        return {
            "frames": frames,
            "display_frames": display_frames,
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "duration": duration,
        }

    def analyze(self, video_path):

        start_time = time.time()

        video_data = self.extract_frames(video_path)

        frames = video_data["frames"]

        # Prepare frames for VideoMAE
        inputs = self.processor(
            frames,
            return_tensors="pt"
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # Model inference
        with torch.no_grad():

            outputs = self.model(**inputs)

            probabilities = torch.softmax(
                outputs.logits,
                dim=1
            )[0]

        # Model configuration:
        # 0 = real
        # 1 = fake
        real_probability = float(probabilities[0].item())
        fake_probability = float(probabilities[1].item())

        if fake_probability >= real_probability:
            prediction = "DEEPFAKE"
            confidence = fake_probability
        else:
            prediction = "REAL"
            confidence = real_probability

        processing_time = time.time() - start_time

        # Show 8 representative frames in the interface
        display_frames = video_data["display_frames"]

        if len(display_frames) > 8:

            display_indices = np.linspace(
                0,
                len(display_frames) - 1,
                8
            ).astype(int)

            display_frames = [
                display_frames[i]
                for i in display_indices
            ]

        return {
            "prediction": prediction,

            "real_probability": real_probability,
            "fake_probability": fake_probability,

            "confidence": confidence,

            "processing_time": processing_time,

            "frames_analyzed": len(frames),
            "frames_displayed": len(display_frames),

            "display_frames": display_frames,

            "total_frames": video_data["total_frames"],
            "fps": video_data["fps"],
            "width": video_data["width"],
            "height": video_data["height"],
            "duration": video_data["duration"],
        }