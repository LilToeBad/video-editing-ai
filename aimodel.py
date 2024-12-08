import cv2
from ultralytics import YOLO
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import numpy as np
import os
import json

print("All packages imported successfully!")

def extract_audio(video_path):
    """Analyze audio for sound patterns and volume"""
    try:
        # Load audio from video
        audio = AudioSegment.from_file(video_path)

        # Detect louder moments
        nonsilent = detect_nonsilent(audio, min_silence_len=500, silence_thresh=-30)
        num_segments = len(nonsilent)

        # Rules
        if num_segments > 10: # More than 10 loud segments detected
            return "action"
        elif "laughter" in audio.raw_data.decode(errors="ignore"):
            return "funny"
        else:
            return "calm"
        
        # Make more audio rules if wanted

    except Exception as e:
        print(f"Error processing audio for {video_path}: {e}")
        return "error"

def extract_video(video_path):
    """Analyze video frames and video patterns"""
    try:
        # Load YOLO model
        model = YOLO("yolov8n.pt")

        # Open video file
        clip = cv2.VideoCapture(video_path)
        action_objects = {"explosion", "gun", "vechicle"}   # Can Update to your liking
        detected_objects = set()

        while clip.isOpened():
            ret, frame = clip.read()
            if not ret:
                break

            # Get the objects
            results = model(frame)
            for result in results:
                detected_objects.update(result.names)
        
        # End Clip
        clip.release()

        # Rules
        if action_objects & detected_objects:   # Check if they both were found
            return "action"
        elif "funny_face" in detected_objects:  # Face cam applicable
            return "funny"
        else:
            return "calm"
    except Exception as e:
        print(f"Error processing visuals for {video_path}: {e}")
        return "error"
    
def get_clips(video_path):
    """Classify a clip based on audio and visual features."""
    audio_label = extract_audio(video_path)
    visual_label = extract_video(video_path)

    # Combine rules logic
    if "action" in (audio_label, visual_label):
        return "action"
    elif "funny" in (audio_label, visual_label):
        return "funny"
    return "calm"

def process_clips(input_dir, output):
    """
    Process all clips in the directory and classify them.
    Input: input_dir = Directory of clips
    Output: output = Json metadata
    """

    metadata = {} 
    for filename in os.listdir(input_dir):
        if filename.endswith(".mp4"):
            file_path = os.path.join(input_dir, filename)
            label = get_clips(file_path)
            metadata[filename] = {"label": label}
            print(f"Processed {filename}: {label}")
    
    # Save metadata to JSON
    with open(output, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"Metadata saved  to {output}")


# Run script
if __name__ == "__main__":
    input_directory = "clips/"
    output_metadata = "metadata.json"
    process_clips(input_directory, output_metadata)