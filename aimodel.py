import cv2
from ultralytics import YOLO
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import numpy as np
import whisper
import contextlib
import os
import json

print("All packages imported successfully!")

def get_audio_intensity(audio):
        """Analyze the total duration of nonsilent segments."""
        nonsilent = detect_nonsilent(audio, min_silence_len=300, silence_thresh=-25)
        loud = sum((end - start) for start, end in nonsilent) / 1000 # Second conversion
        return loud

def analyze_speech(audio_path):
    """Transcribe audio and analyze for keywords."""
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    speech = result["text"].lower()

    if any(word in speech for word in ["shoot", "run", "hardpoint", "fire", "explode"]):
        return "action"
    elif any(word in speech for word in ["laugh", "funny", "joke"]):
        return "funny"
    return "calm"

def extract_audio(video_path):
    """Analyze audio for sound patterns and volume"""
    try:
        # Load audio from video
        audio = AudioSegment.from_file(video_path)

        # Detect louder moments
        nonsilent = get_audio_intensity(audio)

        # Speech analysis
        speech_label = analyze_speech(video_path)

        # Rules
        if nonsilent > 30 or speech_label == "action": # More than 30 loud segments detected
            return "action"
        elif speech_label == "funny":
            return "funny"
        else:
            return "calm"
        
        # Make more audio rules if wanted

    except Exception as e:
        print(f"Error processing audio for {video_path}: {e}")
        return "error"

def count_action_objects(dectected):
    """Count occurrences of action-related objects."""
    action_objects = {"gun", "explosion", "car", "knife"}
    return sum(1 for object in dectected if object in action_objects)

def new_scene(video_path):
    """Detect new scene changes."""
    capture = cv2.VideoCapture(video_path)
    previous_frame = None
    changes = 0

    while capture.isOpened():
        ret, frame = capture.read()
        if not ret:
            break
        if previous_frame is not None:
            diff = cv2.absdiff(frame, previous_frame)
            counter = cv2.countNonZero(cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY))
            if counter > 5000: # New scene detected
                changes += 1
        previous_frame = frame
    capture.release()
    return changes


def extract_video(video_path):
    """Analyze video frames and video patterns"""
    try:
        # Load YOLO model
        model = YOLO("yolov8n.pt", verbose=False)

        # Open video file
        clip = cv2.VideoCapture(video_path)
        detected_objects = []

        while clip.isOpened():
            ret, frame = clip.read()
            if not ret:
                break
            
            results = model(frame)
            for result in results:
                detected_objects.extend(result.names)

        # End Clip
        clip.release()

        # Scence changes
        scences = new_scene(video_path)

        # Combine results for action
        action_count = count_action_objects(detected_objects)

        # Rules
        if action_count > 10 or scences > 20:   # Check if they both were found
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
    try:
        assert isinstance(metadata, dict)
        with open(output, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"Metadata saved  to {output}")
    except Exception as e:
        print(f"Error saving metadata to JSOJN: {e}")


# Run script
if __name__ == "__main__":
    input_directory = "clips/"
    output_metadata = "metadata.json"

    if not os.path.exists(input_directory):
        print(f"Error: Input directory '{input_directory}' does not exist.")
        exit(1)

    process_clips(input_directory, output_metadata)