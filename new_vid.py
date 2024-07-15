import cv2
import numpy as np
from PIL import ImageFont, ImageDraw, Image
import assemblyai as aai
import math
import cv2
import os
import numpy as np
from pydub import AudioSegment
import subprocess
import pyttsx3
from pydub import AudioSegment
import io
from PIL import Image
import random
import requests
import google.generativeai as genai
from moviepy.editor import *
def createVideo(topic='',description=""):
    GOOGLE_API_KEY='YOUR_API_KEY'
    genai.configure(api_key=GOOGLE_API_KEY)
    # Model Configuration
    MODEL_CONFIG = {
    "temperature": 1.99,
    "top_p": 0.4,
    "top_k": 32,
    "max_output_tokens": 4096,
    }

    ## Safety Settings of Model
    safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE"
    }
    ]# Model Configuration
    GOOGLE_API_KEY='YOUR_API_KEY'

    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(model_name = "models/gemini-1.0-pro-latest",
                                generation_config = MODEL_CONFIG,
                                safety_settings = safety_settings)
    
    l=['random facts'] if topic=='' else [topic]
    parse=random.choice(l)
    print(parse)
    messages=[
            {
                "role": "model",
                "parts": ["""You are a YouTube short narration generator. You generate 30 seconds to 1 minute of narration. The shorts you create have a background that fades from image to image as the narration is going on.

    You will need to generate descriptions of images for each of the sentences in the short. They will be passed to an AI image generator. DO NOT IN ANY CIRCUMSTANCES use names of celebrities or people in the image descriptions. It is illegal to generate images of celebrities. Only describe persons without their names. Do not reference any real person or group in the image descriptions. Don't mention the female figure or other sexual content in the images because they are not allowed.

    You are however allowed to use any content, including real names in the narration. Only image descriptions are restricted.

    Note that the narration will be fed into a text-to-speech engine, so don't use special characters.

    Respond with a pair of an image description in square brackets and a narration below it. Both of them should be on their own lines, as follows:

    ###

    [Description of a background image]

    Narrator: "One sentence of narration"

    [Description of a background image]

    Narrator: "One sentence of narration"

    [Description of a background image]

    Narrator: "One sentence of narration"

    ###

    The short should be 6 sentences 
    make sure the description of the image is equal to narration
    You should add a description of a fitting backround image in between all of the narrations. It will later be used to generate an image with AI.}
    """
    ]},
            {
                "role": "user",
                "parts": [f"Create a YouTube short narration based on the following source material:\n\n {parse+description}"]
            }
        ]
    l=model.generate_content(messages).text.split('\n')
    imagecontent=[]
    speechcontent=[]
    for i in l:
        if '[' in i and ']' in i:
            imagecontent.append(i[1:len(i)-1])
        if 'Narrator:' in i:
            speechcontent.append(i.split('Narrator:')[1].replace('"',""),)
    speechdata=''.join(speechcontent)

    API_URL = "https://api-inference.huggingface.co/models/fluently/Fluently-XL-Final"
    headers = {"Authorization": "Bearer YOUR_API_KEY"}
    def crop_to_aspect_ratio(image, aspect_ratio):
        width, height = image.size
        target_width = width
        target_height = int(target_width / aspect_ratio)

        if target_height > height:
            target_height = height
            target_width = int(target_height * aspect_ratio)

        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2

        return image.crop((left, top, right, bottom))
    def query(payload):
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.content
    # l=prompt.text.split("Rephrased:")[1].split('\n')

    l1=imagecontent
    images=[]
    j=0
    for i in range(len(l1)):
        while not os.path.exists("generated_"+str(i)+".jpg"):
            try:
                image_bytes = query({
                    "inputs": l1[i],
                })
                print(image_bytes)
                image = Image.open(io.BytesIO(image_bytes))
                image=crop_to_aspect_ratio(image,9/16)
                images.append(image)
                image.save("generated_"+str(i)+".jpg")
            except Exception as e:
                if not os.path.exists("generated_"+str(i)+".jpg"):
                    l1[i]=model.generate_content("rephrase the sentence "+l1[i]).text
                
    def text_to_speech(text, output_file,gender='MALE'):
        # Initialize the TTS engine
        engine = pyttsx3.init()
        voices = engine.getProperty('voices')
        x=1 if gender=="MALE" else 0
        engine.setProperty('voice', voices[x].id)
        # Set properties (optional)
        engine.setProperty('volume', 1.0)  # Volume (0.0 to 1.0)
        # Save the speech to a file
        rate = engine.getProperty('rate')
        engine.setProperty('rate', 130)
        engine.save_to_file(text, output_file)
        
        # Run the speech engine
        engine.runAndWait()
    for i in range(len(speechcontent)):
        text_to_speech(speechcontent[i],f"output{i}.mp3")

    def get_audio_duration(audio_file):
        print(f"Loading audio file: {audio_file}")
        return len(AudioSegment.from_file(audio_file))

    def add_narration_to_video(narrations, input_video, output_dir, output_file):
        full_narration = AudioSegment.empty()

        for narration in narrations:
            full_narration += AudioSegment.from_file(narration)

        temp_narration = os.path.join(output_dir, "narration.mp3")
        full_narration.export(temp_narration, format="mp3")

        ffmpeg_command = [
            'ffmpeg',
            '-y',
            '-i', input_video,
            '-i', temp_narration,
            '-map', '0:v',   
            '-map', '1:a',  
            '-c:v', 'copy',  
            '-c:a', 'aac',  
            '-strict', 'experimental',
            os.path.join(output_dir, output_file)
        ]

        subprocess.run(ffmpeg_command, capture_output=True)

        os.remove(temp_narration)

    def resize_image(image, width, height):
        aspect_ratio = image.shape[1] / image.shape[0]

        if aspect_ratio > (width / height):
            new_width = width
            new_height = int(width / aspect_ratio)
        else:
            new_height = height
            new_width = int(height * aspect_ratio)

        resized_image = cv2.resize(image, (new_width, new_height))

        # Padding the resized image to match the target size
        top = (height - new_height) // 2
        bottom = height - new_height - top
        left = (width - new_width) // 2
        right = width - new_width - left

        padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT)

        return padded_image

    def create(narrations, images, output_dir, output_filename, caption_settings=None):
        if caption_settings is None:
            caption_settings = {}

        width, height = 1080, 1920  # Change as needed for your vertical video
        frame_rate = 30  # Adjust as needed

        fade_time = 1000

        fourcc = cv2.VideoWriter_fourcc(*'XVID')  # You can change the codec as needed
        temp_video = os.path.join(output_dir, "temp_video.avi")  # Output video file name
        out = cv2.VideoWriter(temp_video, fourcc, frame_rate, (width, height))

        image_count = len(images)

        for i in range(image_count):
            image1 = cv2.imread(images[i])

            if i+1 < image_count:
                image2 = cv2.imread(images[i+1])
            else:
                image2 = cv2.imread(images[0])

            image1 = resize_image(image1, width, height)
            image2 = resize_image(image2, width, height)

            narration = narrations[i]
            duration = get_audio_duration(narration)

            if i > 0:
                duration -= fade_time

            if i == image_count-1:
                duration -= fade_time

            for _ in range(math.floor(duration / 1000 * 30)):
                vertical_video_frame = np.zeros((height, width, 3), dtype=np.uint8)
                vertical_video_frame[:image1.shape[0], :] = image1

                out.write(vertical_video_frame)

            for alpha in np.linspace(0, 1, math.floor(fade_time / 1000 * 30)):
                blended_image = cv2.addWeighted(image1, 1 - alpha, image2, alpha, 0)
                vertical_video_frame = np.zeros((height, width, 3), dtype=np.uint8)
                vertical_video_frame[:image1.shape[0], :] = blended_image

                out.write(vertical_video_frame)

        out.release()
        cv2.destroyAllWindows()

        with_narration = output_filename
        add_narration_to_video(narrations, temp_video, output_dir, with_narration)
    narrations = [
        f'output{i}.mp3' for i in range(len(speechcontent))
    ]

    images = [
        f'generated_{i}.jpg' for i in range(len(speechcontent))
    ]

    output_dir = "D://Ai//video//"
    output_filename = "final_video.mp4"

    # Call the create function with the prepared arguments
    create(narrations, images, output_dir, output_filename)
    aai.settings.api_key = "API_KEY"
    transcript = aai.Transcriber().transcribe("final_video.mp4")
    print(transcript.words)
    print(transcript.text)
    if transcript.status == aai.TranscriptStatus.error:
        print(f"Transcription failed: {transcript.error}")
    # Function to convert the transcript to SRT format with word-level timestamps
    def transcript_to_srt(transcript):
        subtitles = []
        for i, word in enumerate(transcript.words):
            start = word.start / 1000  # convert to seconds
            end = word.end / 1000  # convert to seconds
            text = word.text
            start_time = f"{int(start // 3600):02}:{int((start % 3600) // 60):02}:{int(start % 60):02},{int(start * 1000 % 1000):03}"
            end_time = f"{int(end // 3600):02}:{int((end % 3600) // 60):02}:{int(end % 60):02},{int(end * 1000 % 1000):03}"
            subtitles.append(f"{i+1}\n{start_time} --> {end_time}\n{text}\n")

        return "\n".join(subtitles)

    # Convert the transcript to SRT format
    subtitles = transcript_to_srt(transcript)

    # Save the subtitles to a .srt file
    subtitle_file = "subtitles.srt"
    with open(subtitle_file, "w") as f:
        f.write(subtitles)

    print(f"Subtitles saved to {subtitle_file}")
    import pysrt

    # Read the SRT file
    srt_file = 'subtitles.srt'
    subs = pysrt.open(srt_file)

    # Helper function to convert time to seconds
    def time_to_seconds(time):
        return time.hours * 3600 + time.minutes * 60 + time.seconds + time.milliseconds / 1000.0

    # Create segments with start and end times as floats
    segments = []
    for sub in subs:
        start_time = time_to_seconds(sub.start)
        end_time = time_to_seconds(sub.end)
        text = sub.text
        segment = {
            'start': start_time,
            'end': end_time,
            'word': text
        }
        segments.append(segment)

    def split_text_into_lines(data):

        MaxChars = 30
        #maxduration in seconds
        MaxDuration = 2.5
        #Split if nothing is spoken (gap) for these many seconds
        MaxGap = 1.5

        subtitles = []
        line = []
        line_duration = 0
        line_chars = 0


        for idx,word_data in enumerate(data):
            word = word_data["word"]
            start = word_data["start"]
            end = word_data["end"]

            line.append(word_data)
            line_duration += end - start

            temp = " ".join(item["word"] for item in line)


            # Check if adding a new word exceeds the maximum character count or duration
            new_line_chars = len(temp)

            duration_exceeded = line_duration > MaxDuration
            chars_exceeded = new_line_chars > MaxChars
            if idx>0:
                gap = word_data['start'] - data[idx-1]['end']
                # print (word,start,end,gap)
                maxgap_exceeded = gap > MaxGap
            else:
                maxgap_exceeded = False


            if duration_exceeded or chars_exceeded or maxgap_exceeded:
                if line:
                    subtitle_line = {
                        "word": " ".join(item["word"] for item in line),
                        "start": line[0]["start"],
                        "end": line[-1]["end"],
                        "textcontents": line
                    }
                    subtitles.append(subtitle_line)
                    line = []
                    line_duration = 0
                    line_chars = 0


        if line:
            subtitle_line = {
                "word": " ".join(item["word"] for item in line),
                "start": line[0]["start"],
                "end": line[-1]["end"],
                "textcontents": line
            }
            subtitles.append(subtitle_line)

        return subtitles
    linelevel=split_text_into_lines(segments)

    def format_string(input_string, max_width=30):
        if len(input_string) <= max_width:
            return input_string
        lines = []
        current_line = ""
        for word in input_string.split():
            if len(current_line) + len(word) + 1 <= max_width:
                current_line += " " + word if current_line else word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        formatted_string = "\n".join(lines)
        return formatted_string

    def load_subtitles_from_json(file_path):
        subs = file_path
        subtitle_list = []
        for sub in subs:
            start = sub["start"]
            end = sub["end"]
            words = [(word_info["start"], word_info["end"], word_info["word"]) for word_info in sub["textcontents"]]
            formatted_text = format_string(sub["word"]).upper()
            subtitle_list.append((start, end, formatted_text, words))
        return subtitle_list

    def put_fancy_text_on_frame(frame, text, position, font_path='arial.ttf', font_size=48, text_color=(30, 240, 30), outline_color=(0, 0, 0), highlight_word=None, highlight_color=(255, 0, 0)):
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        try:
            font = ImageFont.truetype(font_path, font_size)
        except OSError:
            font = ImageFont.load_default()
            print(f"Warning: Could not open font resource '{font_path}', using default font.")
        
        lines = text.split('\n')
        total_text_height = sum([draw.textbbox((0, 0), line, font=font)[3] - draw.textbbox((0, 0), line, font=font)[1] for line in lines])
        y = position[1] - total_text_height // 2
        
        for line in lines:
            words = line.split(' ')
            line_width = sum([draw.textbbox((0, 0), word, font=font)[2] for word in words])
            x = position[0] - line_width // 2
            
            for word in words:
                word_bbox = draw.textbbox((0, 0), word, font=font)
                text_width = word_bbox[2] - word_bbox[0]
                text_height = word_bbox[3] - word_bbox[1]
                fill_color = highlight_color if word == highlight_word else text_color
                
                # Draw outline
                draw.text((x-1, y-1), word, font=font, fill=outline_color)
                draw.text((x+1, y-1), word, font=font, fill=outline_color)
                draw.text((x-1, y+1), word, font=font, fill=outline_color)
                draw.text((x+1, y+1), word, font=font, fill=outline_color)
                
                # Draw text
                draw.text((x, y), word, font=font, fill=fill_color)
                x += text_width + draw.textbbox((0, 0), ' ', font=font)[2]
            
            y += text_height + 10
        
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        return frame

    def add_subtitles_to_video(video_path, json_path, output_path, font_path='arial.ttf', font_size=60):
        cap = cv2.VideoCapture(video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        subtitles = load_subtitles_from_json(json_path)
        subtitle_index = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            current_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            if subtitle_index < len(subtitles):
                start, end, text, words = subtitles[subtitle_index]
                highlight_word = None
                for word_start, word_end, word in words:
                    if word_start <= current_time <= word_end:
                        highlight_word = word.upper()
                        break
                if start <= current_time <= end:
                    frame = put_fancy_text_on_frame(frame, text, position=(width // 2, height // 2), font_path=font_path, font_size=font_size, highlight_word=highlight_word)
                elif current_time > end:
                    subtitle_index += 1
            out.write(frame)

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    # Example usage
    video_path = 'final_video.mp4'
    json_path = linelevel
    output_path = 'output_video.mp4'
    add_subtitles_to_video(video_path, json_path, output_path, font_path='Rowdies-Bold.ttf', font_size=50)
    from moviepy.editor import VideoFileClip, concatenate_videoclips

    def add_audio_to_video(video_path, audio_path, output_path):
        # Load the videos
        video_clip = VideoFileClip(video_path)
        audio_clip = VideoFileClip(audio_path)

        # Set the audio of the video clip to the audio clip
        video_clip = video_clip.set_audio(audio_clip.audio)

        # Write the video with merged audio to a file
        video_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')

        # Close the video clips
        video_clip.close()
        audio_clip.close()

    # Example usage
    video_path = 'output_video.mp4'  # Replace with your video file path without audio
    audio_path = 'final_video.mp4'  # Replace with your audio file path
    output_path = 'output_video_with_audio.mp4'  # Replace with your desired output file path

    add_audio_to_video(video_path, audio_path, output_path)
import streamlit as st

st.title("YouTube Short Generator")

if st.button("Generate Video"):
    result = createVideo()
    st.video('output_video_with_audio.mp4')