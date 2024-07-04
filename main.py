import openai
import streamlit as st
import pyaudio
import wave
import os
import cv2
import logging
from datetime import datetime, timedelta
from gtts import gTTS
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Set up OpenAI API key from environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize PyAudio
p = pyaudio.PyAudio()

# Set up logging
logging.basicConfig(level=logging.INFO)

# Function to start recording audio
def start_recording(output_file="recorded_audio.wav"):
    try:
        chunk = 1024
        sample_format = pyaudio.paInt16
        channels = 1
        fs = 16000
        stream = p.open(format=sample_format, channels=channels, rate=fs, frames_per_buffer=chunk, input=True)
        frames = []

        st.session_state.is_recording = True
        logging.info("Started recording")

        while st.session_state.is_recording:
            data = stream.read(chunk)
            frames.append(data)

        stream.stop_stream()
        stream.close()

        wf = wave.open(output_file, 'wb')
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(sample_format))
        wf.setframerate(fs)
        wf.writeframes(b''.join(frames))
        wf.close()

        logging.info(f"Recording saved to {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Error during recording: {e}")
        st.error("An error occurred during recording. Please try again.")

# Function to stop recording audio
def stop_recording():
    st.session_state.is_recording = False
    logging.info("Stopped recording")

# Function to transcribe audio using OpenAI Whisper
def transcribe_audio(audio_file):
    try:
        with open(audio_file, "rb") as f:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        logging.info(f"Transcription completed for {audio_file}")
        return transcript.text
    except Exception as e:
        logging.error(f"Error during transcription: {e}")
        st.error("An error occurred during transcription. Please try again.")

# Function to evaluate the answer using GPT-4 and provide a rating
def evaluate_answer(question, answer):
    try:
        prompt = f"Evaluate the following answer to the question '{question}':\n\nAnswer: {answer}\n\nProvide feedback on the strengths, weaknesses, and areas for improvement. Give a rating between 0 to 10 where 0 is wrong and 10 is almost correct."
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an evaluation assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=150,
            temperature=0.7,
            n=1,
            stop=None
        )
        logging.info(f"Evaluation completed for question: {question}")
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"Error during evaluation: {e}")
        st.error("An error occurred during evaluation. Please try again.")

# Function to read questions from a file
def read_questions_from_file(file):
    try:
        if file.name.endswith('.txt') or file.name.endswith('.md'):
            questions = file.read().decode('utf-8').split('\n')
            questions = [q.split('. ', 1)[1].strip() for q in questions if q.strip()]
            logging.info(f"Loaded {len(questions)} questions from file.")
            return questions
        else:
            st.error("Please upload a valid .txt or .md file")
            return []
    except Exception as e:
        logging.error(f"Error reading questions from file: {e}")
        st.error("An error occurred while reading the file. Please try again.")

# Function to capture a photo using the webcam
def capture_photo(output_file="photo.jpg"):
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Could not open webcam")
            return None

        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture image")
            return None

        cv2.imwrite(output_file, frame)
        cap.release()
        logging.info(f"Photo captured and saved to {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Error capturing photo: {e}")
        st.error("An error occurred while capturing the photo. Please try again.")

# Function to convert text to speech
def text_to_speech(text, output_file="question.mp3"):
    try:
        tts = gTTS(text)
        tts.save(output_file)
        logging.info(f"Text-to-speech conversion completed and saved to {output_file}")
        return output_file
    except Exception as e:
        logging.error(f"Error during text-to-speech conversion: {e}")
        st.error("An error occurred during text-to-speech conversion. Please try again.")

# Streamlit UI
def main():
    st.title("Voice-Based Mock Interview Bot")

       # Initialize session state variables
    if 'questions' not in st.session_state:
        st.session_state.questions = []

    if 'question_index' not in st.session_state:
        st.session_state.question_index = 0

    if 'last_photo_time' not in st.session_state:
        st.session_state.last_photo_time = datetime.now()

    if 'answers' not in st.session_state:
        st.session_state.answers = []

    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False

    # File uploader
    uploaded_file = st.file_uploader("Upload a text or markdown file with interview questions", type=['txt', 'md'])
    if uploaded_file is not None:
        st.session_state.questions = read_questions_from_file(uploaded_file)
        st.session_state.question_index = 0
        st.session_state.answers = []
        st.session_state.is_recording = False

    # Display current question
    if st.session_state.questions and st.session_state.question_index < len(st.session_state.questions):
        question = st.session_state.questions[st.session_state.question_index]
        st.header(f"Question {st.session_state.question_index + 1}: {question}")

        try:
            tts_file = text_to_speech(question, f"question_{st.session_state.question_index}.mp3")
            if os.path.exists(tts_file):
                st.audio(tts_file, format='audio/mp3')
            else:
                st.error("Failed to create audio file.")
        except Exception as e:
            logging.error(f"Error playing TTS audio: {e}")
            st.error("An error occurred while playing the audio. Please try again.")

        if not st.session_state.is_recording:
            st.session_state.is_recording = True
            start_recording(f"answer_{st.session_state.question_index}.wav")

        
            if st.button("Next Question"):
                 if st.session_state.is_recording:
                    stop_recording()
                    st.session_state.answers.append(f"answer_{st.session_state.question_index}.wav")
                    st.session_state.question_index += 1
                    st.session_state.is_recording = False
                    st.session_state.question_index += 1
                    st.experimental_rerun()
                    
            if st.button("Finish Interview"):
                if st.session_state.is_recording:
                    stop_recording()
                    st.session_state.answers.append(f"answer_{st.session_state.question_index}.wav")
                st.session_state.question_index = len(st.session_state.questions)
                st.experimental_rerun()  # Force rerun to end the interview

        # Capture a photo every 5 minutes
        if datetime.now() - st.session_state.last_photo_time >= timedelta(minutes=5):
            photo_file = capture_photo()
            if photo_file:
                st.image(photo_file, caption="Captured photo")
                st.session_state.last_photo_time = datetime.now()
    elif st.session_state.question_index >= len(st.session_state.questions) and st.session_state.questions:
        # Interview completion logic here
        st.header("Interview Complete")
        st.write("Processing answers...")

        for i, question in enumerate(st.session_state.questions):
            st.subheader(f"Question {i + 1}")
            st.write(question)
            audio_file = st.session_state.answers[i]
            answer = transcribe_audio(audio_file)
            st.write("Transcribed Answer: " + answer)

            feedback = evaluate_answer(question, answer)
            st.subheader("Feedback")
            st.write(feedback)

        if st.button("Restart Interview"):
            st.session_state.question_index = 0
            st.session_state.questions = []
            st.session_state.answers = []

if __name__ == "__main__":
    main()
