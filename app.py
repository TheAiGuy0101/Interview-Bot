import streamlit as st
from openai import OpenAI
import pyaudio
import wave
import numpy as np
from pydub import AudioSegment
import io
import logging
import tempfile
import os
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(filename='interview_bot.log', level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Set up OpenAI client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No OpenAI API key found. Please check your .env file.")

client = OpenAI(api_key=api_key)

# Create audio recordings directory
if not os.path.exists('audio_recordings'):
    os.makedirs('audio_recordings')

def read_questions(file_path):
    """Read questions from a file."""
    try:
        with open(file_path, 'r') as file:
            questions = [line.strip() for line in file]
        logging.info(f"Successfully read {len(questions)} questions from {file_path}")
        return questions
    except Exception as e:
        logging.error(f"Error reading questions file: {str(e)}")
        st.error(f"Error reading questions file: {str(e)}")
        return []

def record_audio():
    """Record audio using PyAudio until stopped."""
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 44100
    
    p = pyaudio.PyAudio()
    
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)
    
    frames = []
    recording = True
    
    def stop_recording():
        nonlocal recording
        recording = False
    
    stop_button = st.button("Stop Recording")
    
    st.write("Recording... Press 'Stop Recording' when you're done.")
    
    while recording:
        if stop_button:
            break
        data = stream.read(CHUNK)
        frames.append(data)
        
    st.write("Recording finished.")
    
    stream.stop_stream()
    stream.close()
    p.terminate()
    
    return frames, RATE

def save_audio(frames, rate, question_number):
    """Save audio to a WAV file."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_recordings/question_{question_number}_{timestamp}.wav"
        wf = wave.open(filename, 'wb')
        wf.setnchannels(1)
        wf.setsampwidth(pyaudio.PyAudio().get_sample_size(pyaudio.paInt16))
        wf.setframerate(rate)
        wf.writeframes(b''.join(frames))
        wf.close()
        logging.info(f"Audio saved as {filename}")
        return filename
    except Exception as e:
        logging.error(f"Error saving audio: {str(e)}")
        st.error(f"An error occurred while saving audio: {str(e)}")
        return None

def transcribe_audio(audio_file):
    """Transcribe audio using OpenAI's Whisper model."""
    try:
        with open(audio_file, "rb") as audio:
            transcript = client.audio.transcriptions.create(
                model="whisper-1", 
                file=audio
            )
        logging.info(f"Audio transcribed: {transcript.text[:50]}...")
        return transcript.text
    except Exception as e:
        logging.error(f"Error during audio transcription: {str(e)}")
        st.error(f"An error occurred during transcription: {str(e)}")
        return None

def evaluate_answer(question, answer):
    """Evaluate the answer using ChatGPT."""
    try:
        prompt = f"Question: {question}\nAnswer: {answer}\n\nEvaluate this answer on a scale of 1-10 and provide a possible best answer. Start your response with the numeric score."
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an interview evaluator. Evaluate the answer and provide a score out of 10 along with a possible best answer. Always start your response with the numeric score."},
                {"role": "user", "content": prompt}
            ]
        )
        evaluation = response.choices[0].message.content.strip()
        
        # Extract score from the evaluation
        score_str = evaluation.split()[0]
        try:
            score = int(score_str)
            if score < 1 or score > 10:
                raise ValueError("Score out of range")
        except ValueError:
            logging.warning(f"Could not parse score from '{score_str}'. Using default score of 5.")
            score = 5  # Default score if parsing fails
        
        logging.info(f"Answer evaluated. Score: {score}/10")
        return score, evaluation
    except Exception as e:
        logging.error(f"Error during answer evaluation: {str(e)}")
        st.error(f"An error occurred during evaluation: {str(e)}")
        return None, None

def save_transcript(questions, answers, scores, evaluations):
    """Save the interview transcript to a text file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"interview_transcript_detailed_{timestamp}.txt"
    try:
        with open(filename, 'w') as file:
            for i, (question, answer, score, evaluation) in enumerate(zip(questions, answers, scores, evaluations), 1):
                file.write(f"Question {i}: {question}\n")
                file.write(f"Your answer: {answer}\n")
                file.write(f"Score: {score}/10\n")
                file.write(f"Evaluation: {evaluation}\n")
                file.write(f"Audio file: audio_recordings/question_{i}_{timestamp}.wav\n\n")
        st.success(f"Detailed transcript saved as {filename}")
        logging.info(f"Detailed transcript saved as {filename}")
    except Exception as e:
        st.error(f"Failed to save detailed transcript: {str(e)}")
        logging.error(f"Error saving detailed transcript: {str(e)}")

def save_user_transcript(questions, answers):
    """Save the user's responses to a text file."""
    filename = "interview_transcript.txt"
    try:
        with open(filename, 'w') as file:
            for i, (question, answer) in enumerate(zip(questions, answers), 1):
                file.write(f"Question {i}: {question}\n")
                file.write(f"Your answer: {answer}\n\n")
        st.success(f"User transcript saved as {filename}")
        logging.info(f"User transcript saved as {filename}")
    except Exception as e:
        st.error(f"Failed to save user transcript: {str(e)}")
        logging.error(f"Error saving user transcript: {str(e)}")

def main():
    st.title("Interview Bot")

    # Check if API key is available
    if not api_key:
        st.error("OpenAI API key not found. Please check your .env file.")
        return

    questions = read_questions("questions.txt")

    if 'current_question' not in st.session_state:
        st.session_state.current_question = 0
    if 'answers' not in st.session_state:
        st.session_state.answers = []
    if 'scores' not in st.session_state:
        st.session_state.scores = []
    if 'evaluations' not in st.session_state:
        st.session_state.evaluations = []
    if 'attempted' not in st.session_state:
        st.session_state.attempted = set()
    if 'skipped' not in st.session_state:
        st.session_state.skipped = set()

    if questions:
        st.write(f"Question {st.session_state.current_question + 1} of {len(questions)}:")
        st.write(questions[st.session_state.current_question])

        if st.button("Start Recording"):
            frames, rate = record_audio()
            
            audio_file = save_audio(frames, rate, st.session_state.current_question + 1)
            
            if audio_file:
                with st.spinner("Transcribing and evaluating your answer..."):
                    transcript = transcribe_audio(audio_file)
                    if transcript:
                        st.session_state.answers.append(transcript)
                        score, evaluation = evaluate_answer(questions[st.session_state.current_question], transcript)
                        if score is not None:
                            st.session_state.scores.append(score)
                            st.session_state.evaluations.append(evaluation)
                            st.session_state.attempted.add(st.session_state.current_question)
                            st.write(f"Your answer: {transcript}")
                            st.write(f"Evaluation: {evaluation}")
                            st.write(f"Score: {score}/10")
                        else:
                            st.error("Failed to evaluate the answer. Please try again.")
                    else:
                        st.error("Failed to transcribe the audio. Please try again.")
            else:
                st.error("Failed to save the audio. Please try again.")

        # Navigation buttons
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Previous") and st.session_state.current_question > 0:
                st.session_state.current_question -= 1

        with col2:
            if st.button("Skip"):
                st.session_state.skipped.add(st.session_state.current_question)
                st.session_state.current_question = min(st.session_state.current_question + 1, len(questions) - 1)

        with col3:
            if st.button("Next") and st.session_state.current_question < len(questions) - 1:
                st.session_state.current_question += 1

        # Interview progress
        st.sidebar.header("Interview Progress")
        st.sidebar.write(f"Questions attempted: {len(st.session_state.attempted)}")
        st.sidebar.write(f"Questions skipped: {len(st.session_state.skipped)}")
        if st.session_state.scores:
            average_score = sum(st.session_state.scores) / len(st.session_state.scores)
            st.sidebar.write(f"Average score: {average_score:.2f}/10")

        # Finish button
        if st.button("Finish Interview"):
            st.write("Interview completed!")
            total_questions = len(questions)
            attempted = len(st.session_state.attempted)
            skipped = len(st.session_state.skipped)
            correct = sum(1 for score in st.session_state.scores if score >= 7)  # Assuming a score of 7 or higher is correct
            wrong = attempted - correct

            st.write(f"Total questions: {total_questions}")
            st.write(f"Questions attempted: {attempted}")
            st.write(f"Questions skipped: {skipped}")
            st.write(f"Correct answers: {correct}")
            st.write(f"Wrong answers: {wrong}")

            if st.session_state.scores:
                total_score = sum(st.session_state.scores)
                average_score = total_score / attempted if attempted > 0 else 0
                st.write(f"Total score: {total_score}/{attempted*10}")
                st.write(f"Average score: {average_score:.2f}/10")
                logging.info(f"Interview completed. Total score: {total_score}/{attempted*10}")
                
                # Save detailed transcript
                save_transcript(questions, st.session_state.answers, st.session_state.scores, st.session_state.evaluations)
                
                # Save user transcript
                save_user_transcript(questions, st.session_state.answers)
            else:
                st.write("No scores recorded.")
                logging.info("Interview completed. No scores recorded.")

            for i, question in enumerate(questions):
                st.write(f"\nQuestion {i+1}: {question}")
                if i in st.session_state.attempted:
                    answer_index = list(st.session_state.attempted).index(i)
                    st.write(f"Your answer: {st.session_state.answers[answer_index]}")
                    st.write(f"Score: {st.session_state.scores[answer_index]}/10")
                elif i in st.session_state.skipped:
                    st.write("Skipped")
                else:
                    st.write("Not attempted")

    else:
        st.error("No questions available. Please check the questions file.")
        logging.error("No questions available. Unable to start the interview.")

if __name__ == "__main__":
    main()