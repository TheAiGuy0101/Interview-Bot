# Interview Bot

This is a Streamlit-based application that conducts interviews using OpenAI's speech recognition and language models.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- FFmpeg (for audio processing)

## Installation

1. Clone this repository or download the source code.

2. Navigate to the project directory:

    cd path/to/interview-bot

3. Install FFmpeg:
    - On Windows:
    - Download FFmpeg from https://ffmpeg.org/download.html
    - Extract the downloaded file
    - Add the FFmpeg `bin` folder to your system PATH
    - On macOS (using Homebrew):
    ```
    brew install ffmpeg
    ```
    - On Ubuntu or Debian:
    ```
    sudo apt update && sudo apt install ffmpeg
    ```

4. Create a virtual environment:
    python -m venv venv

5. Activate the virtual environment:
    - On Windows:
    ```
    venv\Scripts\activate
    ```
    - On macOS and Linux:
    ```
    source venv/bin/activate
    ```

6. Install the required packages:
    pip install -r requirements.txt

7. Create a file named `questions.txt` in the project directory and add your interview questions, one per line.

8. Open the `app.py` file and replace `"your_openai_api_key_here"` with your actual OpenAI API key.

## Running the Application

1. Ensure your virtual environment is activated.

2. Run the Streamlit app:
    streamlit run app.py

3. Open a web browser and go to the URL displayed in the terminal (usually http://localhost:8501).

## Usage

- Click "Start Recording" to begin answering a question.
- Speak your answer.
- Click "Stop Recording" when you've finished answering.
- The application will process your answer and provide an evaluation.
- Use the "Previous", "Skip", and "Next" buttons to navigate through questions.
- Click "Finish Interview" to see your final results.

## Logging

The application logs information and errors to a file named `interview_bot.log` in the project directory.

## Troubleshooting

If you encounter any issues with audio recording, make sure you have the appropriate audio drivers installed for your system.

For any other issues, please check the `interview_bot.log` file for error messages.