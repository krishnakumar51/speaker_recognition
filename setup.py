from setuptools import setup, find_packages

setup(
    name="speaker-recognition-app",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit",
        "sounddevice",
        "numpy",
        "matplotlib",
        "torchaudio",
        "speechbrain",
        "scikit-learn",
        "librosa",
        "pyaudio"
    ],
    python_requires=">=3.10",
    description="A Streamlit-based speaker recognition app",
    author="Krishna Kumar",
    author_email="godkrishna@gmail.com",
    url="https://github.com/krishnakumar51/speaker_recognition.git",
)
