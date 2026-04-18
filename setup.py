from setuptools import setup, find_packages

setup(
    name="asl-recognition",
    version="1.0.0",
    description="Real-time American Sign Language Recognition System",
    author="Your Name",
    author_email="your.email@example.com",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "mediapipe>=0.10.0",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "scipy>=1.11.0",
        "pyttsx3>=2.90",
        "gtts>=2.3.0",
        "onnx>=1.14.0",
        "onnxruntime>=1.16.0",
    ],
    entry_points={
        "console_scripts": [
            "asl-recognize=src.main:main",
        ],
    },
)
