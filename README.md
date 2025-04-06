# 🤖 Real-Time Hand Gesture Recognition using Vision Transformers

This project demonstrates **real-time hand gesture recognition** by combining **MediaPipe** for hand tracking and a **Vision Transformer (ViT)** for gesture classification. It captures hand landmarks from a live webcam feed, processes them into input tensors, and feeds them into a transformer-based model to predict the user's hand gesture — all in real time.

---

## 📌 Features

- 🖐️ **Real-time hand tracking** using MediaPipe (Google)
- 🧠 **Gesture classification** using pretrained Vision Transformers (HuggingFace ViT)
- ⚡ Fast and lightweight pipeline suitable for modern hardware
- 🧩 Modular code structure — easy to train, test, and extend
- 🎥 Live webcam-based inference with OpenCV

---

## 📁 Project Structure

```
Real-Time-Hand-Gesture-Recognition/
├── models/
│   └── vit_model.py         # ViT model definition
├── utils/
│   ├── dataset.py           # Dataset class for training
│   └── mediapipe_utils.py   # Landmark extraction using MediaPipe
├── train.py                 # Training pipeline
├── inference.py             # Real-time webcam inference
├── requirements.txt         # Python dependencies
├── .gitignore               # Ignore temp files and configs
└── README.md                # You're here!
```

---

## 🚀 Setup

1. **Clone the repository**

```bash
git clone https://github.com/your-username/real-time-gesture-transformer.git
cd real-time-gesture-transformer
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

> Requirements:
> - `torch`, `transformers` (for ViT model)
> - `opencv-python`, `mediapipe` (for hand tracking and webcam feed)
> - `numpy`

---

## 🏋️‍♀️ Training the Model

By default, the project includes a minimal training setup using dummy hand landmark data.

To train:

```bash
python train.py
```

- Loads dummy landmark data (replace this with your real dataset)
- Trains a ViT-based classifier
- Saves the trained model to `gesture_model.pth`

If you have real landmark vectors and gesture labels, update `train.py` with your data loading logic.

---

## 🎯 Real-Time Inference

To run real-time gesture recognition with webcam:

```bash
python inference.py
```

- Starts webcam stream
- Detects hand landmarks using MediaPipe
- Classifies gestures using the trained Vision Transformer model
- Displays gesture predictions over the live video

Press `q` to quit the video window.

---

## 🧠 Model Architecture

We use a pretrained [ViT-B/16](https://huggingface.co/google/vit-base-patch16-224-in21k) model from HuggingFace's Transformers library. The architecture consists of:

- A ViT backbone (frozen or fine-tuned)
- A lightweight classification head (`nn.Linear`) for gesture prediction

You can optionally replace this with a **Swin Transformer**, **CNN**, or **MLP**, depending on your use case.

---

## 🧾 Requirements

List of all required packages:

```
torch
torchvision
torchaudio
transformers
opencv-python
mediapipe
numpy
```

Install via:

```bash
pip install -r requirements.txt
```

---

## 📸 Sample Output (Webcam View)

```text
[Camera Feed]
Detected: ✊ Fist
Detected: 🖐️ Palm
Detected: 🤞 Peace
```

Prediction text will appear at the top-left of your webcam window.

---

## 🛠 Tips for Customization

- Replace dummy data in `train.py` with real recorded hand landmarks and labels
- Modify `gesture_names = [...]` in `inference.py` to reflect your custom gestures
- Add more gesture classes and increase training data for better accuracy
- Try Swin Transformer by replacing the ViT model in `vit_model.py`

---

## 🤝 Acknowledgements

- [MediaPipe](https://google.github.io/mediapipe/) for robust real-time hand tracking
- [HuggingFace Transformers](https://huggingface.co/transformers/) for pretrained ViT models
- [PyTorch](https://pytorch.org/) for model training and deployment

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 💬 Questions or Contributions?

If you found this project helpful, feel free to ⭐ star the repo. Pull requests and issues are welcome!

