# ✋ Sign Language Recognizer using AI 🤖

A deep learning-based project that recognizes sign language gestures in real-time using a **Convolutional Neural Network (CNN)** and converts them into **readable text**. Designed to help bridge the communication gap between speech and hearing-impaired individuals.

---

## 🚀 Project Overview

This project captures hand gestures through a **webcam**, detects the shape and position of the hand using **MediaPipe** and **OpenCV**, and classifies them into corresponding **letters, numbers, or common words** using a **CNN model** trained on gesture datasets.

---

## 🧠 Key Features

✅ Real-time gesture recognition via webcam
✅ Converts recognized gestures into readable text
✅ Pre-trained CNN model for high accuracy
✅ Streamlit / OpenCV-based user interface
✅ Easy to use and extend for custom gestures

---

## 🛠️ Technologies Used

| Category             | Tools / Libraries         |
| -------------------- | ------------------------- |
| Programming Language | Python                    |
| Deep Learning        | TensorFlow / Keras        |
| Computer Vision      | OpenCV, MediaPipe         |
| Frontend / UI        | Streamlit                 |
| Others               | NumPy, Pandas, Matplotlib |

---

## 📂 Project Structure

```
SignLanguageRecognizer/
│
├── app.py                     # Main application script
├── model/                     # Trained CNN model files (.h5)
├── dataset/                   # Gesture image dataset
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation
└── utils/                      # Helper functions (if any)
```

---

## ⚙️ Installation & Setup

Follow these steps to run the project locally:

### Step 1: Clone the Repository

```bash
git clone https://github.com/<your-username>/SignLanguageRecognizer.git
cd SignLanguageRecognizer
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Run the Application

If using **Streamlit**:

```bash
streamlit run app.py
```

If using **OpenCV**:

```bash
python app.py
```

---

## 📸 Screenshots

<br><img width="1919" height="876" alt="Screenshot 2025-11-05 083037" src="https://github.com/user-attachments/assets/94a0bfaf-f3dc-45a3-b571-150b42a5e563" /></br>
<br><img width="1170" height="316" alt="Screenshot 2025-11-05 090207" src="https://github.com/user-attachments/assets/07d85553-bc6b-4dcb-9bab-077fdcf1aab3" /></br>
<br><img width="917" height="298" alt="Screenshot 2025-11-05 090133" src="https://github.com/user-attachments/assets/fc99c250-ba48-4aa8-b9dc-a95776998786" /></br>



| Gesture | Prediction |
| ------- | ---------- |
| ✋       | Hello      |
| 👍      | Yes        |
| 👎      | No         |
| 🤞      | Thank You  |

---

## 🧩 Model Information

The CNN model was trained on a dataset of hand gesture images representing alphabets (A–Z), numbers (0–9), and commonly used signs.
You can retrain the model or fine-tune it for new gestures using the provided training script.

---

## 💡 Future Enhancements

* Add support for dynamic gestures (motion-based)
* Include speech synthesis for recognized text
* Build a mobile-friendly version using **TensorFlow Lite**
* Add multilingual gesture datasets

---

## 👩‍💻 Author

**Soundarya G M**
🎓 B.Tech in Artificial Intelligence and Machine Learning
🏫 Srinivas University Institute of Engineering and Technology

> “Nothing feels better in this world.” 💫

