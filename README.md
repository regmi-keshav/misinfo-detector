# Misinfo-Detector

**AI-Powered Platform for News Credibility & Misinformation Detection**

---

## Why It Matters

Misinformation spreads faster than truth—and the consequences are real. Whether you're a journalist, researcher, educator, or everyday reader, knowing **which news to trust** is critical.

**misinfo-detector** is an intelligent system designed to evaluate the **credibility of news content** using natural language processing, machine learning, and explainable AI. It doesn’t just tell you whether something looks fake—it explains *why*.

---

## What This Tool Does

* Identifies whether a news article is **likely real or fake**
* Performs **linguistic analysis** and evaluates writing patterns
* Assigns a **confidence score** and **risk level**
* Uses **SHAP explanations** to show feature influence
* Offers tools for readability scoring, formatting flags, and metadata review

---

## Features at a Glance

| Feature Category        | Highlights                                                      |
| ----------------------- | --------------------------------------------------------------- |
| **ML-Based Prediction** | Detects misinformation using trained NLP models                 |
| **Explainable AI**      | Uses SHAP to provide transparency into decisions                |
| **Risk & Confidence**   | Assigns severity levels and percentage-based trust scores       |
| **Content Diagnostics** | Flags strange patterns, excessive punctuation, all-caps, etc.   |
| **Visual Insights**     | Charts and graphs reveal writing styles and decision factors    |
| **API Access**          | FastAPI backend available for integration with external systems |
| **Web App Interface**   | Built with Streamlit for real-time news verification            |

---

## Try It Yourself

### Web Interface

Launch the app:
🔗 [Streamlit App](https://regmi-keshav-misinfo-detector-app-yscuev.streamlit.app/)

### API Access

Explore API endpoints:
🔗 [FastAPI Docs](https://misinfo-detector.onrender.com/docs)

---

## Example Use Cases

* **Newsrooms**: Cross-check articles before publication
* **Educators**: Use in digital literacy and critical thinking curricula
* **Social Platforms**: Integrate API to flag suspicious links
* **Researchers**: Study misinformation trends and language patterns

---

## Under the Hood

### Core Components

| Component            | Technology Used             |
| -------------------- | --------------------------- |
| **Frontend**         | Streamlit                   |
| **Backend**          | FastAPI                     |
| **ML Models**        | scikit-learn, SHAP          |
| **Text Processing**  | NLTK, custom NLP utilities  |
| **Visualization**    | Plotly, Matplotlib, Seaborn |
| **Containerization** | Docker                      |

---

## Model Insights

The system was trained on real-world news datasets and evaluated using:

* **Precision-Recall Curves**
* **F1-Score & AUC-PR**
* **Confusion Matrix**

Results showed strong performance in distinguishing true and false content, even with class imbalances.

<img src="./reports/evaluation_metrics/confusion_matrix.png" width="400">
<img src="./reports/evaluation_metrics/precision_recall_curve.png" width="400">

---

## 📂 Project Layout

```bash
misinfo-detector/
│
├── app.py                  # Streamlit app
├── Dockerfile              # Container config
├── requirements.txt        # Python dependencies
│
├── api/
│   ├── main.py             # FastAPI entry point
│   └── schema/             # Request/response models
│
├── data/                   # Raw and processed datasets
│
├── model/                  # ML model files
│   └── model_pipeline.pkl  # Trained model pipeline
│
├── notebooks/              # Jupyter notebooks (EDA, training)
│
├── reports/                # Visualizations and analysis reports
│
├── utils/                  # Helper modules for text preprocessing, prediction, etc.
└── README.md               # You're here
```

---

## Getting Started

### Prerequisites

* Python 3.8+
* pip or conda

### Setup Instructions

1. **Clone the Repo**

```bash
git clone https://github.com/regmi-keshav/misinfo-detector.git
cd misinfo-detector
```

2. **Install Dependencies**

```bash
pip install -r requirements.txt
```

3. **Run the Web App**

```bash
streamlit run app.py
```

---

## Docker Option

Prefer containers?

```bash
docker pull keshavregmi/misinfo-detector:latest
docker run -p 8000:8000 keshavregmi/misinfo-detector:latest
```

API will be available at `http://localhost:8000`.

---

## Important Notes

This project:

* **Does not guarantee truth detection**—it assesses *linguistic credibility*
* Should be used to **support human judgment**, not replace it
* Requires ongoing dataset updates for improved reliability

---

## How to Contribute

We welcome contributions that:

* Improve model accuracy
* Add new visualizations or frontend features
* Enhance multilingual support
* Improve documentation or testing

Submit a pull request or open an issue to get started!

---

## License

Distributed under the MIT License.
See [`LICENSE`](./LICENSE) for more info.

---
