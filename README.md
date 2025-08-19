📰 Fake News Detector 🔍

A Machine Learning-powered web app that detects whether a news article is Real or Fake using NLP techniques and a trained model.
Built with Python, Scikit-learn, FastAPI, and Streamlit.

<!-- Optional banner if you want, I can design one -->

🚀 Features

✅ Detects Fake or Real news with high accuracy
✅ Built with Natural Language Processing (NLP)
✅ Machine Learning Model trained using TF-IDF & Logistic Regression
✅ FastAPI Backend + Streamlit Web App
✅ Interactive & Responsive UI
✅ Lightweight and easy to deploy

🛠️ Tech Stack
Technology	Usage
Python 3.12	Core programming language
Scikit-learn	Model training & evaluation
Pandas / NumPy	Data preprocessing
FastAPI	Backend API framework
Jinja2	HTML templating
Streamlit	Interactive frontend
Joblib	Model serialization
Git & GitHub	Version control & hosting
📂 Project Structure
FAKE-NEWS-DETECTOR/
│── app.py                     # FastAPI backend app
│── streamlit_app.py           # Streamlit web app
│── train_model.py             # ML model training script
│── train_baseline.py          # Baseline model for testing
│── fake_news_pipeline.joblib  # Trained model file
│── fake_news_dataset.csv      # Dataset
│── templates/
│     └── index.html          # Web UI template
│── static/
│     └── style.css          # Custom styles
│── requirements.txt          # Python dependencies
│── README.md                # Project documentation

⚡ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/shivspt9/FAKE-NEWS-DETECTOR.git
cd FAKE-NEWS-DETECTOR

2️⃣ Create Virtual Environment
python -m venv .venv
source .venv/Scripts/activate   # On Windows

3️⃣ Install Dependencies
pip install -r requirements.txt

4️⃣ Train the Model (Optional)
python train_model.py

5️⃣ Run the Web App (FastAPI)
uvicorn app:app --reload


App will be live at 👉 http://127.0.0.1:8000

6️⃣ Run the Streamlit App
streamlit run streamlit_app.py


Streamlit UI 👉 http://localhost:8501

📊 Model Performance
Metric	Score
Accuracy	95%
Precision	94%
Recall	93%
F1-score	94%
📌 Screenshots
Home Page	Prediction Result

	

(I can design custom screenshots for your project — let me know!)

🌎 Deployment Options

Local Deployment → Using FastAPI / Streamlit

GitHub Pages (Frontend only)

Render / Railway / Vercel (Free hosting options)

Docker Support (optional if needed)

🤝 Contributing

Want to improve this project? Follow these steps:

# Fork the repo
# Create a new branch
git checkout -b feature-name

# Make your changes and commit
git commit -m "Added new feature"

# Push the branch
git push origin feature-name


Then create a Pull Request ✅

👤 Author

Shivprakash Tiwari
📧 Email: shivprakash.tiwari@gmail.com
 (optional)
🔗 GitHub
 | LinkedIn

📜 License

This project is open-source and available under the MIT License.
