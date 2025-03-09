Brisk Logo Menu
# 📚 LLM Semantic Book Recommender

## 🚀 Overview
The **LLM Semantic Book Recommender** is a machine learning-powered system that leverages **Large Language Models (LLMs)** and **vector search** to provide personalized book suggestions based on user input. By analyzing book descriptions, the system finds similar books based on semantic meaning and emotional tone.

## ✨ Features
✅ **Semantic Search:** Find books based on content similarity using **OpenAI embeddings**.  
✅ **Emotion-Based Filtering:** Select books based on emotional tones like **happy, sad, suspenseful**.  
✅ **Interactive UI:** A **Gradio-based** web dashboard for seamless user experience.  
✅ **Fast Recommendations:** Efficient vector search using `ChromaDB`.  
✅ **Text Classification:** Extracts **sentiment and emotion** from book descriptions.

---

## 🛠 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/AymenGabsi/semantic-book-recommender.git
cd semantic-book-recommender
```

### 2️⃣ Set Up a Virtual Environment
```bash
python -m venv env
source env/bin/activate  # Windows: env\Scripts ctivate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Usage
### ▶️ Run the Gradio Dashboard
```bash
python gradio-dashboard.py
```
This will launch an **interactive web UI** where you can enter book descriptions and receive personalized recommendations.

### 🔬 Explore Data & Train Models
- 📊 **`data-exploration.ipynb`** – Load & analyze the dataset.  
- 🎭 **`text-classification.ipynb`** – Perform sentiment & emotion analysis.  
- 🧠 **`vector-search.ipynb`** – Generate embeddings & run semantic search.

---

## 🏗 How It Works
🔹 **Text Embedding:** Converts book descriptions into high-dimensional vectors using `OpenAIEmbeddings`.  
🔹 **Vector Search:** Stores & retrieves embeddings using `ChromaDB`.  
🔹 **Filtering:** Sorts recommendations based on **category** & **emotional tone**.  
🔹 **Interactive UI:** Users input book descriptions & receive recommendations instantly.

---

## 🛠 Technologies Used
🧠 **NLP & Machine Learning:** `sentence-transformers`, `torch`, `transformers`  
📊 **Data Processing:** `pandas`, `numpy`  
📌 **Vector Search:** `chromadb`, `langchain`  
🌐 **Web App UI:** `gradio`  
📈 **Visualization:** `matplotlib`, `seaborn`

---

## 🚀 Future Improvements
🚀 **Fine-tune embeddings** for more accurate recommendations.  
🚀 **Deploy online** (Hugging Face Spaces, FastAPI, or Streamlit Cloud).  
🚀 **Expand emotion categories** for better personalization.  

---

## 🤝 Contributing
Feel free to **fork** this repository, submit **pull requests**, and help improve the project! 🎉

## 📜 License
This project is licensed under the **MIT License**.

---

## 👤 Author
🔹 **Aymen Gabsi**  
🔹 **GitHub:** [My Github Profile](https://github.com/AymenGabsi)  

💡 *If you find this project useful, don't forget to give it a ⭐ on GitHub!* 🚀
