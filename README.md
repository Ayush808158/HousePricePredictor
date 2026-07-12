# 🏠 House Price Predictor

A Linear Regression model that predicts house prices (INR) based on
property features such as area, bedrooms, bathrooms, age, distance to
city center, amenities, location and furnishing status — served through
a Streamlit web app.

---

## 📁 Project Structure

```
house-price-predictor/
├── app.py                    # Streamlit web app
├── model.pkl                 # Trained model (generate this, see below)
├── feature_columns.json      # Exact feature/column order used in training
├── categories.json           # Valid Location & Furnished values
├── requirements.txt          # Python dependencies
├── house_price_data.csv      # Training dataset
├── Untitled.ipynb            # Training notebook (EDA + model training)
└── README.md
```

> **Note:** `model.pkl`, `feature_columns.json` and `categories.json` are
> not included by default — you generate them by running the notebook
> once (instructions below). They are ignored by `.gitignore`-style setups
> in many projects, but for a simple deploy you can just commit them.

---

## ⚙️ 1. Generate the model files

Open `Untitled.ipynb`, run all cells up to and including model training
(the cell with `model.fit(x_train, y_train)`), then add a **new cell** at
the end with the contents of `save_model_snippet.py` and run it.

This will create three files in the notebook's folder:

| File | Purpose |
|---|---|
| `model.pkl` | The trained `LinearRegression` model |
| `feature_columns.json` | Exact column order the model expects |
| `categories.json` | Valid dropdown values for Location & Furnished |

Copy these three files into the same folder as `app.py`.

---

## 💻 2. Run locally

```bash
# 1. Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## ☁️ 3. Deploy for free (Streamlit Community Cloud)

1. Create a GitHub repository and push these files to it:
   ```
   app.py
   model.pkl
   feature_columns.json
   categories.json
   requirements.txt
   ```
2. Go to **[share.streamlit.io](https://share.streamlit.io)** and sign in
   with GitHub.
3. Click **"New app"** → select your repository, branch, and set
   **Main file path** to `app.py`.
4. Click **Deploy**. In a minute or two you'll get a public URL like:
   ```
   https://your-app-name.streamlit.app
   ```

That's it — no servers to manage, and it's free.

---

## 🖥️ Alternative: Deploy as an API (Flask/FastAPI + Render/Railway)

If you'd rather expose this as a REST API instead of a UI (e.g., to call
from another app), the same `model.pkl` + `feature_columns.json` can be
wrapped in a small Flask/FastAPI service and deployed on
[Render](https://render.com) or [Railway](https://railway.app) — happy to
build that version too if you need it.

---

## 🧠 Model Details

- **Algorithm:** Linear Regression (`sklearn.linear_model.LinearRegression`)
- **Target:** `Price_INR`
- **Features:** Area, Bedrooms, Bathrooms, Age, Distance to city center,
  Garden, Swimming Pool, Security System, Gym, Location (one-hot),
  Furnishing status (one-hot)
- **Preprocessing:** Duplicate removal, one-hot encoding for categorical
  columns (`Location`, `Furnished`), `Garage` dropped due to low
  correlation with price

Evaluation metrics (from the notebook, on the held-out test split):
MAE, MSE, RMSE and R² — see the last cell of `Untitled.ipynb` for the
exact values from your training run (they'll vary slightly each run
since the train/test split isn't seeded).

> 💡 **Tip:** For reproducible results, consider adding
> `random_state=42` to `train_test_split(...)` in the notebook.

---

## 📄 License

This project is provided as-is for educational/portfolio purposes.
