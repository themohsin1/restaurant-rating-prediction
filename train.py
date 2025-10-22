import pandas as pd
from sklearn.model_selection import train_test_split
from src.features import FeatureBuilder
from src.model import train_model, evaluate_model
from src.io import load_data, save_model

# 1. Load data
df = load_data("data/zomato.csv")

# 2. Clean and preprocess
df = df.dropna(subset=["rate"])

# Remove unwanted entries like 'NEW' or '-'
df = df[~df["rate"].isin(["NEW", "-", ""])]

# Extract only the numeric part before '/5'
df["rate"] = df["rate"].astype(str).str.split("/").str[0]

# Convert to float safely
df["rate"] = df["rate"].astype(float)


# 3. Split
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# 4. Feature engineering
builder = FeatureBuilder(topk=30).fit(train_df)
X_train = builder.transform(train_df)
X_test  = builder.transform(test_df)
y_train, y_test = train_df["rate"], test_df["rate"]

# 5. Train model
model = train_model(X_train, y_train)

# 6. Evaluate
rmse, r2 = evaluate_model(model, X_test, y_test)
print(f"Model Evaluation: RMSE={rmse:.3f}, R2={r2:.3f}")

# 7. Save model pipeline
save_model({"builder": builder, "model": model}, "models/restaurant_pipeline.pkl")
print("✅ Model saved successfully!")

