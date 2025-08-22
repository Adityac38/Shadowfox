import os
import random
import math
import logging
from datetime import timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    pipeline,
)
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(1)
MODEL_NAME = "prajjwal1/bert-tiny"
MAX_LEN = 64
MLM_EPOCHS = 1
MLM_BATCH_SIZE = 8
SAMPLE_DAYS = 220
SAVE_DIR = "./models_market_pulse"
os.makedirs(SAVE_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("market_pulse")
def create_synthetic_market_data():
    logger.info("Creating synthetic market dataset...")
    products = ["Potato", "Onion", "Tomato", "Flour", "Mustard Oil"]
    suppliers = ["FreshMart", "AgroBest", "GreenLeaf", "Farm2You"]
    today = pd.to_datetime("2025-08-20")
    rows = []
    for p in products:
        base = {
            "Potato": 40,
            "Onion": 35,
            "Tomato": 30,
            "Flour": 50,
            "Mustard Oil": 160
        }.get(p, 50)
        seasonal_amp = {
            "Potato": 6,
            "Onion": 5,
            "Tomato": 8,
            "Flour": 3,
            "Mustard Oil": 12
        }.get(p, 5)
        trend = {
            "Potato": 0.01,
            "Onion": 0.02,
            "Tomato": 0.015,
            "Flour": 0.005,
            "Mustard Oil": 0.02
        }.get(p, 0.01)
        for day in range(SAMPLE_DAYS):
            date = today - pd.Timedelta(days=SAMPLE_DAYS - day)
            price = base + seasonal_amp * math.sin(2 * math.pi * (day) / 30) + trend * day + np.random.normal(0, 1.5)
            price = round(max(price, 5.0), 2)
            supplier = random.choice(suppliers)
            demand_change = int(round( (5 * math.sin(2 * math.pi * (day) / 14)) + np.random.normal(0,6) ))
            waste_income = int(max(0, np.random.normal(1200, 400)))
            rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "product": p,
                "price": price,
                "supplier": supplier,
                "demand_change_pct": demand_change,
                "waste_monetization": waste_income
            })
    df = pd.DataFrame(rows)
    logger.info("Synthetic data size: %d rows", len(df))
    return df
def structured_to_text(df):
    texts = []
    for _, r in df.iterrows():
        demand_sign = "+" if r["demand_change_pct"] >= 0 else ""
        txt = (f"Date: {r['date']}. Product: {r['product']}. Price: ₹{r['price']}/kg. "
               f"Supplier: {r['supplier']}. Demand change: {demand_sign}{r['demand_change_pct']}%. "
               f"Waste income: ₹{r['waste_monetization']}.")
        texts.append(txt)
    return pd.DataFrame({"text": texts})
def fine_tune_bert_mlm(text_df):
    logger.info("Preparing dataset for MLM...")
    ds = Dataset.from_pandas(text_df)
    logger.info("Loading tokenizer and model: %s", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))
    def tok_fn(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=MAX_LEN, return_special_tokens_mask=True)
    tokenized = ds.map(tok_fn, batched=True)
    cols_to_delete = [c for c in ["text", "__index_level_0__"] if c in tokenized.column_names]
    if cols_to_delete:
        tokenized = tokenized.remove_columns(cols_to_delete)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    import inspect
    train_kwargs = dict(
        output_dir=os.path.join(SAVE_DIR, "mlm_results"),
        num_train_epochs=MLM_EPOCHS,
        per_device_train_batch_size=MLM_BATCH_SIZE,
        save_total_limit=1,
        logging_steps=200,
    )
    try:
        sig = inspect.signature(TrainingArguments.__init__)
        supported = set(sig.parameters.keys())
        filtered = {k: v for k, v in train_kwargs.items() if k in supported}
        training_args = TrainingArguments(**filtered)
    except Exception:
        training_args = TrainingArguments(output_dir=train_kwargs["output_dir"], num_train_epochs=MLM_EPOCHS, per_device_train_batch_size=MLM_BATCH_SIZE)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    logger.info("Starting MLM training (CPU). This may take a few minutes...")
    try:
        trainer.train()
    except Exception as e:
        logger.exception("Training failed (exception): %s", e)
    out = os.path.join(SAVE_DIR, "bert_domain")
    os.makedirs(out, exist_ok=True)
    trainer.save_model(out)
    tokenizer.save_pretrained(out)
    logger.info("Saved domain-adapted BERT to %s", out)
    return out, tokenizer
def train_price_predictor(df):
    logger.info("Preparing features for price predictor...")
    df_sorted = df.sort_values(["product", "date"]).reset_index(drop=True)
    df_sorted["date_dt"] = pd.to_datetime(df_sorted["date"])
    df_sorted = df_sorted.sort_values(["product", "date_dt"])
    df_sorted["price_prev"] = df_sorted.groupby("product")["price"].shift(1)
    df_feat = df_sorted.dropna(subset=["price_prev"]).copy()
    le_prod = LabelEncoder().fit(df_feat["product"])
    le_sup = LabelEncoder().fit(df_feat["supplier"])
    df_feat["product_id"] = le_prod.transform(df_feat["product"])
    df_feat["supplier_id"] = le_sup.transform(df_feat["supplier"])
    start = df_feat["date_dt"].min()
    df_feat["days_since_start"] = (df_feat["date_dt"] - start).dt.days
    df_feat["day_of_year"] = df_feat["date_dt"].dt.dayofyear
    feature_cols = ["price_prev", "product_id", "supplier_id", "days_since_start", "day_of_year", "demand_change_pct", "waste_monetization"]
    X = df_feat[feature_cols].values
    y = df_feat["price"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.12, random_state=SEED)
    logger.info("Training RandomForestRegressor (fast, CPU-friendly)...")
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    logger.info("RandomForest MAE: %.3f (₹ units)", mae)
    import joblib
    joblib.dump({
        "model": rf,
        "le_prod": le_prod,
        "le_sup": le_sup,
        "feature_cols": feature_cols,
        "start_date": start
    }, os.path.join(SAVE_DIR, "price_predictor.joblib"))
    logger.info("Price predictor saved.")
    return rf, le_prod, le_sup, feature_cols, start
def predict_next_price_for_product(df, rf_dict, product_name, supplier_preference=None):
    """
    Compute a next-day price prediction for `product_name`.
    supplier_preference: if provided, choose that supplier. Otherwise pick most recent supplier for that product.
    """
    rf = rf_dict["model"]
    le_prod = rf_dict["le_prod"]
    le_sup = rf_dict["le_sup"]
    feature_cols = rf_dict["feature_cols"]
    start = rf_dict["start_date"]
    dfp = df[df["product"] == product_name].sort_values("date_dt")
    if dfp.empty:
        raise ValueError("Unknown product: " + product_name)
    last = dfp.iloc[-1]
    price_prev = last["price"]
    days_since_start = (last["date_dt"] - start).days + 1
    day_of_year = (last["date_dt"] + pd.Timedelta(days=1)).dayofyear
    if supplier_preference and supplier_preference in le_sup.classes_:
        supplier_id = int(np.where(le_sup.classes_ == supplier_preference)[0][0])
    else:
        supplier_name = last["supplier"]
        supplier_id = int(le_sup.transform([supplier_name])[0])
    demand_change = int(last["demand_change_pct"])
    waste = int(last["waste_monetization"])
    x = np.array([[price_prev, int(le_prod.transform([product_name])[0]), supplier_id, days_since_start, day_of_year, demand_change, waste]])
    pred = rf.predict(x)[0]
    return round(pred, 2)
def recommend_supplier(df, product_name, top_n=2):
    """
    Simple supplier recommendation: return the suppliers with lowest average price for the product.
    """
    subset = df[df["product"] == product_name]
    if subset.empty:
        return []
    agg = subset.groupby("supplier")["price"].mean().reset_index().sort_values("price")
    return agg.head(top_n).to_dict(orient="records")
def plot_price_trend(df, product_name, show_last_days=40, predicted_next=None):
    subset = df[df["product"] == product_name].sort_values("date_dt")
    if subset.empty:
        print("No data for", product_name); return
    to_plot = subset.tail(show_last_days)
    plt.figure(figsize=(8,4))
    plt.plot(pd.to_datetime(to_plot["date"]), to_plot["price"], marker="o", label="Historical price")
    if predicted_next is not None:
        next_date = pd.to_datetime(to_plot["date"].iloc[-1]) + pd.Timedelta(days=1)
        plt.scatter([next_date], [predicted_next], color="red", label="Predicted next-day price", zorder=5)
    plt.title(f"Price trend (last {show_last_days} days): {product_name}")
    plt.xlabel("Date")
    plt.ylabel("Price (₹)")
    plt.xticks(rotation=30)
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.show()
def generate_insight_mask(model_dir, tokenizer, prompt_with_mask, top_k=5):
    """
    Use fine-tuned fill-mask pipeline for short domain prompts.
    prompt_with_mask must contain tokenizer.mask_token (e.g., [MASK]) at least once.
    """
    try:
        fill = pipeline("fill-mask", model=model_dir, tokenizer=tokenizer, device=-1)  # CPU
        preds = fill(prompt_with_mask, top_k=top_k)
        return preds
    except Exception as e:
        logger.exception("fill-mask pipeline failed: %s", e)
        return []
def main():
    df_struct = create_synthetic_market_data()
    text_df = structured_to_text(df_struct)
    text_df = text_df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    text_df = text_df.head(3000)
    model_dir, tokenizer = fine_tune_bert_mlm(text_df)
    df_struct["date_dt"] = pd.to_datetime(df_struct["date"])
    df_struct = df_struct.sort_values(["product", "date_dt"]).reset_index(drop=True)
    rf_model, le_prod, le_sup, feat_cols, start = train_price_predictor(df_struct)
    import joblib
    rf_dict = joblib.load(os.path.join(SAVE_DIR, "price_predictor.joblib"))
    product = "Potato"
    print("\n--- Dashboard demo for product:", product, "---")
    next_pred = predict_next_price_for_product(df_struct, rf_dict, product)
    print(f"Predicted next-day price for {product}: ₹{next_pred}/kg")
    recs = recommend_supplier(df_struct, product, top_n=3)
    print("Top supplier recommendations (by avg price):")
    for r in recs:
        print("  -", r["supplier"], " avg price ₹", round(r["price"],2))
    plot_price_trend(df_struct, product, show_last_days=40, predicted_next=next_pred)
    mask_examples = [
        f"Top supplier for {product} is {tokenizer.mask_token}.",
        f"Trending product this week is {tokenizer.mask_token}.",
        f"Waste income can be ₹{tokenizer.mask_token} per month."
    ]
    for ex in mask_examples:
        print("\nPrompt:", ex)
        preds = generate_insight_mask(model_dir, tokenizer, ex, top_k=5)
        for p in preds[:5]:
            token = p.get("token_str", "")
            seq = p.get("sequence", "")
            score = p.get("score", 0.0)
            print(f"  → {seq}  (token: {repr(token)}, score: {score:.3f})")
    print("\nAll models saved under:", SAVE_DIR)
    print("You can plug predict_next_price_for_product(), recommend_supplier(), generate_insight_mask() into your dashboard.")
if __name__ == "__main__":
    main()