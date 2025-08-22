import pandas as pd
import logging
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from datasets import Dataset
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
logger.info("Preparing domain-specific dataset for Market Pulse...")
data_samples = []
price_points = [10, 15, 20, 25, 30]
waste_reduction = [1, 2, 3, 4, 5]
recipes = ["Vada Pav", "Bhel", "Pani Puri", "Idli", "Pakora"]
hygiene_tips = ["wash hands regularly", "use gloves", "clean utensils daily"]
sustainability_tips = ["use biodegradable plates", "reduce plastic usage"]
for i in range(8000):
    recipe = recipes[i % len(recipes)]
    price = price_points[i % len(price_points)]
    waste = waste_reduction[i % len(waste_reduction)]
    hygiene = hygiene_tips[i % len(hygiene_tips)]
    sustain = sustainability_tips[i % len(sustainability_tips)]
    text = (
        f"Street food item {recipe} is sold at Rs.{price}. "
        f"Waste generated: {waste} kg daily. "
        f"Hygiene tip: {hygiene}. "
        f"Sustainability advice: {sustain}."
    )
    data_samples.append(text)
df = pd.DataFrame(data_samples, columns=["text"])
logger.info("\n✅ Sample Data:\n%s", df.head())
dataset = Dataset.from_pandas(df)
model_name = "prajjwal1/bert-tiny"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name).to(device)
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)
logger.info("\nTokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True)
cols_to_remove = [col for col in ["text", "__index_level_0__"] if col in tokenized_dataset.column_names]
if cols_to_remove:
    lm_dataset = tokenized_dataset.remove_columns(cols_to_remove)
else:
    lm_dataset = tokenized_dataset
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
training_args = TrainingArguments(
    output_dir="./marketpulse_model",
    overwrite_output_dir=True,
    num_train_epochs=1,
    per_device_train_batch_size=8,
    save_total_limit=1,
    prediction_loss_only=True,
    logging_steps=100
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=lm_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)
logger.info("\nStarting training...")
trainer.train()
trainer.save_model("./marketpulse_model")
logger.info("✅ Model saved at ./marketpulse_model")
