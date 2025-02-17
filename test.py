from transformers import BertForSequenceClassification, BertTokenizer
import torch

hf_model_name = "rohan57/mymodel"
tokenizer = BertTokenizer.from_pretrained(hf_model_name)
model = BertForSequenceClassification.from_pretrained(hf_model_name)

test_text = "This is a sample job description."
inputs = tokenizer(test_text, return_tensors="pt", truncation=True, padding=True, max_length=512)

with torch.no_grad():
    outputs = model(**inputs)
    print("🔍 Model Output:", outputs.logits)
