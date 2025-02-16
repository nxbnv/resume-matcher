from transformers import AutoModel

model = AutoModel.from_pretrained("rohan57/mymodel")
print(model)