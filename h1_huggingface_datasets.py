
#%%
from datasets import list_datasets

all_datasets = list_datasets()
print(f"There are {len(all_datasets)} datasets currentrly available on hub")
print(f"The first 10 are {all_datasets[:10]}")


#%%
# loading a dataset
from datasets import load_dataset
emotions = load_dataset("emotion")

train_ds = emotions['train']

print(train_ds.features)
print(train_ds[0])


#%% datasets to dataframes
import pandas as pd
emotions.set_format(type = 'pandas')

df = emotions['train'][:] #set format visualise the slices in pandas format for the asy of visualisation

df['label_name'] = df['label'].map(lambda x: emotions['train'].features['label'].int2str(x))

df.head(10)

#%% visualise
import matplotlib.pyplot as plt
df['label_name'].value_counts(ascending = True).plot.barh()
plt.title("Frequency of Classes")
plt.show()


#%%
df["words per tweet"] = df["text"].str.split().apply(len)
df.boxplot("words per tweet", by = "label_name", grid = False, showfliers = False)
# %% Rsest Format
emotions.reset_format()


# %% tokenisation
from transformers import AutoTokenizer, DistilBertTokenizer #auto will pickup based on model, but can also specific models

model_ckpt = "distilbert-base-uncased"
tokeniser = AutoTokenizer.from_pretrained(model_ckpt)

tokeniser = DistilBertTokenizer.from_pretrained(model_ckpt)

test = "i love music and it make me feel wonderful. part of NLP?"
encode_text = tokeniser(test)
print(encode_text)

tokens = tokeniser.convert_ids_to_tokens(encode_text.input_ids)
print(tokens)

print(tokeniser.convert_tokens_to_string(tokens))
print(tokeniser.vocab_size)
# %% tokenise over a batch
def tokenise(batch):
    return tokeniser(batch['text'], padding = True, truncation = True)

emotions_encoded = emotions.map(tokenise, batched = True, batch_size = None)

# %% models
from transformers import AutoModel
import torch

model_ckpt = "distilbert-base-uncased"
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
model = AutoModel.from_pretrained(model_ckpt).to(device)

# %%
import torch_directml
dml = torch_directml.device()

text = "this is a test"
inputs = tokeniser(text, return_tensors = "pt") # by default tokeniser returns arrays
inputs = {k: v.to(device) for k, v in inputs.items()}

# %%
with torch.no_grad():
    outputs = model(**inputs)
# %%
