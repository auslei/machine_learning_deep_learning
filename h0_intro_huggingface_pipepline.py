#%%
from transformers import pipeline
import pandas as pd

# %%
classifier = pipeline('text-classification')

text = """After 6 months my daughter was complaining how the left airpod volume was considerably lower than right. That bad that she reverted back to the wired ones.

Amazon were great in organising replacement. Issues arrised when I received an email suggesting I sent back the wrong airpods, serial numbers not matching. I contacted Apple, and they advised the case and airpods serial numbers should all match. They did advise possibly my daughter mixed up her case with a friends.

This got me thinking, I checked the serial numbers of the replacement, the only serial number that matched that of the box was the case. So, either non genuine, or refurbished.

I suggest purchasing from JBHIFI, Officeworks or the Goodguys. I ended up purchasing from the Goodguys and they price matched Officeworks special $189.00.

If you do go down the path of purchasing from Amazon store, I encourage you to video the unboxing and serial numbers for evidence.
"""

outputs = classifier(text)
pd.DataFrame(outputs)
# %% Named Entity Recognition
ner_tagger = pipeline("ner", aggregation_strategy = "simple", model="dbmdz/bert-large-cased-finetuned-conll03-english")
outputs = ner_tagger(text)
pd.DataFrame(outputs)
# %% Q&A
reader = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
question = "What does the customer want?"
outputs = reader(question = question, context = text)
pd.DataFrame([outputs])
# %% Summarisation
summariser = pipeline("summarization", model = "sshleifer/distilbart-cnn-12-6")
outputs = summariser(text, max_length = 60, clean_up_tokenization_spaces = True)
print(outputs[0]['summary_text'])

# %% Translation
from transformers import AutoModelWithLMHead, AutoTokenizer,pipeline

mode_name = 'liam168/trans-opus-mt-en-zh'
model = AutoModelWithLMHead.from_pretrained(mode_name)
tokenizer = AutoTokenizer.from_pretrained(mode_name)

translation = pipeline("translation_en_to_zh", model=model, tokenizer=tokenizer)
translation(text)

# %% Text Generation
generator = pipeline("text-generation", model = "gpt2")
response = "Dear customer, sorry to hear about your issue."
prompt = text + "\n\nCustomer Service Response:\n" + response
outputs= generator(prompt, max_length = 2000)
print(outputs[0]['generated_text'])

   # %%
