#%%
import openai
import json

api_key = open("./api.key").read()

openai.api_key = api_key

model = "text-davinci-003"


def generate_text(prompt):
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-0301",
        prompt=prompt,
        max_tokens=1024,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message = response.choices[0].text
    return message.strip()

prompt = "Hello, how are you doing today?"

generated_text = generate_text(prompt)

print(generated_text)
# %%
