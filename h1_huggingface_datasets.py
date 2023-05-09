from datasets import list_datasets

all_datasets = list_datasets()
print(f"There are {len(all_datasets)} datasets currentrly available on hub")
print(f"The first 10 are {all_datasets[:10]}")

# loading a dataset

from datasets import load_dataset
