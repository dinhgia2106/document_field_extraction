import os
import math
from openai import OpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import numpy as np
import pandas as pd

# 1. Load API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not set in .env file")

# 2. Initialize Client
client = OpenAI(api_key=api_key)


def get_embeddings(texts, model="text-embedding-3-small"):
    inputs = [str(text).replace("\n", " ") for text in texts]
    return [i.embedding for i in client.embeddings.create(input=inputs, model=model).data]

embs = []
for i in tqdm(range(0, len(df), 100)):
    embs.extend(get_embeddings(df.body.iloc[i:i + 100]))
embs = np.array(embs)
embs.shape