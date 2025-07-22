from src.agent.rag_system import RagSystem, reranker_RagSystem
from src.rag_framework import RagDatabase, HfWrapper, OpenAiWrapper, transpose_json, RagDatabase, transpose_jsonl, text_similarity
from utils import load_json_database, load_parquet_database_as_doc
from sentence_transformers import SentenceTransformer
import FlagEmbedding
import os
import json
from dotenv import load_dotenv
import torch
from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizer, PreTrainedModel
import time
from scipy.stats import pearsonr
import numpy as np
from tqdm import tqdm
from datetime import datetime

def init_rag(dataset_path):
    embedding_model_name = "BAAI/bge-base-en"
    embedding_model = SentenceTransformer(embedding_model_name, device='cuda')
    reranker_model_name = "BAAI/bge-reranker-v2-m3"
    reranker = FlagEmbedding.FlagReranker(reranker_model_name, device='cuda')
    # llm = HfWrapper("Llama-3.1-8B-Instruct")
    load_dotenv()
    llm = OpenAiWrapper(model_name='deepseek-chat', api_url='https://api.deepseek.com/v1', api_key=DEEPSEEK_API)
    
    '''build rag system'''
    defaul_rag_save_name = "rag_database_"+ embedding_model_name.split('/')[-1] + '_' + '.'.join(dataset_path.split('/')[-1].split('.')[:-1])
    
    print("Start building RAG system...\n")
    if not os.path.exists(defaul_rag_save_name):
        rag_dataset = transpose_json(dataset_path, "input", "output")
        rag_database = RagDatabase.from_texts(
            embedding_model,
            rag_dataset["input"],
            {"question": rag_dataset["input"], "answer": rag_dataset["output"]},
            batch_size=4
        )
        rag_database.save(defaul_rag_save_name)
    else:
        rag_database = RagDatabase.load(defaul_rag_save_name, embedding_model)
    
    Rag_system = reranker_RagSystem(rag_database, embedding_model, llm, reranker)
    
    return Rag_system

def main():
    dataset_path = "datasets/HealthCareMagic-100k.json" 
    dataset_label = dataset_path.split('/')[-1].split('.')[-2]
    now = datetime.now() 
    custom_label = now.strftime("%Y-%m-%d %H:%M:%S")
    load_dotenv()
    rag_system = init_rag(dataset_path)
    while True:
        question = input("# Ask something (or type 'exit' to quit): ")
        if question.lower() == "exit":
            print("Exiting...")
            break
        answer, similarity_list, retrieval = rag_system.ask(question, n_retrieval=8, n_rerank=4, return_retrieval=True)
        retrieval_list = [f"Similarity: {x}\nContent:\n{y}" for x, y in zip(similarity_list, retrieval)]
        print("# Answer:\n", answer, "\n")
        print("# Retrieval:\n\n", "\n\n".join(retrieval_list))

if __name__ == "__main__":
   os.environ["CUDA_VISIBLE_DEVICES"] = "3"
   from transformers import logging as hf_logging
   hf_logging.set_verbosity_error()
   import warnings
   warnings.filterwarnings("ignore")
   main()

