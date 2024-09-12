from datetime import datetime, timedelta
import re
import json
from langchain.schema import Document
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
import os 
from tqdm import tqdm 
from openai import OpenAI
from langchain_community.vectorstores.utils import DistanceStrategy
import numpy as np 
from sklearn.metrics.pairwise import cosine_similarity
os.environ["OPENAI_API_KEY"] = ""  # 여기에 실제 OpenAI API 키를 입력하세요.


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


# MSC 데이터셋 시간 형식 전처리 
def calculate_new_time(time_back):
    # 기준 시간 설정
    current_time = datetime.strptime("2020/6/20/20:00", "%Y/%m/%d/%H:%M")
    days = 0
    hours = 0

    # "3 days ago" 또는 "6 hours ago" 같은 형식을 처리하기 위한 정규 표현식
    match = re.search(r'(\d+)\s*days?', time_back)
    if match:
        days = int(match.group(1))

    match = re.search(r'(\d+)\s*hours?', time_back)
    if match:
        hours = int(match.group(1))

    time_difference = timedelta(days=days, hours=hours)
    new_time = current_time - time_difference
    return new_time.strftime("%Y/%m/%d/%H:%M")

# dialogs -> 벡터 DB 저장 단위 리스트로 변경
def process_dialogs(json_file_path, n):
    # JSON 파일 읽기
    with open(json_file_path, 'r') as f: 
        data = json.load(f)

    # previous 대화와 현재 대화 다르게 처리 -> 마지막 세션 json을 가져왔기 때문
    previous_dialogs = data.get('previous_dialogs', None)
    current_dialogs = data.get('dialog', None)

    dialog_strings = []
    
    # previous 대화 처리
    for session_idx in range(len(previous_dialogs)):
        time_back = previous_dialogs[session_idx]['time_back']
        session_time = calculate_new_time(time_back)
        
        session_dialogs = []
        for i, p_dialog in enumerate(previous_dialogs[session_idx]['dialog']):
            if i % 2 == 0: 
                session_dialogs.append(f"Speaker1: {p_dialog['text']}")
            else:
                session_dialogs.append(f"Speaker2: {p_dialog['text']}")

        # 세션별 대화 n개씩 묶기 
        for j in range(0, len(session_dialogs),n):
            grouped_dialog = session_dialogs[j:j+n]
            dialog_string = f"time: {session_time}\n" + "\n".join(grouped_dialog) + "\n"
            dialog_strings.append(dialog_string)

    # 현재 대화 처리
    session_dialogs = []
    for c_dialog in current_dialogs:
        speaker = "Speaker1" if c_dialog["id"] == "Speaker1" else "Speaker2"
        session_dialogs.append(f"{speaker}: {c_dialog['text']}")

    # n개씩 묶어서 저장
    for j in range(0, len(session_dialogs), n):
        grouped_dialog = session_dialogs[j:j+n]
        dialog_string = f"time: 2020/6/20/20:00\n" + "\n".join(grouped_dialog) + "\n"
        dialog_strings.append(dialog_string)

    return dialog_strings

def create_single_chunk(text):
    client = OpenAI(api_key="")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""
                The following Provided dialogue content contains a conversation where Speaker 1 and Speaker 2 alternate speaking.

                Convert Provided dialogue content into the following format and summarize the conversation concisely while including key information:

                [time-stamp][concise summary]

                Please ensure the summary is no longer than 1-2 sentences.
                
                text example)
                [{{“text”: “I need some advice on where to go on vacation, have you been anywhere lately?”}},
                {{“text”: “I have been all over the world. I’m military.”}},
                {{“time”: “2020/8/12”}}]
                                

                Output example)
                [2020/8/12] Speaker 1 asks for vacation advice, and Speaker 2 mentions they have traveled extensively due to being in the military.

                Provided dialogue content:
                {text}

                Output:
                
                """
            },
        ],
        temperature=0,
    )
    return response.choices[0].message.content

# a와 b는 idx와 ref의 가중치
def create_multiple_chunks(dialogs, a, b):
    global embeddings
    chunks = []
    # 각 dialog 처리하고 meta data와 함께 저장하여 chunk 반환
    for idx, dialog in enumerate(tqdm(dialogs, desc="Creating Chunks")):
        chunk = create_single_chunk(dialog)
        ref = 0 
        score = a * idx + b * ref 
        chunks.append(
            Document(
                page_content=chunk,
                
                metadata={"dialog": dialog,"ids":idx, "ref": ref, "score": score}
            )
        )
    return chunks

def create_vector_db(chunks):
    db = FAISS.from_documents(chunks, embedding=embeddings, distance_strategy = DistanceStrategy.COSINE)
    return db 

def get_retriver(chunks, db):
    bm25_retriever = BM25Retriever.from_documents(chunks)
    faiss_retriever = db.as_retriever()
    
    retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, faiss_retriever],
        weights=[0.7, 0.3],
        search_type="mmr",
    )
    return retriever
    
def search_documents(retriever, query, num_results=5):
    results = retriever.get_relevant_documents(query, k=num_results)
    return results

def compute_similarity(embedding1, embedding2):
    """
    두 벡터 간의 코사인 유사도를 계산하는 함수
    """
    embedding1 = np.array(embedding1).reshape(1, -1)
    embedding2 = np.array(embedding2).reshape(1, -1)
    return cosine_similarity(embedding1, embedding2)[0][0]

def search_documents_with_similarity(retriever, query, embeddings, num_results=5):
    """
    검색 결과와 함께 similarity 점수를 계산하고 출력하는 함수
    """
    # 1. 쿼리 벡터 임베딩 계산
    query_embedding = embeddings.embed_query(query)

    # 2. 검색 결과 가져오기
    results = retriever.get_relevant_documents(query, k=num_results)
    
    # 3. 각 결과에 대해 유사도 계산 및 출력
    for idx, result in enumerate(results):
        # 문서의 벡터 임베딩 계산
        doc_embedding = embeddings.embed_query(result.page_content)
        
        # 쿼리 임베딩과 문서 임베딩 간의 코사인 유사도 계산
        similarity_score = compute_similarity(query_embedding, doc_embedding)
        
        # 결과 출력
        print(f"Document {idx+1}:")
        print(f"Content: {result.page_content}")
        print(f"Similarity: {similarity_score}\n")
    
    return results




# 벡터 DB에 저장된 벡터의 개수를 반환하는 함수
def get_vectordb_size(db):
    return db.index.ntotal

def pop_cache(cache_db):
    
    vector_count = get_vectordb_size(cache_db)
    # 모든 문서 가져오기 
    docs = cache_db.similarity_search(query="", k=vector_count)
    # 가장 score가 낮은 문서 반환 
    lowest_score_doc = min(docs, key=lambda doc: doc.metadata.get('score',float('inf')))
    
    # 가장 낮은 score 문서 삭제 
    remaining_docs = [doc for doc in docs if doc != lowest_score_doc]
    
    cache_db = FAISS.from_documents(remaining_docs, embedding=embeddings, distance_strategy = DistanceStrategy.COSINE)
    
    return lowest_score_doc, cache_db

def append_long_term(longterm_db, new_document, embeddings):
    
    # 새로운 벡터를 Longterm에 추가
    longterm_db.add_texts([new_document.page_content], [new_document.metadata], embeddings)
    return longterm_db


