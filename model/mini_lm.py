# model/mini_lm.py

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
from fastapi import APIRouter
from pydantic import BaseModel
import torch.nn.functional as F


lm_router = APIRouter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to(device)

text = ""

class TextInput(BaseModel):
    text: str


# 클라이언트에서 받은 일기 텍스트 저장
@lm_router.post("/save_text")
def save_text(data: TextInput):
    global text
    text = data.text 
    return {"message": "Text saved successfully", "text": text}


# 구두점으로 1차 분리
def split_into_sentences(text):
   return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]


# 문맥 기반 2차 문단 분리
def context_based_paragraph_split(text, embedding_model, threshold=0.7,  window_size=2):
    sentences = split_into_sentences(text)
    
    if len(sentences) < 3:
        return [text]

    embeddings = embedding_model.encode(sentences, convert_to_tensor=True).to(device)

    # 유사도 행렬 계산 (torch 기반)
    similarity_matrix = torch.mm(embeddings, embeddings.T) / (
    torch.norm(embeddings, dim=1, keepdim=True) * torch.norm(embeddings.T, dim=0, keepdim=True)
    )
    similarity_matrix = similarity_matrix.cpu().numpy()


    # 문단 시작점 설정
    paragraph_indices = [0]
    
    for i in range(1, len(sentences)):
        # 최근 `window_size`개의 문장과 평균 유사도를 계산하여 임계값 이하인지 확인
        avg_similarity = sum(similarity_matrix[i, max(0, i - window_size):i]) / window_size

        if avg_similarity < threshold or (i - paragraph_indices[-1] >= 3):
            paragraph_indices.append(i)

    # 접속사 및 시간 표현 기반 추가 문단 분리
    split_keywords = {"하지만", "그런데", "그래서", "그리고", "결국", "즉", "따라서", "왜냐하면", "그러다가", "때문에",
                      "오늘", "어제", "아침에", "점심에", "저녁에", "밤에", "새벽에", "오전", "오후"}
    
    paragraph_indices.extend(i for i, s in enumerate(sentences) if any(word in s for word in split_keywords))
    paragraph_indices = sorted(set(paragraph_indices))  # 중복 제거

    # 결과 
    paragraphs = [" ".join(sentences[start:end]) for start, end in zip(paragraph_indices, paragraph_indices[1:] + [len(sentences)])]

    return paragraphs


result = context_based_paragraph_split(text, embedding_model)
