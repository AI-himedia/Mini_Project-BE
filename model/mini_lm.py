from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
import main
import requests
from fastapi import FastAPI
from pydantic import BaseModel


app = FastAPI()

text = None

class TextInput(BaseModel):
    text: str

@app.post("/save_text")
def save_text(data: TextInput):
    global text
    text = data.text  # 클라이언트에서 받은 텍스트 저장
    return {"message": "Text saved successfully", "text": text}

# 저장된 텍스트 확인 API (디버깅용)
@app.get("/get_text")
def get_saved_text():
    return {"text": text}


# 문단 분리 모델 로딩
def load_embedding_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 사용 여부 확인
    model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to(device)
    print(f"문단 분리 모델 로드 완료. ({device.upper()} 사용 중)")
    return model, device

embedding_model, device = load_embedding_model()


# 구두점으로 1차 분리
def split_into_sentences(text):
   return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text.strip()) if s]


# 문맥 기반 2차 문단 분리
def context_based_paragraph_split(text, embedding_model, threshold=0.7, max_sentences_per_paragraph=3):
    sentences = split_into_sentences(text)
    
    if len(sentences) < max_sentences_per_paragraph:
        return [text]

    # 문장 임베딩 생성
    embeddings = embedding_model.encode(sentences, convert_to_tensor=True).to(device)

    # 문장 간 유사도 계산
    similarity_matrix = cosine_similarity(embeddings.cpu())  # GPU 사용 시 `.cpu()` 변환 필요

    
    # 문단 분리 기준 (문장 간 유사도가 특정 임계값 이하이면 분리)
    paragraph_indices = [0]
    
    for i in range(1, len(sentences)):
        if similarity_matrix[i-1, i] < threshold or (i - paragraph_indices[-1] >= max_sentences_per_paragraph):
            paragraph_indices.append(i)  # 문단 시작점 추가


    # 특정 패턴(접속사 등)이 나오면 문단을 추가로 분리
    split_keywords = {"하지만", "그런데", "그래서", "그리고", "결국", "즉", "따라서", "왜냐하면", "그러다가", "때문에",
                      "오늘", "어제", "아침에", "점심에", "저녁에", "밤에", "새벽에", "오전", "오후"}

    paragraph_indices.extend(i for i, s in enumerate(sentences) if any(word in s for word in split_keywords))
    paragraph_indices = sorted(set(paragraph_indices))  # 중복 제거

    # 결과 
    paragraphs = [" ".join(sentences[start:end]) for start, end in zip(paragraph_indices, paragraph_indices[1:] + [len(sentences)])]

    return paragraphs


result = context_based_paragraph_split(text, embedding_model, max_sentences_per_paragraph=3)
