from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
from fastapi import APIRouter
from pydantic import BaseModel


lm_router = APIRouter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to(device)

text = "비행기는 부드럽게 웅웅거렸고, 창밖으로 끝없이 펼쳐진 구름을 바라보았다. 여행을 할 때마다 항상 설렘과 긴장감이 동시에 밀려온다. 팜스프링스에 도착하자 황금빛 사막이 나를 반겨주었다. 공기는 건조했지만 상쾌했고, 야자수를 보는 순간 마치 엽서 속으로 들어온 것 같았다. 빨리 이곳을 탐험하고 싶다. 어쩌면 조용한 곳을 찾아 앉아 이 순간을 온전히 느껴봐야겠다."

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
def context_based_paragraph_split(text, embedding_model, threshold=0.7):
    sentences = split_into_sentences(text)
    
    if len(sentences) < 3:
        return [text]

    # 문장 임베딩 생성
    embeddings = embedding_model.encode(sentences, convert_to_tensor=True).to(device)

    # 문장 간 유사도 계산
    embeddings = embedding_model.encode(sentences, convert_to_tensor=True).to(device)
    similarity_matrix = cosine_similarity(embeddings.cpu())


    # 문단 분리 기준 (문장 간 유사도가 특정 임계값 이하이면 분리)
    paragraph_indices = [0]
    
    for i in range(1, len(sentences)):
        if similarity_matrix[i-1, i] < threshold or (i - paragraph_indices[-1] >= 3):
            paragraph_indices.append(i)  # 문단 시작점 추가


    # 특정 패턴(접속사 등)이 나오면 문단을 추가로 분리
    split_keywords = {"하지만", "그런데", "그래서", "그리고", "결국", "즉", "따라서", "왜냐하면", "그러다가", "때문에",
                      "오늘", "어제", "아침에", "점심에", "저녁에", "밤에", "새벽에", "오전", "오후"}

    paragraph_indices.extend(i for i, s in enumerate(sentences) if any(word in s for word in split_keywords))
    paragraph_indices = sorted(set(paragraph_indices))  # 중복 제거

    # 결과 
    paragraphs = [" ".join(sentences[start:end]) for start, end in zip(paragraph_indices, paragraph_indices[1:] + [len(sentences)])]

    return paragraphs


result = context_based_paragraph_split(text, embedding_model)
