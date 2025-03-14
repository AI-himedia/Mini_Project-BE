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

text = """아침부터 흐린 하늘을 보며 비가 올 것 같은 예감이 들었다. 점심 무렵, 창밖에서 조용히 빗방울이 떨어지기 시작했다.

나는 창문을 열어 촉촉한 공기를 들이마셨다. 비 냄새가 퍼지며 거리의 풍경이 더욱 감성적으로 변했다.

가로등 불빛이 젖은 도로에 반사되어 반짝이고, 사람들이 우산을 쓰고 조용히 걸어가는 모습이 영화 속 한 장면 같았다.

나는 따뜻한 홍차를 한 잔 내렸다. 손끝에서 차의 온기가 전해지며 마음이 차분해졌다.

창밖을 보며 조용한 빗소리를 들었다. 빗방울이 유리창을 타고 흐르는 모습이 마치 작은 강처럼 보였다.

책을 한 장 넘기며, 이 조용한 오후를 즐겼다. 빗소리는 마치 음악처럼 일정한 리듬을 가지고 있었다.

잠시 후, 거리는 어두워졌고, 카페의 네온사인이 더욱 밝게 빛나기 시작했다.

나는 우산을 들고 밖으로 나왔다. 젖은 도로 위에 반짝이는 불빛들이 내 발걸음을 따라 춤을 추는 듯했다.

비 오는 거리를 걷는 이 순간, 세상은 더 조용하고 평온하게 느껴졌다."""

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
