from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch
import re
from fastapi import APIRouter
from pydantic import BaseModel


lm_router = APIRouter()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to(device)

text = """따뜻한 봄날, 나는 벚꽃이 만개한 거리를 걸으며 카페에 들렀다. 거리는 분홍빛 꽃잎이 흩날리며 한 폭의 그림 같았다.

햇살이 부드럽게 스며드는 창가 자리에 앉아 따뜻한 라떼를 주문했다. 커피 향이 코끝을 스치며 기분이 포근해졌다.

창문 너머로 벚꽃나무 아래에서 사진을 찍는 사람들이 보였다. 가벼운 바람이 불 때마다 꽃잎이 흩날려 마치 꿈속 같은 풍경을 만들었다.

카페 안에는 잔잔한 재즈 음악이 흐르고 있었다. 바리스타는 정성스럽게 커피를 내리고, 커피 머신 소리가 공간을 가득 채웠다.

나는 노트북을 꺼내 여행 계획을 정리했다. 가고 싶은 도시, 보고 싶은 풍경을 적으며 설렘을 느꼈다.

한 입 마신 라떼는 부드럽고 고소했다. 벚꽃 향이 가득한 거리에서 마시는 커피는 특별한 여운을 남겼다.

잠시 후, 나는 카페를 나와 벚꽃이 가득한 공원을 걸었다. 벤치에 앉아 하늘을 올려다보니 분홍빛 꽃잎들이 바람에 흩날리고 있었다.

햇살이 따뜻하게 내리쬐며, 벚꽃과 함께 봄을 만끽하는 기분이 들었다. 이 순간이 오래도록 기억에 남을 것 같다."""

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
