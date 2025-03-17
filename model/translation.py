from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import model.mini_lm as mini_lm
import random
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 모델, 토크나이저 로드
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

print(f"✅ 번역 모델 로드 완료. ({device.type.upper()} 사용 중)")

# 기존 번역된 문장 리스트
translated_diary = []

# 랜덤으로 문단 3개 추출
def get_random_samples(data_list, num_samples=3):
    return random.sample(data_list, min(num_samples, len(data_list)))

# 한글 -> 영어 번역 함수
def translate_korean_to_english(text):
    inputs = tokenizer(text, return_tensors="pt") 
    inputs = {key: value.to(device) for key, value in inputs.items()}

    forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    translated_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=100)

    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
    return translated_text

# ✅ 번역 데이터 최신화 함수 추가
def update_translated_diary():
    global translated_diary
    print("✅ DEBUG: update_translated_diary() 실행됨")

    if not mini_lm.result:
        print("❌ ERROR: mini_lm.result가 비어 있음, save_text가 실행되었는지 확인 필요")
        translated_diary = []
        return

    # 문단 3개 선택 후 번역
    random_korean_samples = get_random_samples(mini_lm.result, 3)
    translated_diary = [translate_korean_to_english(sentence) for sentence in random_korean_samples]

    print("✅ DEBUG: 번역된 문장 업데이트 완료:", translated_diary)

# ✅ `save_text`가 호출되었을 때 자동으로 번역 데이터 업데이트
update_translated_diary()

print("✅ 초기 번역 완료:", translated_diary)
