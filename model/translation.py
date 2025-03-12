from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import mini_lm
import time
import random
import torch

korean_diary = mini_lm.result

start_time = time.time()

device = "cuda" if torch.cuda.is_available() else "cpu"

# 모델, 토크나이저 로드
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

print(f"번역 모델 로드 완료. ({device.upper()} 사용 중)")


# 한글 -> 영어 
def translate_korean_to_english(text):
    inputs = tokenizer(text, return_tensors="pt").to(device)
    
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    translated_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id)
    
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
    return translated_text


# 번역 결과 추출
translated_diary = [translate_korean_to_english(sentence) for sentence in korean_diary]

# 랜덤으로 문단 3개 추출
def get_random_samples(data_list, num_samples=3):

    if len(data_list) <= num_samples:
        return data_list
    else:
        return random.sample(data_list, num_samples)


end_time = time.time()
elapsed_time = end_time - start_time

print(get_random_samples(translated_diary))

print(f"총 실행 시간: {elapsed_time:.4f} 초")
print(f"평균 실행 시간 (문장당): {elapsed_time / len(korean_diary):.4f} 초")