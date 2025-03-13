from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import model.mini_lm as mini_lm
import random
import torch



device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 모델, 토크나이저 로드
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

print(f"번역 모델 로드 완료. ({device.type.upper()} 사용 중)")


korean_diary = mini_lm.result

# 한글 -> 영어 
def translate_korean_to_english(text):
    inputs = tokenizer(text, return_tensors="pt") 
    inputs = {key: value.to(device) for key, value in inputs.items()}
    
    forced_bos_token_id = tokenizer.convert_tokens_to_ids("eng_Latn")
    translated_tokens = model.generate(**inputs, forced_bos_token_id=forced_bos_token_id, max_length=100)
    
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    
    return translated_text


# 번역 결과 추출
translated_diary = [translate_korean_to_english(sentence) for sentence in korean_diary]

# 랜덤으로 문단 3개 추출
def get_random_samples(data_list, num_samples=3):
    return random.sample(data_list, min(num_samples, len(data_list)))


print(get_random_samples(translated_diary))

result = get_random_samples(translated_diary)
