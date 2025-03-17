# AI가 그려주는 내 하루

<p align="center">
  <img src="https://aicreation-file.miricanvas.com/private/txt2img/2025/03/16/18/3d5b0fc8-b297-434c-bdf3-4482ad8c582b.jpg?mode=modal" width="500px" />
</p>

## 프로젝트 개요
**AI가 그려주는 내 하루**는 사용자의 일기를 분석하여 텍스트를 요약하고, 각 문단에 어울리는 이미지를 생성하는 프로젝트입니다. OCR(광학 문자 인식) 기술을 활용하여 손글씨 또는 타이핑된 일기 내용을 분석하고, 이를 AI 기반 이미지 생성 모델을 통해 시각적으로 표현합니다.

## 주요 기능
- **OCR을 통한 텍스트 인식**: 사용자가 업로드한 일기의 내용을 디지털 텍스트로 변환합니다.
- **문단별 요약**: 일기를 3개의 핵심 문단으로 분리하여 정리합니다.
- **AI 이미지 생성**: 각 문단의 내용을 반영한 그림을 자동으로 생성합니다.

## 기술 스택
- **프로그래밍 언어**: BE: Python | FE: React.js
- **텍스트 처리**: OCR(무료) or Naver CLOVA OCR(유료)
- **이미지 생성**: thibaud/sdxl_dpo_turbo, etri-vilab/koala-700m-llava-cap, sd-community/sdxl-flash AI Model
- **웹 프레임워크**: FastAPI
- **프론트엔드**: React.js

## 개발팀 소개

### 프론트엔드
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/dijeungi">
        <img src="https://github.com/dijeungi.png" width="100px" style="border-radius: 50%;" />
      </a>
      <br />
      <b>최준호</b>
    </td>
  </tr>
</table>

### 백엔드
<table>
  <tr>
    <td align="center">
      <a href="https://github.com/jinnni91">
        <img src="https://github.com/jinnni91.png" width="100px" style="border-radius: 50%;" />
      </a>
      <br />
      <b>김현진</b>
    </td>
    <td align="center">
      <a href="https://github.com/ppochacco">
        <img src="https://github.com/ppochacco.png" width="100px" style="border-radius: 50%;" />
      </a>
      <br />
      <b>정우선</b>
    </td>
    <td align="center">
      <a href="https://github.com/won24">
        <img src="https://github.com/won24.png" width="100px" style="border-radius: 50%;" />
      </a>
      <br />
      <b>진원</b>
    </td>
    <td align="center">
      <a href="https://github.com/NJ-KIM11">
        <img src="https://github.com/NJ-KIM11.png" width="100px" style="border-radius: 50%;" />
      </a>
      <br />
      <b>김남준</b>
    </td>
    <td align="center">
      <a href="https://github.com/kimsegwang">
        <img src="https://github.com/kimsegwang.png" width="100px" style="border-radius: 50%;" />
      </a>
      <br />
      <b>김세광</b>
    </td>
  </tr>
</table>


## 사용 방법
1. 사용자가 일기 작성하거나 이미지를 업로드합니다.
2. OCR이 텍스트를 추출하고, 버튼(AI 그림받기)을 이용해 AI가 3개 문단으로 요약합니다.
3. 각 문단을 기반으로 AI가 그림을 생성합니다.
4. 결과 화면에서 요약된 일기와 함께 생성된 이미지를 확인할 수 있습니다.

## 실행 방법
```bash
# 프로젝트 클론
git clone https://github.com/your-repo/ai-diary.git
cd ai-diary

# 가상 환경 설정 (선택 사항) [개발 기준: MiniForge3]
conda create --name ai-diary-env python=3.12
conda activate ai-diary-env

# 필수 패키지 설치
pip install -r requirements.txt

# 애플리케이션 실행
`python app.py` or `uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
```

## 기여 방법
1. 이 저장소를 포크합니다.
2. 새로운 기능을 추가한 후 PR을 생성해주세요.

## 라이선스
이 프로젝트는 MIT 라이선스를 따릅니다.
