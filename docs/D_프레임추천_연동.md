# D 담당 프레임 추천 연동

D는 `rag/client.py` 한 파일을 복사하고 `recommend_frames()`를 호출하면 됩니다. 질문을 C의 PC로 보내면 C 서버가 질문을 임베딩하고 저장된 이미지 벡터에서 유사한 프레임을 찾아 영상 ID·프레임 번호·원본 초·유사도를 반환합니다. D 쪽에는 모델이나 PyTorch를 설치하지 않습니다.

다른 PC의 함수가 C의 모델을 실행하려면 C의 서버가 먼저 켜져 있어야 합니다. 같은 PC에서는 localhost, 다른 PC에서는 접근 가능한 LAN/VPN 주소와 API key를 사용합니다. 현재 시험은 localhost에서 완료했으며 실제 D PC의 네트워크 연결은 별도 확인해야 합니다.

## D가 받는 파일과 호출 예제

`rag/client.py`를 D의 프로젝트에 `cctv_rag_client.py`로 복사합니다. 이 파일은 Python 표준 라이브러리만 사용하며 저장소의 다른 파일을 import하지 않습니다.

```python
import os
from cctv_rag_client import recommend_frames

result = recommend_frames(
    "빨간 자동차가 있는 프레임을 찾아줘",
    top_k=5,
    base_url="http://C_PC_IP:8765",
    api_key=os.environ["CCTV_RAG_API_KEY"],
)

print(result["answer"])
for candidate in result["recommendations"]:
    print(candidate["video_id"], candidate["frame_index"], candidate["seconds"])
```

`C_PC_IP`는 C의 실제 IPv4 주소로 바꿉니다. API key는 C와 D의 백엔드 설정으로만 전달하고 브라우저 코드나 공개 GitHub에 넣지 않습니다. 기본 timeout은 120초입니다. 서버의 모델은 시작 시 한 번 로드하며, 요청마다 다시 내려받거나 로드하지 않습니다.

같은 PC에서 API key 없이 localhost로 실행한 서버라면 `recommend_frames("빨간 자동차")`만 호출하면 됩니다. 선택 인자는 `video_ids=None`(전체), `video_ids=[]`(후보 없음), `representation="combined"`(scene/object 함께), `use_llm=True`입니다. D의 다른 PC에서 localhost를 사용하면 C의 PC에 연결되지 않습니다.

## 반환값

```json
{
  "status": "success",
  "answer": "관련 장면 후보가 있어요. 이 장면을 확인해 보시겠어요?\n1. 영상 fixture01, 프레임 0 (0부터 시작), 0.000초, 유사도 0.1317",
  "llm_status": "generated",
  "recommendations": [
    {
      "rank": 1,
      "item_id": "fixture-scene-0",
      "video_id": "fixture01",
      "frame_index": 0,
      "timestamp_ms": 0,
      "seconds": 0.0,
      "source_offset_ms": 0,
      "playback_time_ms": 0,
      "score": 0.13172760605812073,
      "representation": "scene",
      "bbox": null
    }
  ]
}
```

위 값은 합성 예제입니다. frame_index는 명세서의 0-based 번호를 유지합니다. timestamp_ms/seconds는 원본 영상 기준 실제 시각이며, 잘라낸 파일의 재생 시각은 `playback_time_ms / 1000`입니다. 예를 들어 원본 시각 12.3초·파일 offset 10초라면 파일 재생 위치는 2.3초입니다.

score는 코사인 유사도이며 확률이나 정답 보장 수치가 아닙니다. 동일 영상·실제 프레임·시각의 scene/object가 중복되면 최고 점수 항목 하나로 합칩니다. combined는 이 프레임별 최댓값 방식이며 D의 트랙별 RRF 순위 구현과는 별도의 프레임 추천입니다. top_k는 1..20이며 가능한 고유 프레임이 적으면 그 수만 반환합니다. 동점은 영상 ID·프레임 번호·item_id 순으로 결정합니다.

## C의 서버 시작

프로젝트 루트에서 준비된 환경은 `.venv-siglip2`입니다. 새 PC는 README의 SigLIP2 설치 절차를 사용합니다. 추가 Python 패키지는 없으며 `python -m rag download`가 고정된 GGUF와 Windows x64 CPU 런타임을 받고 크기·SHA256을 검사합니다.

```powershell
.\.venv-siglip2\Scripts\python.exe -m rag download
```

인덱스에는 items.jsonl·vectors.npy·index_manifest.json과 media가 있어야 합니다. 설정의 임베딩 모델·전처리가 해당 인덱스와 일치해야 하며, 실패한 단계의 인덱스는 거부합니다. 현재 PC의 작은 검증 인덱스로 시작하는 명령은 다음과 같습니다.

```powershell
.\.venv-siglip2\Scripts\python.exe -m rag serve --index runs/siglip2-commit-gpu-20261104/cold
```

`--index`는 index_manifest.json이 있는 폴더입니다. 같은 터미널에 `status=ready`가 표시된 후 D가 호출합니다. 기본 주소는 127.0.0.1:8765입니다. 종료는 Ctrl+C이며, C가 시작한 내부 LLM 프로세스도 함께 종료합니다. `--no-llm`을 붙이면 동일한 프레임 추천을 정형 안내문으로 반환합니다.

실제 A/B 입력이 준비되면 생성한 실제 인덱스 경로로 교체합니다. 새 팀원이 작은 입력부터 생성할 경우 다음을 사용합니다.

```powershell
.\.venv-siglip2\Scripts\python.exe -m fixtures.embedding.generate
.\.venv-siglip2\Scripts\python.exe -m embedding build --config configs/embedding.siglip2.yaml --samples fixtures/embedding/samples.jsonl --manifest fixtures/embedding/manifest.jsonl --data-root fixtures/embedding --output runs/team-index-01 --cache .cache/embedding --frame-reader fixtures.embedding.reader:read_frame --decoder-id fixture-rgb-v1
.\.venv-siglip2\Scripts\python.exe -m rag serve --index runs/team-index-01
```

## D의 다른 PC에서 연결

C의 PC에는 `configs/rag.local.json` API key 파일을 생성해 두었습니다. 새 C 서버 PC에서만 처음 한 번 `keygen`을 실행하고, 기존 파일이 있으면 생성 명령을 생략합니다. keygen은 키를 화면에 출력하지 않으며 기존 파일을 덮어쓰지 않습니다.

```powershell
.\.venv-siglip2\Scripts\python.exe -m rag keygen
.\.venv-siglip2\Scripts\python.exe -m rag serve --index runs/team-index-01 --host 0.0.0.0 --api-key-file configs/rag.local.json
```

0.0.0.0은 서버의 수신 설정이며 D의 base_url에는 C의 실제 주소를 넣습니다. 같은 신뢰 LAN/VPN에서 8765/TCP에 접근 가능해야 합니다. Windows 방화벽 설정은 변경하지 않았으므로 연결 환경에서 허용 여부를 확인해야 합니다. 인터넷 포트 포워딩은 구성하지 않았습니다. 다른 인터페이스에 bind할 때는 32자 이상의 API key가 필수입니다.

D는 안전하게 전달받은 키를 백엔드 환경변수 CCTV_RAG_API_KEY에 설정합니다. client 함수는 api_key 인자가 없으면 이 환경변수를 사용합니다. *.local.json은 Git과 소스 패키지에서 제외하며, keygen은 이전 브랜치로 전환해도 보호가 유지되도록 로컬 Git 제외 규칙도 등록합니다.

## LLM의 역할과 선정

최초 에이전트 토의 기록은 `RAG_에이전트_토의.md`에 있습니다. 이후 사용자 요청으로 기본 모델을 **Qwen3.5-4B Q4_K_M**으로 교체했습니다. Unsloth의 `unsloth/Qwen3.5-4B-GGUF` 배포본이며 GGUF는 2,740,937,888바이트, CPU 런타임 ZIP은 24,730,921바이트입니다. URL·리비전·SHA256은 configs/rag-model.json에 고정했습니다. 이미지 입력용 mmproj는 현재 텍스트 안내 기능에 필요하지 않아 다운로드하지 않습니다.

SigLIP2는 GPU, Qwen은 CPU에서 실행합니다. Qwen의 실행 context는 2048, 출력 상한은 128토큰, CPU 스레드는 최대 4, 동시 요청은 1개입니다. 현재 4GB GPU를 임베딩과 공유하므로 LLM의 GPU 레이어는 0으로 유지합니다. 자연스러운 선택 질문 생성 시험에서 Qwen3.5-4B 프로세스 peak working set은 4,545,437,696바이트(약 4.55GB)였습니다. 모델 파일 크기와 실제 RAM 사용량은 다르며 전체 앱과 OS에도 별도 RAM이 필요합니다.

이전 모델과 비교할 때는 다운로드 및 serve 명령에 `--manifest configs/rag-model.qwen3-0.6b.json`을 붙입니다. 기존 0.6B 가중치는 삭제하지 않았고 D의 함수 시그니처와 응답 형식은 동일합니다.

프레임·초·순위는 검색 코드가 확정합니다. LLM은 질문과 검색 후보 수를 받아, 여러 후보 중 어떤 것을 확인할지 묻는 자연스러운 한국어 한두 문장을 생성합니다. 후보가 하나면 그 장면을 확인할지 묻습니다. 고정 안내문 목록과 enum 제한은 제거했습니다. summary는 생성 문장, answer는 summary와 코드가 작성한 프레임 목록의 조합입니다. 예를 들면 “관련 장면 후보가 여러 개 있어요. 어떤 장면을 확인하시겠어요?” 뒤에 후보 목록이 붙습니다.

생성 안내는 200자 이하의 한 줄이며 물음표로 끝납니다. 숫자를 포함하거나 형식이 잘못된 출력, LLM 오류/시간 초과에는 후보 수에 맞는 정형 선택 질문으로 대체하고 llm_status=fallback을 반환합니다. LLM 없이도 추천 목록은 같습니다. 이 검사는 출력 형식 검사이며 생성 문장의 의미를 보증하지 않습니다. 실제 영상을 보았다고 단정하거나 관찰 사실을 추가하지 않도록 프롬프트로 지시하며, 영상 내용을 판독하는 자유형 질의응답 기능은 아닙니다.

검색 임베딩의 SigLIP2 한도는 특수 토큰 포함 64토큰입니다. 길이를 넘기면 TEXT_TOO_LONG을 반환하며 임의로 자르지 않습니다. LLM의 context 2048은 이 검색 한도를 늘리지 않습니다.

## 오류와 검사

유효한 요청의 빈 결과는 success와 recommendations=[]입니다. 잘못된 영상·긴 질문은 각각 UNKNOWN_VIDEO·TEXT_TOO_LONG, 인증 오류는 UNAUTHORIZED입니다. 다른 요청이 처리 중이면 BUSY(503)이며 잠시 후 다시 호출합니다. D용 함수는 오류 code/message를 RuntimeError에 담습니다. 서버가 꺼졌거나 주소·방화벽·timeout 문제가 있으면 SERVER_UNAVAILABLE을 반환합니다.

전체 자동검사와 실제 모델·HTTP 검사는 다음 명령으로 재현합니다.

```powershell
.\.venv311\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-siglip2\Scripts\python.exe -m tools.verify_rag --index runs/siglip2-commit-gpu-20261104/cold --output runs/my-rag-check
```

2026-11-05 자연스러운 선택 질문 생성으로 변경한 뒤 자동검사는 101개, 실제 연동 검사는 9개 PASS입니다. 실제 연동에는 한국어 선택 질문 생성, 단일 후보 확인 질문, 프레임/시각 대조, 중복 제거, LLM 사용/미사용 목록 일치, 빈 필터, 인증·긴 질문·없는 영상, LLM timeout 대체 응답, 허위 999초 지시 방어, 외부 패키지를 끈 별도 Python 프로세스에서 D 파일 단독 호출을 포함합니다. 실측 원문은 `docs/verification/qwen35-conversation-live-20261105.json`, 자동검사 원문은 `docs/verification/qwen35-conversation-unit-20261105.txt`입니다. 기존 enum 제한 방식의 기록은 qwen35-4b-live-20261105.json에 별도로 보존합니다.

합성 예제에서 빨간 자동차 질문의 1위는 프레임 0·0.000초, 파란 자동차는 프레임 1·0.200초였습니다. 자연스러운 선택 질문 생성 시 응답 시간은 각각 약 10.99초·5.03초, 단일 후보 질문은 약 3.30초였습니다. 서버 시작·모델 로딩을 제외한 소수 예제의 작동 확인 수치이며 엄격한 속도 비교 실험은 아닙니다. 실제 응답은 “빨간 자동차에 대한 검색 결과가 두 가지로 나왔어요. 어떤 후보를 확인하시겠어요?”, 단일 후보에서는 “자동차에 대한 검색 결과가 하나 나왔습니다. 확인하시겠습니까?”였습니다. 실제 CCTV 검색 정확도와 D PC의 LAN 연결은 아직 검증하지 않았습니다.
