import os
import re
import sys
import json
import sqlite3
import shutil
import logging
import time
import subprocess
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# Import Google Cloud and OpenAI libraries
try:
    from google.cloud import texttospeech_v1beta1 as texttospeech, storage
    from google.api_core import operations_v1
    from google.oauth2 import service_account
    from google.auth.transport import grpc as google_auth_grpc
    from google.auth.transport import requests as google_auth_requests
    from openai import OpenAI
    from pydub import AudioSegment
except ImportError as e:
    print(f"Error: A required library is missing. Please install it. Missing: {e.name}")
    print("You can install dependencies using: pip install google-cloud-texttospeech google-cloud-storage google-auth openai pydub")
    sys.exit(1)


# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)


# --- 설정 관리 ---
class Config:
    """애플리케이션 설정을 관리하는 클래스"""
    def __init__(self, config_path="config.json"):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"FATAL: Error reading config file '{config_path}': {e}")
            sys.exit(1)
        self.openai_api_key = config_data.get("OPENAI_API_KEY")
        self.google_credentials_path = config_data.get("GOOGLE_APPLICATION_CREDENTIALS")
        self.gcs_bucket_name = config_data.get("GCS_BUCKET_NAME")
        if not all([self.openai_api_key, self.google_credentials_path, self.gcs_bucket_name]):
            logger.error("FATAL: Required keys missing in config.json.")
            sys.exit(1)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.google_credentials_path
        logger.info(f"Configuration loaded successfully from {config_path}.")


class StudyNoteProcessor:
    """스터디 노트 생성 및 관련 자산 처리를 위한 전체 워크플로우를 관리하는 클래스"""

    def __init__(self, base_folder_path: str, config: Config):
        self.base_path = Path(base_folder_path)
        self.config = config
        
        self.output_path = self.base_path / "output"
        self.audio_summary_path = self.base_path / "summary"
        self.downloads_path = Path.home() / "Downloads"

        try:
            self.openai_client = OpenAI(api_key=self.config.openai_api_key)
            self.tts_client = texttospeech.TextToSpeechLongAudioSynthesizeClient()
            self.storage_client = storage.Client()
            
            self.gcs_bucket = self.storage_client.bucket(self.config.gcs_bucket_name)
            if not self.gcs_bucket.exists():
                logger.error(f"GCS bucket '{self.config.gcs_bucket_name}' does not exist.")
                sys.exit(1)
        except Exception as e:
            logger.error(f"Failed to initialize API clients: {e}")
            sys.exit(1)
            
        logger.info(f"StudyNoteProcessor initialized. All outputs will be saved in: {self.base_path}")

    # --- 단계 1: 텍스트 파일 생성 (DB -> TXT -> AI) ---
    def step1_generate_ai_scripts(self) -> bool:
        logger.info("\n" + "="*20 + " Step 1: Generating AI Scripts " + "="*20)
        try:
            category_data = self._db_to_txt()
            if not category_data:
                logger.error("Failed to process databases. Aborting Step 1.")
                return False
            for category_name, text_content in category_data.items():
                logger.info(f"\n--- Processing category: {category_name} ---")
                category_folder = self.output_path / category_name
                if (category_folder / "2.txt").exists() and (category_folder / "3.txt").exists():
                    logger.info(f"Skipping AI script generation for '{category_name}' as files already exist.")
                    continue
                studynote_content = self._run_studynote_api(category_name, text_content)
                if studynote_content:
                    self._run_class_script_api(category_name, studynote_content)
            logger.info("="*20 + " Step 1: AI Scripts Generation COMPLETE " + "="*20)
            return True
        except Exception as e:
            logger.error(f"An unexpected error in Step 1: {e}", exc_info=True)
            return False

    def _get_sorted_db_files(self) -> List[Path]:
        if not self.base_path.exists():
            logger.error(f"Base data folder not found: {self.base_path}")
            return []
        db_files = list(self.base_path.glob("*.db"))
        db_files.sort(key=lambda p: (re.search(r'(\d+)', p.name) or re.search(r'(\d+)', '0')).group(1), reverse=True)
        logger.info(f"Found and sorted {len(db_files)} DB files in '{self.base_path}'.")
        return db_files
        
    def _db_to_txt(self) -> Dict[str, str]:
        db_files_to_process = self._get_sorted_db_files()
        if not db_files_to_process: return {}
        db_files_to_process = db_files_to_process[:5]
        logger.info(f"Processing latest {len(db_files_to_process)} DB files: {[p.name for p in db_files_to_process]}")
        category_data = defaultdict(list)
        for db_path in db_files_to_process:
            try:
                conn = sqlite3.connect(db_path)
                cursor = conn.cursor()
                cursor.execute("SELECT Question_id, Big_Question, Question, Correct_Option, Option1, Option2, Option3, Option4, Answer_description, Category, Date_information FROM questions")
                for row in cursor.fetchall():
                    (question_id, big_q, q_text, correct_opt, opt1, opt2, opt3, opt4, desc, cat, date_info) = row
                    options = [o if isinstance(o, str) else "[이미지]" for o in [opt1, opt2, opt3, opt4]]
                    question_text = q_text if isinstance(q_text, str) else "[이미지]"
                    try: correct_option_text = options[int(correct_opt) - 1]
                    except (ValueError, IndexError, TypeError): correct_option_text = "[알 수 없음]"
                    output_text = (
                        f"Question_id: {question_id}\n"
                        f"Big_Question: {big_q or ''}\n"
                        f"Question: {question_text}\n"
                        f"Correct_Option: {correct_opt} (옵션: {correct_option_text})\n"
                        f"Answer_description: {desc or ''}\n"
                        f"Date_information: {date_info or '[날짜 정보 없음]'}\n"
                        "------------------------\n"
                    )
                    category_data[cat or "Uncategorized"].append(output_text)
                conn.close()
            except sqlite3.Error as e:
                logger.error(f"Error reading from DB '{db_path.name}': {e}")
                continue
        final_category_content = {}
        self.output_path.mkdir(exist_ok=True)
        for category, entries in category_data.items():
            safe_category_name = category.replace('/', '_').replace('\\', '_')
            category_folder = self.output_path / safe_category_name
            category_folder.mkdir(exist_ok=True)
            full_content = f"Category: {category}\n\n" + "".join(entries)
            final_category_content[safe_category_name] = full_content
            with open(category_folder / "1.txt", 'w', encoding='utf-8') as f:
                f.write(full_content)
        return final_category_content

    # ✅ 수정: _run_studynote_api 함수를 StudyNoteProcessor 클래스 안으로 이동
    def _run_studynote_api(self, category_name: str, text_content: str) -> str:
        """
        시험 문제 텍스트를 분석하여 핵심 개념 중심의 학습 노트를 생성합니다.
        """
        logger.info(f"'{category_name}'에 대한 학습 노트 생성을 시작합니다...")
        system_message = """
당신은 교육 자료 전문가입니다. 주어진 시험 문제들을 분석하여, 학생들이 핵심 개념을 쉽게 파악할 수 있는 통합 학습 노트를 작성해야 합니다.

**핵심 지침:**
1.  **주제 통합:** 개별 문제를 그대로 설명하지 마세요. 여러 문제에서 공통으로 다루는 개념이나 원칙을 하나의 주제로 묶어 통합적으로 설명해야 합니다.
2.  **구조 준수:** 아래의 출력 형식을 반드시 지켜주세요.
    - 각 주제는 `1. 주제명` 과 같이 번호로 시작합니다.
    - 핵심 내용은 `-` 기호를 사용해 개조식으로 명확하게 정리합니다.
    - 주제 설명 마지막에는 `관련 문제:` 섹션을 추가하고, 관련된 모든 문제를 `-<y_bin_46>년 MM월, Question_id: NN` 형식으로 나열합니다.
    - 주제와 주제 사이는 `---` 구분선을 넣어주세요.
3.  **문체:** 전문적이고 간결한 서술체를 사용합니다.

**출력 형식 및 예시:**
1. 공중위생관리법의 주요 용어 정의
   - 공중위생영업: 다수인을 대상으로 위생관리 서비스를 제공하는 영업으로, 숙박업, 목욕장업, 이용업, 미용업 등이 해당됩니다.
   - 영업자: 관계 법령에 따라 허가/신고/등록을 하고 영업을 하는 자.
   - 실무 적용: 허가 없이 영업 시 강력한 행정처분(영업장 폐쇄 등)이 따르므로, 창업 전 허가/신고 절차 확인이 필수적입니다.
   관련 문제:
   - 2023년 3월, Question_id: 15
   - 2023년 3월, Question_id: 18
---
2. 소독 및 위생 기준
   - (내용...)
"""
        user_prompt = f"""
다음 시험 문제들을 분석하여, 시스템 메시지에 명시된 지침과 형식에 따라 학습 노트를 작성해주세요.

[시험 문제 내용]
{text_content}
"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=4096,
                temperature=0.2
            )
            raw_content = response.choices[0].message.content
            cleaned_content = re.sub(r'^\#{2,4}\s*|(\*{2})(.*?)(\*{2})', r'\2', raw_content, flags=re.MULTILINE).strip()
            output_file_path = self.output_path / category_name / "2.txt"
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(cleaned_content)
            logger.info(f"학습 노트가 성공적으로 저장되었습니다: {output_file_path}")
            return cleaned_content
        except Exception as e:
            logger.error(f"'{category_name}'에 대한 학습 노트 생성 중 오류 발생: {e}")
            return ""

    # ✅ 수정: _run_class_script_api 함수를 StudyNoteProcessor 클래스 안으로 이동
    def _run_class_script_api(self, category_name: str, studynote_content: str):
        """
        학습 노트 내용을 기반으로, 전문 강사의 친절한 구어체 강의 스크립트를 생성합니다.
        """
        logger.info(f"'{category_name}'에 대한 강의 스크립트 생성을 시작합니다...")
        system_message = """
당신은 학생들의 눈높이에 맞춰 설명하는 전문 강사입니다. 주어진 학습 노트를 기반으로, 실제 강의처럼 친절하고 생생한 구어체 강의 스크립트를 작성해야 합니다.

**스크립트 작성 원칙:**
- **상세 설명:** 절대 요약하지 마세요. 각 개념을 하나하나 풀어서 자세히 설명해야 합니다.
- **친절한 구어체:** 수강생에게 직접 말하듯 "자, 그럼 다음으로..." 와 같이 자연스러운 말투를 사용하세요.
- **논리적 흐름:** 강의 내용이 처음부터 끝까지 물 흐르듯 자연스럽게 연결되도록 구성하세요.
"""
        user_prompt = f"""
다음 학습 노트를 기반으로, 시스템 메시지에 명시된 원칙에 따라 강의 스크립트를 작성해주세요.
강의는 반드시 "여러분, 안녕하세요!" 라는 인사말로 시작해야 합니다.

---
[학습 노트 내용]
{studynote_content}
---
"""
        try:
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=4096
            )
            raw_content = response.choices[0].message.content
            content = re.sub(r'Question_id\s*:\s*(\d+)', r'\1번', raw_content, flags=re.IGNORECASE)
            content = re.sub(r'(\d{4}년 \d+월),', r'\1', content)
            content = re.sub(r'^- ', '', content, flags=re.MULTILINE)
            content = re.sub(r'\#{2,4}|\*{2}|---', '', content).strip()
            output_file_path = self.output_path / category_name / "3.txt"
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            logger.info(f"강의 스크립트가 성공적으로 저장되었습니다: {output_file_path}")
        except Exception as e:
            logger.error(f"강의 스크립트 생성 중 오류 발생 ('{category_name}'): {e}")

    # --- 단계 2: 강의 오디오 생성 ---
    def step2_generate_lecture_audio(self) -> bool:
        logger.info("\n" + "="*20 + " Step 2: Generating Lecture Audio " + "="*20)
        try:
            if not self.output_path.exists(): return False
            categories = [d.name for d in self.output_path.iterdir() if d.is_dir()]
            if not categories: return True
            with ThreadPoolExecutor(max_workers=5) as executor:
                results = [future.result() for future in [executor.submit(self._process_audio_for_category, c, i) for i, c in enumerate(sorted(categories), 1)]]
                if not all(results): return False
            return True
        except Exception as e:
            logger.error(f"An unexpected error in Step 2: {e}", exc_info=True)
            return False

    def _process_audio_for_category(self, category_name: str, index: int) -> bool:
        try:
            logger.info(f"Starting audio process for '{category_name}' (lecture{index}.mp3)")
            script_path = self.output_path / category_name / "3.txt"
            if not script_path.exists(): return True
            with open(script_path, 'r', encoding='utf-8') as f: text_content = f.read()
            if not text_content.strip(): return True
            operation = self._synthesize_long_audio(text_content, category_name)
            if not operation: return False
            if not self._wait_for_operation_completion(operation, category_name): return False
            downloaded_wav_files = self._download_wavs_from_gcs(f"audio_output/{category_name}/")
            if not downloaded_wav_files:
                logger.error(f"Failed to download any audio files for '{category_name}'. Aborting.")
                return False
            merged_wav_path = self._merge_wav_files(downloaded_wav_files, category_name)
            if not merged_wav_path: return False
            mp3_path = self._convert_wav_to_mp3(merged_wav_path, category_name)
            if not mp3_path: return False
            self._move_final_mp3(mp3_path, index)
            self._cleanup_temp_files(category_name, downloaded_wav_files + [merged_wav_path, mp3_path])
            return True
        except Exception as e:
            logger.error(f"Audio processing failed for category '{category_name}': {e}", exc_info=True)
            return False

    def _synthesize_long_audio(self, text: str, category_name: str):
        try:
            full_gcs_uri = f"gs://{self.config.gcs_bucket_name}/audio_output/{category_name}/synthesized_audio_{int(time.time())}.wav"
            request = texttospeech.SynthesizeLongAudioRequest(
                parent=f"projects/{self.storage_client.project}/locations/global",
                input=texttospeech.SynthesisInput(text=text),
                voice=texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Standard-D"),
                audio_config=texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.LINEAR16, sample_rate_hertz=16000),
                output_gcs_uri=full_gcs_uri,
            )
            logger.info(f"Requesting long audio synthesis to GCS URI: {full_gcs_uri}")
            return self.tts_client.synthesize_long_audio(request=request)
        except Exception as e:
            logger.error(f"SynthesizeLongAudio request failed: {e}")
            return None

    def _wait_for_operation_completion(self, operation, category_name: str) -> bool:
        logger.info(f"Waiting for operation '{operation.operation.name}' to complete...")
        try:
            scopes = ['https://www.googleapis.com/auth/cloud-platform']
            credentials = service_account.Credentials.from_service_account_file(
                self.config.google_credentials_path, scopes=scopes
            )
            request_session = google_auth_requests.Request()
            credentials.refresh(request_session)
            
            channel = google_auth_grpc.secure_authorized_channel(
                credentials, request_session, "texttospeech.googleapis.com:443"
            )
            operations_client = operations_v1.OperationsClient(channel)

            timeout = 1800
            start_time = time.time()
            while time.time() - start_time < timeout:
                op_status = operations_client.get_operation(name=operation.operation.name)
                if op_status.done:
                    if op_status.error.code == 0:
                        if op_status.response:
                            logger.info(f"{category_name} 오디오 생성 완료")
                            return True
                        else:
                            logger.error(f"{category_name} 작업은 '완료'되었으나, 생성된 오디오 파일 정보가 없습니다. (Silent Failure)")
                            return False
                    else:
                        logger.error(f"{category_name} 오디오 생성 실패: {op_status.error}")
                        return False
                
                logger.info(f"{category_name} 작업 진행 중... (경과 시간: {int(time.time() - start_time)}초)")
                time.sleep(30)
            
            logger.error(f"{category_name} 작업 모니터링 시간 초과")
            return False
        except Exception as e:
            logger.error(f"Error while waiting for operation completion: {e}", exc_info=True)
            return False

    def _download_wavs_from_gcs(self, gcs_prefix: str) -> List[Path]:
        try:
            blobs = self.storage_client.list_blobs(self.config.gcs_bucket_name, prefix=gcs_prefix)
            output_dir = self.downloads_path / Path(gcs_prefix).name
            output_dir.mkdir(parents=True, exist_ok=True)
            downloaded_files = []
            for blob in blobs:
                if blob.name.endswith(".wav"):
                    destination = output_dir / Path(blob.name).name
                    blob.download_to_filename(destination)
                    if destination.exists() and destination.stat().st_size > 0:
                        downloaded_files.append(destination)
            return downloaded_files
        except Exception as e:
            logger.error(f"Error downloading from GCS: {e}")
            return []

    def _merge_wav_files(self, wav_files: List[Path], category_name: str) -> Path | None:
        if not wav_files: return None
        if len(wav_files) == 1: return wav_files[0]
        wav_files.sort(key=lambda p: int(re.search(r'(\d+)\.wav$', p.name).group(1)) if re.search(r'(\d+)\.wav$', p.name) else 0)
        combined = AudioSegment.empty()
        for wav_path in wav_files:
            try: combined += AudioSegment.from_wav(wav_path)
            except Exception as e:
                logger.error(f"Could not process WAV {wav_path}: {e}")
                return None
        merged_path = self.downloads_path / f"{category_name}_merged.wav"
        try:
            combined.export(merged_path, format="wav")
            return merged_path
        except Exception as e:
            logger.error(f"Failed to export merged WAV: {e}")
            return None
            
    def _convert_wav_to_mp3(self, wav_path: Path, category_name: str) -> Path | None:
        mp3_path = self.downloads_path / f"{category_name}.mp3"
        try:
            AudioSegment.from_wav(wav_path).export(mp3_path, format="mp3", bitrate="64k")
            return mp3_path
        except Exception as e:
            logger.error(f"Failed to convert to MP3: {e}")
            return None

    def _move_final_mp3(self, mp3_path: Path, index: int):
        target_dir = self.audio_summary_path
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.move(str(mp3_path), str(target_dir / f"lecture{index}.mp3"))
        logger.info(f"Moved final MP3 to {target_dir}")

    def _cleanup_temp_files(self, category_name: str, files_to_delete: List[Path]):
        gcs_prefix = f"audio_output/{category_name}/"
        for file_path in files_to_delete:
            if file_path and file_path.exists():
                try: file_path.unlink()
                except OSError: pass
        temp_dir = self.downloads_path / category_name
        if temp_dir.exists():
            try: shutil.rmtree(temp_dir)
            except OSError: pass
        try:
            for blob in self.storage_client.list_blobs(self.config.gcs_bucket_name, prefix=gcs_prefix):
                blob.delete()
        except Exception:
            pass

# --- 메인 실행 로직 ---
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 4_StudyNote.py <path_to_data_folder>")
        sys.exit(1)
    data_folder = sys.argv[1]
    if not os.path.isdir(data_folder):
        print(f"Error: Provided path '{data_folder}' is not a valid directory.")
        sys.exit(1)
    try:
        config = Config(os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json"))
        processor = StudyNoteProcessor(data_folder, config)
        
        if processor.step1_generate_ai_scripts():
            if processor.step2_generate_lecture_audio():
                logger.info("\n✅ All processes completed successfully!")
            else:
                logger.error("❌ Process failed during Step 2: Generating Lecture Audio.")
                sys.exit(1)
        else:
            logger.error("❌ Process failed during Step 1: Generating AI Scripts.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        sys.exit(1)