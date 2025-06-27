# 개선된 비동기 처리 pro_api.py
import asyncio
import sys
import aiosqlite
import openai
import sqlite3
import os
import glob
import re
import io
import json
import logging
import time
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

from google.cloud import vision
from google.api_core.exceptions import GoogleAPIError, Forbidden, Unauthorized
from PIL import Image
import pytesseract
from openai import AsyncOpenAI

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# 설정 관리
# ============================================================================

@dataclass
class ProcessingConfig:
    """처리 설정을 담는 데이터클래스"""
    openai_api_key: str
    model_name: str = "gpt-4o"
    chunk_size: int = 10
    max_concurrent_tasks: int = 5  # 너무 높으면 rate limit 위험
    max_retries: int = 3
    retry_delay: float = 2.0
    timeout: float = 180.0

def load_config(filename="config.json") -> ProcessingConfig:
    """설정 파일을 로드하고 ProcessingConfig 객체 반환"""
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            config_data = json.load(f)
        
        required_keys = ["OPENAI_API_KEY", "MODEL_NAME"]
        for key in required_keys:
            if key not in config_data or not config_data[key]:
                raise ValueError(f"'{filename}' 파일에 '{key}'가 유효한 값으로 설정되지 않았습니다.")
        
        return ProcessingConfig(
            openai_api_key=config_data["OPENAI_API_KEY"],
            model_name=config_data["MODEL_NAME"],
            chunk_size=config_data.get("CHUNK_SIZE", 10),
            max_concurrent_tasks=config_data.get("MAX_CONCURRENT_TASKS", 5),
            max_retries=config_data.get("MAX_RETRIES", 3),
            retry_delay=config_data.get("RETRY_DELAY", 2.0),
            timeout=config_data.get("TIMEOUT", 180.0)
        )
        
    except FileNotFoundError:
        raise FileNotFoundError(f"설정 파일 '{filename}'을 찾을 수 없습니다. 'config.json' 파일을 생성해주세요.")
    except json.JSONDecodeError:
        raise ValueError(f"설정 파일 '{filename}'이 올바른 JSON 형식이 아닙니다.")

# ============================================================================
# 유틸리티 함수들
# ============================================================================

date_pattern = re.compile(r'(\d{8})')

def extract_date_from_filename(filename: str) -> Optional[int]:
    """파일명에서 날짜 추출"""
    match = date_pattern.search(filename)
    return int(match.group(1)) if match else None

def is_blob_image(value: Any) -> bool:
    """값이 유효한 이미지 BLOB인지 확인"""
    if not isinstance(value, bytes) or len(value) < 100:
        return False
    try:
        header = value[:12]
        return (header.startswith(b'\xff\xd8') or  # JPEG
                header.startswith(b'\x89PNG\r\n\x1a\n') or  # PNG
                header.startswith(b'GIF87a') or header.startswith(b'GIF89a'))  # GIF
    except Exception:
        return False

def safely_decode_text(value: Any, question_id: Optional[int] = None) -> str:
    """텍스트를 안전하게 디코딩"""
    if isinstance(value, str):
        return value.strip()
    elif isinstance(value, bytes):
        try:
            return value.decode('utf-8').strip()
        except UnicodeDecodeError:
            logger.warning(f"question_id={question_id}의 바이트 디코딩 실패")
            return ""
    return str(value or "").strip()

def chunk_list(data: List, chunk_size: int) -> List[List]:
    """리스트를 청크로 분할"""
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def remove_code_fences(text: str) -> str:
    """GPT 응답에서 코드 블록 제거"""
    return re.sub(r"```[a-zA-Z]*|```", "", text).strip()

# ============================================================================
# 비동기 OCR 처리 클래스
# ============================================================================

class AsyncOCRProcessor:
    """비동기 OCR 처리를 담당하는 클래스"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.vision_client = None
        self._init_vision_client()
    
    def _init_vision_client(self):
        """Google Cloud Vision 비동기 클라이언트 초기화"""
        try:
            self.vision_client = vision.ImageAnnotatorAsyncClient()
            logger.info("Google Cloud Vision 비동기 클라이언트 초기화 성공")
        except Exception as e:
            logger.warning(f"Google Cloud Vision 초기화 실패: {e}")
            self.vision_client = None
    
    # --- [수정] 상세 에러 로깅을 위해 함수 수정 ---
    async def extract_text_from_blob(self, blob_data: bytes, context: str = "") -> str:
        """BLOB 데이터에서 텍스트 추출 (Google Vision + Tesseract 백업)"""
        if not is_blob_image(blob_data):
            return ""
        
        # 1차: Google Cloud Vision 시도
        if self.vision_client:
            google_vision_result = ""
            try:
                # _google_vision_ocr 함수가 에러 메시지 또는 성공 결과를 반환
                google_vision_result = await self._google_vision_ocr(blob_data, context)
                # 성공적인 결과(에러 메시지 패턴으로 시작하지 않음)인 경우 즉시 반환
                if google_vision_result and not google_vision_result.startswith("(OCR"):
                    return google_vision_result
                else:
                    # API가 에러를 반환한 경우, 경고 로그에 상세 내용 기록
                    logger.warning(f"{context} - Google Vision API가 에러를 반환했습니다: {google_vision_result}")

            except Exception as e:
                # API 호출 중 예외가 발생한 경우, 에러 로그에 상세 내용 기록
                logger.error(f"{context} - Google Vision OCR 호출 중 예외 발생", exc_info=True)
        
        # 2차: Tesseract 백업 (Google Vision이 실패했거나, 클라이언트가 없거나, 에러를 반환한 모든 경우)
        logger.info(f"{context} - Tesseract OCR로 재시도")
        return await asyncio.get_event_loop().run_in_executor(
            None, self._tesseract_ocr, blob_data, context
        )
    # --- [수정 끝] ---
    
    async def _google_vision_ocr(self, blob_data: bytes, context: str) -> str:
        """Google Cloud Vision 비동기 OCR"""
        try:
            # 이미지 최적화
            with Image.open(io.BytesIO(blob_data)) as pil_image:
                if pil_image.width > 2000 or pil_image.height > 2000:
                    pil_image.thumbnail((2000, 2000), Image.Resampling.LANCZOS)
                
                buf = io.BytesIO()
                pil_image.save(buf, format='PNG', optimize=True)
                content = buf.getvalue()

            image = vision.Image(content=content)
            response = await self.vision_client.text_detection(image=image)

            if response.error.message:
                return f"OCR 에러: {response.error.message}"
            
            texts = response.text_annotations
            if texts:
                result = texts[0].description.strip()
                logger.debug(f"{context} - Google OCR 성공: '{result[:50]}...'")
                return result
            
            return ""
            
        except (Forbidden, Unauthorized) as auth_err:
            raise RuntimeError(f"Google Cloud Vision 인증 실패: {auth_err}") from auth_err
        except GoogleAPIError as api_err:
            return f"(Google OCR API 오류: {api_err})"
        except Exception as e:
            return f"(Google OCR 처리 실패: {e})"
    
    def _tesseract_ocr(self, blob_data: bytes, context: str) -> str:
        """Tesseract OCR (동기 함수)"""
        try:
            with Image.open(io.BytesIO(blob_data)) as pil_image:
                result = pytesseract.image_to_string(
                    pil_image, 
                    lang='kor+eng',
                    config='--psm 6'
                ).strip()
                logger.debug(f"{context} - Tesseract OCR 완료: '{result[:50]}...'")
                return result
        except Exception as e:
            logger.error(f"{context} - Tesseract OCR 실패: {e}")
            return f"(Tesseract OCR 실패: {e})"

# ============================================================================
# 비동기 OpenAI 클라이언트
# ============================================================================

class AsyncOpenAIClient:
    """OpenAI API 비동기 호출을 담당하는 클래스"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.client = AsyncOpenAI(api_key=config.openai_api_key)
        self.total_tokens = {'prompt': 0, 'completion': 0, 'total': 0}
    
    async def generate_explanations_batch(
        self, 
        questions_batch: List[Dict], 
        chunk_id: str, 
        semaphore: asyncio.Semaphore
    ) -> Tuple[List[Dict], Optional[Any]]:
        """문제 배치에 대한 해설 생성"""
        async with semaphore:
            if not questions_batch:
                return [], None
            
            system_message = self._get_system_message()
            user_prompt = self._build_user_prompt(questions_batch)
            
            for attempt in range(self.config.max_retries):
                try:
                    logger.info(f"Chunk {chunk_id}: OpenAI API 호출 시작 (시도 {attempt + 1})")
                    
                    response = await self.client.chat.completions.create(
                        model=self.config.model_name,
                        messages=[
                            {"role": "system", "content": system_message},
                            {"role": "user", "content": user_prompt}
                        ],
                        max_completion_tokens=8000,
                        timeout=self.config.timeout
                    )
                    
                    if not response.choices:
                        logger.warning(f"Chunk {chunk_id}: 빈 응답 수신")
                        continue
                    
                    content = response.choices[0].message.content.strip()
                    cleaned_content = remove_code_fences(content)
                    
                    try:
                        explanations = json.loads(cleaned_content)
                        logger.info(f"Chunk {chunk_id}: 해설 생성 성공 ({len(explanations)}개)")
                        self._update_token_usage(response.usage)
                        return explanations, response.usage
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Chunk {chunk_id}: JSON 파싱 실패 (시도 {attempt + 1}): {e}")
                        if attempt == self.config.max_retries - 1:
                            logger.error(f"Chunk {chunk_id}: 원본 응답: {content[:200]}...")
                
                except openai.RateLimitError as e:
                    wait_time = 60 * (attempt + 1)
                    logger.warning(f"Chunk {chunk_id}: Rate limit 도달, {wait_time}초 대기")
                    await asyncio.sleep(wait_time)
                    
                except openai.APITimeoutError as e:
                    logger.warning(f"Chunk {chunk_id}: API 타임아웃 (시도 {attempt + 1})")
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
                    
                except Exception as e:
                    logger.error(f"Chunk {chunk_id}: API 호출 오류 (시도 {attempt + 1}): {e}")
                    if attempt == self.config.max_retries - 1:
                        break
                    await asyncio.sleep(self.config.retry_delay * (attempt + 1))
            
            logger.error(f"Chunk {chunk_id}: 모든 재시도 실패")
            return [], None
    
    def _update_token_usage(self, usage: Any):
        """토큰 사용량 업데이트"""
        if usage:
            self.total_tokens['prompt'] += usage.prompt_tokens
            self.total_tokens['completion'] += usage.completion_tokens  
            self.total_tokens['total'] += usage.total_tokens
            logger.info(
                f"토큰 사용: prompt={usage.prompt_tokens}, "
                f"completion={usage.completion_tokens}, total={usage.total_tokens}"
            )
    
    @staticmethod
    def _get_system_message() -> str:
        """시스템 메시지 반환"""
        return """
당신은 자격증 시험 기출문제 해설을 작성하는 매우 숙련된 교육 전문가입니다. 다음 지침을 엄격히 준수하여 전문적이고 상세한 해설을 한국어 높임말로 작성해 주세요:
1.해설
   - 정답의 근거가 되는 핵심 개념이나 원리를 명확하고 상세하게 설명하세요.
   - '옳은 것'을 묻는 경우: 왜 해당 선택지가 정답인지 상세히 설명하고, 나머지 오답 선택지들이 왜 틀렸는지 각각 간략하게 이유를 밝히세요.
   - '옳지 않은 것'을 묻는 경우: 왜 해당 선택지가 옳지 않은지 상세히 설명하세요.
   - 환각 없이 제공된 정보에만 근거하여 정확한 정보를 제공하세요.
2. 계산 문제: 필요시 계산 과정이나 공식을 단계별로 명확하게 제시하세요.
3. 형식 및 언어:
   - 문제 내용이나 선택지를 해설에 다시 쓰지 마세요.
   - 선택지 언급 시 숫자(1, 2, 3, 4)를 사용하세요.
   - '정답은 X번 입니다.'로 시작하지 마세요.
   - LaTeX 코드 기호는 사용하지 마세요.
6. 출력 형식: 순수한 JSON 배열 형식만 출력하세요. 코드 블록 마크다운은 절대 포함하지 마세요.

예시: [{"question_id": 1, "explanation": "상세한 해설 내용..."}, {"question_id": 2, "explanation": "다음 문제 해설..."}]
""".strip()
    
    def _build_user_prompt(self, questions_data: List[Dict]) -> str:
        """사용자 프롬프트 생성"""
        prompt_parts = [
            f"아래는 총 {len(questions_data)}개의 문제입니다. 각 문제별 해설을 JSON 배열로 출력해 주세요.\n"
        ]
        
        for idx, q in enumerate(questions_data, 1):
            prompt_parts.extend([
                f"{idx}) question_id: {q['question_id']}",
                f"문제: {q['question_text']}",
                f"보기:",
                f"1) {q['option1']}",
                f"2) {q['option2']}",
                f"3) {q['option3']}",
                f"4) {q['option4']}",
                f"정답 번호: {q['correct_option']}\n"
            ])
        
        return "\n".join(prompt_parts)

# ============================================================================
# 메인 처리 클래스
# ============================================================================

class AsyncProAPIProcessor:
    """비동기 API 처리 메인 클래스"""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.ocr_processor = AsyncOCRProcessor(config)
        self.openai_client = AsyncOpenAIClient(config)
    
    async def process_folder(self, folder_path: str, queue=None):
        """폴더 내 모든 DB 파일 비동기 처리"""
        def log(msg: str):
            if queue:
                try:
                    queue.put(f"{msg}\n")
                except Exception:
                    print(f"[Log] {msg}")
            else:
                print(msg)
        
        log(f"폴더 처리 시작: {folder_path}")
        start_time = time.time()
        
        try:
            # DB 파일 검색 및 정렬
            db_files = await self._find_and_sort_db_files(folder_path)
            if not db_files:
                log("처리할 DB 파일이 없습니다.")
                if queue: queue.put("===API_PROCESS_FAILED===")
                return
            
            log(f"발견된 DB 파일: {len(db_files)}개")
            
            # 각 DB 파일 순차 처리
            for db_index, (db_file, _) in enumerate(db_files, 1):
                await self._process_single_db(db_file, db_index, log)
            
            # 완료 로그
            end_time = time.time()
            tokens = self.openai_client.total_tokens
            log("\n=== 모든 DB 파일 처리 완료 ===")
            log(f"총 소요 시간: {end_time - start_time:.2f}초")
            log(f"전체 토큰 사용량: Prompt={tokens['prompt']}, Completion={tokens['completion']}, Total={tokens['total']}")
            
            if queue: queue.put("===API_PROCESS_COMPLETED===")
            
        except Exception as e:
            log(f"폴더 처리 중 심각한 오류: {e}")
            if queue: queue.put("===API_PROCESS_FAILED===")
    
    async def _find_and_sort_db_files(self, folder_path: str) -> List[Tuple[str, int]]:
        """DB 파일 찾기 및 날짜순 정렬"""
        try:
            db_files = glob.glob(os.path.join(folder_path, '*.db'))
            db_info_list = [
                (db_file, extract_date_from_filename(os.path.basename(db_file)) or 0)
                for db_file in db_files
            ]
            db_info_list.sort(key=lambda x: x[1], reverse=True)
            return db_info_list
        except Exception as e:
            logger.error(f"DB 파일 검색 중 오류: {e}")
            return []
    
    async def _process_single_db(self, db_file: str, exam_session: int, log_func):
        """단일 DB 파일 처리"""
        db_name = os.path.basename(db_file)
        log_func(f"\n--- DB 파일 처리 시작: {db_name} ---")
        
        try:
            async with aiosqlite.connect(db_file) as conn:
                # 스키마 업데이트
                await self._update_db_schema(conn, exam_session, log_func)
                
                # 처리할 문제들 조회
                questions = await self._fetch_unprocessed_questions(conn)
                if not questions:
                    log_func("해설이 필요한 새로운 문제가 없습니다.")
                    return
                
                log_func(f"총 {len(questions)}개 문제 해설 생성 시작")
                
                # 청크로 분할
                chunks = chunk_list(questions, self.config.chunk_size)
                log_func(f"총 {len(chunks)}개 청크로 분할")
                
                # 데이터 준비 (OCR 병렬 처리)
                log_func("데이터 준비 중 (OCR 병렬 처리)...")
                prepared_chunks = await self._prepare_all_chunks(chunks)
                
                # GPT 호출 (동시 처리)
                log_func(f"GPT 해설 생성 시작 (최대 동시 요청: {self.config.max_concurrent_tasks})")
                explanations_results = await self._generate_explanations_for_chunks(prepared_chunks, db_name)
                
                # DB 업데이트
                log_func("데이터베이스 업데이트 중...")
                await self._update_database_with_results(conn, prepared_chunks, explanations_results, log_func)
                
                log_func(f"DB 파일 처리 완료: {db_name}")
                
        except Exception as e:
            log_func(f"DB 파일 처리 중 오류 [{db_file}]: {e}")
            logger.exception("상세 오류 정보:")
    
    async def _update_db_schema(self, conn: aiosqlite.Connection, exam_session: int, log_func):
        """DB 스키마 업데이트"""
        columns_to_add = ["ExamSession", "Answer_description", "Category"]
        
        for col_name in columns_to_add:
            try:
                await conn.execute(f"ALTER TABLE questions ADD COLUMN {col_name} TEXT;")
            except aiosqlite.OperationalError:
                pass  # 이미 존재하는 컬럼
        
        await conn.execute("UPDATE questions SET ExamSession = ?", (str(exam_session),))
        await conn.commit()
        log_func("스키마 업데이트 완료")
    
    async def _fetch_unprocessed_questions(self, conn: aiosqlite.Connection) -> List[sqlite3.Row]:
        """해설이 없는 문제들 조회"""
        async with conn.execute("""
            SELECT Question_id, Big_Question, Question, Option1, Option2, Option3, Option4, Correct_Option
            FROM questions 
            WHERE Answer_description IS NULL OR Answer_description = ''
        """) as cursor:
            return await cursor.fetchall()
    
    async def _prepare_all_chunks(self, chunks: List[List]) -> List[List[Dict]]:
        """모든 청크의 데이터를 병렬로 준비"""
        tasks = [self._prepare_single_chunk(chunk) for chunk in chunks]
        return await asyncio.gather(*tasks)
    
    async def _prepare_single_chunk(self, chunk_questions: List) -> List[Dict]:
        """단일 청크 데이터 준비 (OCR 포함)"""
        prepared_data = []
        
        # 각 문제에 대해 OCR 작업 수집
        ocr_tasks = []
        question_info = []
        
        for row in chunk_questions:
            question_id = row[0]
            big_question = row[1]
            question = row[2]
            options = [row[i] for i in range(3, 7)]
            correct_option = row[7]
            
            # OCR이 필요한 필드들 확인
            ocr_fields = {}
            
            # 질문 필드 처리
            if is_blob_image(question):
                task = self.ocr_processor.extract_text_from_blob(question, f"Q{question_id}-Question")
                ocr_tasks.append(task)
                ocr_fields['question'] = len(ocr_tasks) - 1
            else:
                ocr_fields['question'] = safely_decode_text(question, question_id)
            
            # 선택지 필드들 처리
            for i, option in enumerate(options, 1):
                if is_blob_image(option):
                    task = self.ocr_processor.extract_text_from_blob(option, f"Q{question_id}-Option{i}")
                    ocr_tasks.append(task)
                    ocr_fields[f'option{i}'] = len(ocr_tasks) - 1
                else:
                    ocr_fields[f'option{i}'] = safely_decode_text(option, question_id)
            
            question_info.append({
                'question_id': question_id,
                'big_question': big_question,
                'correct_option': correct_option,
                'ocr_fields': ocr_fields
            })
        
        # 모든 OCR 작업을 동시에 실행
        if ocr_tasks:
            ocr_results = await asyncio.gather(*ocr_tasks, return_exceptions=True)
        else:
            ocr_results = []
        
        # 결과를 다시 매핑하여 최종 데이터 구성
        for info in question_info:
            question_text_parts = []
            
            # Big_Question 처리
            big_q_text = safely_decode_text(info['big_question'], info['question_id'])
            if big_q_text:
                question_text_parts.append(big_q_text)
            
            # Question 처리
            question_field = info['ocr_fields']['question']
            if isinstance(question_field, int):
                # OCR 결과
                q_text = ocr_results[question_field] if not isinstance(ocr_results[question_field], Exception) else ""
            else:
                # 직접 텍스트
                q_text = question_field
            
            if q_text:
                question_text_parts.append(q_text)
            
            full_question_text = "\n".join(question_text_parts).strip()
            
            # 선택지 처리
            options_text = []
            for i in range(1, 5):
                option_field = info['ocr_fields'][f'option{i}']
                if isinstance(option_field, int):
                    # OCR 결과
                    opt_text = ocr_results[option_field] if not isinstance(ocr_results[option_field], Exception) else ""
                else:
                    # 직접 텍스트
                    opt_text = option_field
                options_text.append(opt_text)
            
            # 모든 옵션이 유효한지 확인
            is_valid = all(opt.strip() for opt in options_text)
            
            prepared_data.append({
                'question_id': info['question_id'],
                'is_valid': is_valid,
                'question_text': full_question_text,
                'option1': options_text[0],
                'option2': options_text[1],
                'option3': options_text[2],
                'option4': options_text[3],
                'correct_option': info['correct_option']
            })
        
        return prepared_data
    
    async def _generate_explanations_for_chunks(self, prepared_chunks: List[List[Dict]], db_name: str) -> List[Tuple]:
        """모든 청크에 대해 GPT 해설 생성"""
        semaphore = asyncio.Semaphore(self.config.max_concurrent_tasks)
        
        tasks = []
        for i, chunk_data in enumerate(prepared_chunks):
            valid_data = [item for item in chunk_data if item['is_valid']]
            if valid_data:
                chunk_id = f"{db_name}-{i+1}"
                task = self.openai_client.generate_explanations_batch(valid_data, chunk_id, semaphore)
                tasks.append(task)
            else:
                tasks.append(asyncio.coroutine(lambda: ([], None))())
        
        return await asyncio.gather(*tasks)
    
    async def _update_database_with_results(
        self, 
        conn: aiosqlite.Connection, 
        prepared_chunks: List[List[Dict]], 
        explanations_results: List[Tuple], 
        log_func
    ):
        """데이터베이스에 결과 업데이트"""
        updated_count = 0
        
        # 옵션 누락 문제 처리
        for chunk_data in prepared_chunks:
            for item in chunk_data:
                if not item['is_valid']:
                    await conn.execute(
                        "UPDATE questions SET Answer_description = ? WHERE Question_id = ?",
                        (f"정답은 {item['correct_option']}번 입니다", item['question_id'])
                    )
        
        # GPT 생성 해설 업데이트
        for explanations, usage in explanations_results:
            if explanations:
                for explanation_item in explanations:
                    question_id = explanation_item.get('question_id')
                    explanation = explanation_item.get('explanation', '')
                    if question_id:
                        await conn.execute(
                            "UPDATE questions SET Answer_description = ? WHERE Question_id = ?",
                            (explanation, question_id)
                        )
                        updated_count += 1
        
        await conn.commit()
        log_func(f"데이터베이스 업데이트 완료: {updated_count}개 해설 저장")

# ============================================================================
# 공개 API (기존 호환성 유지)
# ============================================================================

async def process_api_folder(folder_path: str, queue=None):
    """기존 호환성을 위한 비동기 래퍼 함수"""
    try:
        config = load_config()
        processor = AsyncProAPIProcessor(config)
        await processor.process_folder(folder_path, queue)
    except Exception as e:
        error_msg = f"처리 중 오류 발생: {e}"
        if queue:
            queue.put(f"{error_msg}\n")
            queue.put("===API_PROCESS_FAILED===")
        else:
            print(error_msg)

# ============================================================================
# 메인 실행부
# ============================================================================

async def main():
    """비동기 메인 함수"""
    if len(sys.argv) > 1:
        folder_to_process = sys.argv[1]
    else:
        folder_to_process = input("처리할 DB 파일들이 있는 폴더 경로를 입력하세요: ")

    if os.path.isdir(folder_to_process):
        await process_api_folder(folder_to_process)
    else:
        print(f"오류: '{folder_to_process}'는 유효한 디렉토리가 아닙니다.")

if __name__ == "__main__":
    # Windows에서 asyncio 이벤트 루프 정책 설정
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    # 비동기 메인 함수 실행
    asyncio.run(main())