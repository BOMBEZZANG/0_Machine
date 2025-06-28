# 5_OX_Quiz_Creating.py

import os
import sys
import json
import logging
import re
from pathlib import Path
from openai import OpenAI
import time
import sqlite3

# --- 로깅 설정 ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', stream=sys.stdout)
logger = logging.getLogger(__name__)


class Config:
    """설정 파일을 관리하는 클래스"""
    def __init__(self, config_path):
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"FATAL: 설정 파일 읽기 오류 '{config_path}': {e}")
            sys.exit(1)
        
        self.api_key = config_data.get("OPENAI_API_KEY")
        self.model_name = config_data.get("MODEL_NAME", "gpt-4o")
        
        if not self.api_key:
            logger.error("FATAL: config.json 파일에 OPENAI_API_KEY가 없습니다.")
            sys.exit(1)
        logger.info(f"설정 로드 완료: {config_path}")


class OxQuizGenerator:
    """1.txt 파일들을 기반으로 OX 퀴즈를 생성하고, JSON 및 DB 파일로 저장하는 클래스"""

    def __init__(self, config: Config, selected_folder: str):
        self.client = OpenAI(api_key=config.api_key)
        self.model = config.model_name
        self.base_path = Path(selected_folder)
        # 최종 산출물 경로 정의
        self.final_json_path = self.base_path / "assets" / "OX.json"
        self.final_db_path = self.base_path / "assets" / "quiz.db"
        self.all_quiz_data = []
        self.has_errors = False

    def _parse_quiz_response(self, response_text: str) -> list:
        """
        GPT 응답 텍스트에서 JSON 배열을 파싱합니다. (디버깅 강화 버전)
        """
        json_text = None
        try:
            match = re.search(r'```json\s*([\s\S]*?)\s*```', response_text, re.DOTALL)
            if match:
                json_text = match.group(1).strip()
                logger.info("JSON 코드 블록을 감지하여 파싱을 시도합니다.")
            else:
                match = re.search(r'\[\s*{[\s\S]*}*\s*\]', response_text, re.DOTALL)
                if match:
                    json_text = match.group(0)
                    logger.info("JSON 배열 패턴을 감지하여 파싱을 시도합니다.")

            if json_text:
                return json.loads(json_text)
            else:
                logger.warning("응답에서 유효한 JSON 배열이나 코드 블록을 찾지 못했습니다.")
                logger.warning("아래는 API로부터 받은 전체 응답 내용입니다. 응답 형식이 왜 잘못되었는지 확인하세요.")
                logger.warning(f"--- API 전체 응답 시작 ---\n{response_text}\n--- API 전체 응답 끝 ---")
                return []

        except json.JSONDecodeError as e:
            logger.error(f"JSON 파싱 중 오류가 발생했습니다: {e}")
            logger.error("파싱하려던 텍스트가 완전한 JSON 형식이 아닐 가능성이 높습니다.")
            if json_text:
                logger.error(f"--- 파싱 시도한 텍스트 시작 ---\n{json_text}\n--- 파싱 시도한 텍스트 끝 ---")
            else:
                 logger.error(f"--- 전체 원본 응답 (일부) ---\n{response_text[:1000]}...\n--- 전체 원본 응답 (일부) 끝 ---")
            return []
        except Exception as e:
            logger.error(f"퀴즈 응답 파싱 중 예상치 못한 예외 발생: {e}")
            logger.error(f"--- 전체 원본 응답 (일부) ---\n{response_text[:1000]}...\n--- 전체 원본 응답 (일부) 끝 ---")
            return []

    def _generate_quiz_for_category(self, category_path: Path):
        """단일 카테고리의 1.txt 파일을 읽어 OX 퀴즈를 생성합니다."""
        category_name = category_path.name
        txt_file_path = category_path / "1.txt"

        if not txt_file_path.exists():
            logger.warning(f"파일을 찾을 수 없어 건너뜁니다: {txt_file_path}")
            return

        logger.info(f"--- '{category_name}' 카테고리 처리 시작 ---")
        
        with open(txt_file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        system_prompt = "당신은 주어진 자료를 바탕으로 OX 퀴즈를 만드는 전문가입니다. 반드시 요청받은 형식과 개수에 맞춰서만 응답해야 합니다."
        
        user_prompt = f"""
        아래 텍스트는 '{category_name}' 과목에 대한 내용입니다. 이 내용을 참고하여 OX 퀴즈 100개를 생성해 주세요.
        ### 중요 규칙
        - 각 문제는 명확하게 참(O) 또는 거짓(X)으로 판별할 수 있어야 합니다.
        - 해설은 왜 정답이 O 또는 X인지 구체적인 근거를 들어 간결하게 설명해야 합니다.
        - 출력은 오직 JSON 배열 형식이어야 하며, 다른 부가 설명이나 텍스트를 포함해서는 안 됩니다.
        ### JSON 출력 형식
        각 퀴즈 문제는 반드시 아래와 같은 키(key)를 가진 JSON 객체로 만들어 주세요.
        {{
          "Question_id": <문제 번호 (나중에 재지정되므로 1부터 시작)>,
          "Big_Question": "<O/X 형식의 문제 서술문>",
          "Option1": "O",
          "Option2": "X",
          "Correct_Option": <1(O) 또는 2(X)>,
          "Category": "{category_name}",
          "Answer_description": "<정답에 대한 구체적인 해설>"
        }}
        ---
        ### 참고 텍스트
        {content[:100000]} 
        ---
        이제 위의 규칙과 참고 텍스트를 바탕으로 '{category_name}'에 대한 OX 퀴즈 100개를 생성해 주세요.
        """

        try:
            logger.info(f"'{category_name}'에 대한 API 요청 전송... (모델: {self.model})")
            # ------------------ ▼▼▼ 수정된 부분 ▼▼▼ ------------------
            # 생성할 답변의 최대 길이를 16000 토큰으로 늘려서 공간 부족 문제를 해결합니다.
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_completion_tokens=16000, 
            )
            # ------------------ ▲▲▲ 수정된 부분 ▲▲▲ ------------------
            
            if response.choices:
                finish_reason = response.choices[0].finish_reason
                logger.info(f"API 응답 수신. 생성 중단 사유(finish_reason): {finish_reason}")
                
                if finish_reason == 'content_filter':
                    logger.error("❌ API가 '콘텐츠 필터'에 의해 응답 생성을 중단했습니다.")
                    logger.error("   입력된 텍스트('1.txt')에 OpenAI의 정책을 위반하는 내용이 포함되었을 수 있습니다.")
                    self.has_errors = True
                    return

                response_text = response.choices[0].message.content
            else:
                logger.error("❌ API 응답에 'choices' 배열이 비어 있습니다.")
                response_text = ""

            if response_text is None:
                logger.warning("API 응답의 message.content가 'None'입니다. 빈 문자열로 처리합니다.")
                response_text = ""
            
            quiz_items = self._parse_quiz_response(response_text)
            
            if quiz_items:
                for item in quiz_items:
                    item['Category'] = category_name
                self.all_quiz_data.extend(quiz_items)
                logger.info(f"✅ '{category_name}' 카테고리에서 {len(quiz_items)}개의 OX 퀴즈를 성공적으로 생성했습니다.")
            else:
                logger.error(f"❌ '{category_name}' 카테고리에 대한 OX 퀴즈 생성에 실패했습니다. (API가 유효한 JSON을 반환하지 않음)")
                self.has_errors = True

        except Exception as e:
            logger.error(f"'{category_name}' 처리 중 API 오류 발생: {e}")
            self.has_errors = True

    def _save_to_json(self):
        """취합된 모든 퀴즈 데이터를 JSON 파일로 저장합니다."""
        logger.info(f"생성된 퀴즈를 JSON 파일로 저장합니다: {self.final_json_path}")
        try:
            self.final_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.final_json_path, 'w', encoding='utf-8') as f:
                json.dump(self.all_quiz_data, f, ensure_ascii=False, indent=2)
            logger.info("JSON 파일 저장이 완료되었습니다.")
        except Exception as e:
            logger.error(f"JSON 파일 저장 중 오류 발생: {e}")
            self.has_errors = True

    def _save_to_db(self):
        """취합된 모든 퀴즈 데이터를 SQLite DB 파일로 저장합니다."""
        logger.info(f"생성된 퀴즈를 SQLite DB 파일로 저장합니다: {self.final_db_path}")
        try:
            self.final_db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self.final_db_path))
            cursor = conn.cursor()

            cursor.execute("DROP TABLE IF EXISTS OX_Quiz")
            
            cursor.execute("""
            CREATE TABLE OX_Quiz (
                Question_id INTEGER PRIMARY KEY,
                Big_Question TEXT NOT NULL,
                Option1 TEXT NOT NULL,
                Option2 TEXT NOT NULL,
                Correct_Option INTEGER NOT NULL,
                Category TEXT,
                Answer_description TEXT
            )
            """)

            to_insert = []
            for quiz in self.all_quiz_data:
                to_insert.append((
                    quiz.get('Question_id'),
                    quiz.get('Big_Question'),
                    quiz.get('Option1', 'O'),
                    quiz.get('Option2', 'X'),
                    quiz.get('Correct_Option'),
                    quiz.get('Category'),
                    quiz.get('Answer_description')
                ))

            cursor.executemany("""
            INSERT INTO OX_Quiz (Question_id, Big_Question, Option1, Option2, Correct_Option, Category, Answer_description)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """, to_insert)

            conn.commit()
            conn.close()
            logger.info("SQLite DB 파일 저장이 완료되었습니다.")
        except sqlite3.Error as e:
            logger.error(f"SQLite DB 작업 중 오류 발생: {e}")
            self.has_errors = True
        except Exception as e:
            logger.error(f"DB 파일 저장 중 예상치 못한 오류 발생: {e}")
            self.has_errors = True

    def run_generation(self):
        """전체 퀴즈 생성 및 저장 프로세스를 실행합니다."""
        logger.info("OX 퀴즈 생성을 시작합니다...")
        
        source_path = self.base_path / "output"

        if not source_path.exists() or not source_path.is_dir():
            logger.error(f"퀴즈 소스 경로를 찾을 수 없습니다: {source_path}")
            logger.error(f"'{self.base_path}' 폴더 내에 'output' 폴더가 존재하는지 확인해주세요.")
            return False

        category_folders = [d for d in source_path.iterdir() if d.is_dir()]

        if not category_folders:
            logger.error(f"'{source_path}' 폴더에서 퀴즈를 생성할 카테고리 폴더를 찾을 수 없습니다.")
            return False

        for category_path in category_folders:
            self._generate_quiz_for_category(category_path)
            time.sleep(1) 

        if not self.all_quiz_data:
            logger.error("생성된 퀴즈 데이터가 없습니다. 프로세스를 중단합니다.")
            return False

        logger.info(f"총 {len(self.all_quiz_data)}개의 퀴즈를 생성했습니다. 최종 파일 저장을 시작합니다.")

        for i, quiz in enumerate(self.all_quiz_data):
            quiz['Question_id'] = i + 1

        self._save_to_json()
        self._save_to_db()

        if self.has_errors:
            logger.warning("일부 카테고리에서 오류가 발생했지만, 성공한 데이터로 파일을 생성했습니다.")
            return True 
        
        logger.info("모든 작업이 성공적으로 완료되었습니다.")
        return True

def main():
    """스크립트 실행을 위한 메인 함수"""
    if len(sys.argv) < 2:
        print("사용법: python 5_OX_Quiz_Creating.py <selected_folder_path>")
        sys.exit(1)
    
    selected_folder = sys.argv[1]
    
    script_directory = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_directory, "config.json")
    
    if not os.path.exists(config_path):
        logger.error(f"설정 파일을 찾을 수 없습니다: {config_path}")
        sys.exit(1)

    config = Config(config_path)
    generator = OxQuizGenerator(config, selected_folder)
    
    success = generator.run_generation()
    if not success:
        logger.error("퀴즈 생성 및 변환 과정에서 하나 이상의 오류가 발생했습니다.")
        sys.exit(1)

if __name__ == "__main__":
    main()