import asyncio
import aiosqlite
import sqlite3
import os
import glob
import re
import sys
import subprocess
import logging
from datetime import datetime
from google.cloud import texttospeech
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# [수정] 클라이언트 초기화를 전역에서 제거

# --- 유틸리티 함수 ---
def get_db_files_to_process(db_folder: str) -> list:
    """처리할 DB 파일 목록을 정해진 규칙에 따라 정렬하고 상위 7개를 반환합니다."""
    all_dbs = [f for f in os.listdir(db_folder) if f.endswith('.db')]
    db_with_metadata = []
    
    for db in all_dbs:
        if match := re.match(r"question(\d+)\.db", db):
            number = int(match.group(1))
            db_with_metadata.append({'type': 'question', 'sort_key': number, 'file': db})
        elif match := re.search(r'(\d{8})', db):
            date_str = match.group(1)
            db_with_metadata.append({'type': 'date', 'sort_key': int(date_str), 'file': db})

    db_with_metadata.sort(key=lambda x: (0 if x['type'] == 'question' else 1, -x['sort_key']))
    return [os.path.join(db_folder, item['file']) for item in db_with_metadata[:7]]

def compress_audio(input_file: str):
    """FFmpeg를 사용해 오디오 파일을 압축합니다. (동기 함수)"""
    if not os.path.exists(input_file): return
    output_file_compressed = input_file.replace(".mp3", "_compressed.mp3")
    try:
        subprocess.run(
            ["ffmpeg", "-i", input_file, "-b:a", "32k", "-ar", "16000", "-y", "-loglevel", "error", output_file_compressed],
            check=True
        )
        os.remove(input_file)
        os.rename(output_file_compressed, input_file)
    except Exception as e:
        logger.error(f"오디오 압축 실패 {input_file}: {e}")

# --- 비동기 TTS 및 DB 처리 ---
# [수정] tts_client를 인자로 받도록 변경
async def text_to_speech_async(tts_client: texttospeech.TextToSpeechAsyncClient, text: str, output_file: str, semaphore: asyncio.Semaphore):
    """단일 텍스트를 비동기적으로 MP3로 변환 및 압축합니다."""
    async with semaphore:
        try:
            synthesis_input = texttospeech.SynthesisInput(text=text)
            voice = texttospeech.VoiceSelectionParams(language_code="ko-KR", name="ko-KR-Standard-D")
            audio_config = texttospeech.AudioConfig(audio_encoding=texttospeech.AudioEncoding.MP3, speaking_rate=0.9, pitch=-4.0)

            response = await tts_client.synthesize_speech(input=synthesis_input, voice=voice, audio_config=audio_config)
            
            with open(output_file, "wb") as out:
                out.write(response.audio_content)
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, compress_audio, output_file)
            
            logger.info(f"음성 파일 생성 및 압축 완료: {os.path.basename(output_file)}")
        except Exception as e:
            logger.error(f"TTS 변환/압축 실패 {os.path.basename(output_file)}: {e}")

async def add_audio_column_if_not_exists(conn: aiosqlite.Connection):
    """'audio' 칼럼이 없으면 추가합니다."""
    async with conn.cursor() as cursor:
        await cursor.execute("PRAGMA table_info(questions)")
        columns = [col[1] for col in await cursor.fetchall()]
        if "audio" not in columns:
            await cursor.execute("ALTER TABLE questions ADD COLUMN audio TEXT")
            logger.info("'audio' 칼럼이 추가되었습니다.")
    await conn.commit()

async def update_audio_paths_in_db(conn: aiosqlite.Connection, output_dir: str, question_index: int):
    """생성된 오디오 파일 경로를 DB에 업데이트합니다."""
    logger.info("DB에 오디오 경로 업데이트 시작...")
    updated_count = 0
    async with conn.cursor() as cursor:
        await cursor.execute("SELECT Question_id FROM questions")
        rows = await cursor.fetchall()
        
        for row in rows:
            question_id = row[0]
            audio_file = os.path.join(output_dir, f"question_{question_id}.mp3")
            if os.path.exists(audio_file):
                relative_path = f"assets/audio/question{question_index}/question_{question_id}.mp3"
                await cursor.execute(
                    "UPDATE questions SET audio = ? WHERE Question_id = ?",
                    (relative_path, question_id)
                )
                updated_count += 1
    
    await conn.commit()
    logger.info(f"총 {updated_count}개의 오디오 경로를 DB에 업데이트했습니다.")

# [수정] tts_client를 인자로 받도록 변경
async def process_single_db(tts_client: texttospeech.TextToSpeechAsyncClient, db_path: str, index: int, audio_base_dir: str):
    """단일 DB 파일에 대해 음성 생성 및 경로 업데이트를 모두 수행합니다."""
    db_name = os.path.basename(db_path)
    logger.info(f"\n--- [{index}] DB 처리 시작: {db_name} ---")

    output_dir_name = f"question{index}"
    output_dir = os.path.join(audio_base_dir, output_dir_name)
    os.makedirs(output_dir, exist_ok=True)
    
    conn = None
    try:
        conn = await aiosqlite.connect(db_path)
        
        await add_audio_column_if_not_exists(conn)
        
        cursor = await conn.execute("SELECT Question_id, Big_Question, Question, Option1, Option2, Option3, Option4, Correct_Option, Answer_description FROM questions")
        rows = await cursor.fetchall()
        await cursor.close()
        
        if not rows:
            logger.warning(f"{db_name}에 처리할 데이터가 없습니다.")
            return

        tasks = []
        semaphore = asyncio.Semaphore(10)

        for row in rows:
            question_id, big_question, question, opt1, opt2, opt3, opt4, correct_opt, answer_desc = row
            
            parts = [f"{question_id}번. "]
            if isinstance(big_question, str): parts.append(f"{big_question}. ")
            if isinstance(question, str): parts.append(f"{question}. ")
            if isinstance(opt1, str): parts.append(f"일. {opt1}. ")
            if isinstance(opt2, str): parts.append(f"이. {opt2}. ")
            if isinstance(opt3, str): parts.append(f"삼. {opt3}. ")
            if isinstance(opt4, str): parts.append(f"사. {opt4}. ")
            if isinstance(correct_opt, int): parts.append(f"정답은 {correct_opt}번 입니다. ")
            if isinstance(answer_desc, str):
                cleaned_desc = re.sub(r"따라서 정답은 [1-4]입니다\.", "", answer_desc).strip()
                if cleaned_desc: parts.append(f"{cleaned_desc}. ")
            
            audio_text = "".join(parts)
            
            if audio_text.strip() != f"{question_id}번.":
                output_file = os.path.join(output_dir, f"question_{question_id}.mp3")
                # [수정] tts_client를 인자로 전달
                tasks.append(text_to_speech_async(tts_client, audio_text, output_file, semaphore))

        if not tasks:
            logger.warning(f"{db_name}에서 음성으로 변환할 텍스트가 없습니다.")
            return

        logger.info(f"{db_name}: 총 {len(tasks)}개 문제의 음성 파일 동시 생성 시작...")
        await asyncio.gather(*tasks)
        logger.info(f"{db_name}: 모든 음성 파일 생성 완료.")
        
        await update_audio_paths_in_db(conn, output_dir, index)

    except Exception as e:
        logger.error(f"{db_name} 처리 중 오류 발생: {e}", exc_info=True)
    finally:
        if conn:
            await conn.close()
            logger.info(f"DB 연결 종료: {db_name}")

async def main(db_folder: str):
    """메인 실행 함수"""
    # [수정] main 함수 안에서 클라이언트 초기화
    try:
        tts_client = texttospeech.TextToSpeechAsyncClient()
        logger.info("Google Cloud TTS 비동기 클라이언트 초기화 성공")
    except Exception as e:
        logger.error(f"Google Cloud TTS 클라이언트 초기화 실패: {e}")
        return

    logger.info("="*50)
    logger.info("오디오 생성 및 DB 업데이트 프로세스 시작")
    logger.info("="*50)
    
    start_time = time.time()
    
    target_dbs = get_db_files_to_process(db_folder)
    if not target_dbs:
        logger.warning(f"폴더에 처리할 DB 파일이 없습니다: {db_folder}")
        return

    logger.info(f"처리 대상 DB 파일 ({len(target_dbs)}개): {[os.path.basename(p) for p in target_dbs]}")
    
    audio_base_dir = os.path.join(db_folder, "audio")
    os.makedirs(audio_base_dir, exist_ok=True)
    
    for i, db_path in enumerate(target_dbs, 1):
        # [수정] 생성된 tts_client를 인자로 전달
        await process_single_db(tts_client, db_path, i, audio_base_dir)

    end_time = time.time()
    logger.info("\n" + "="*50)
    logger.info(f"모든 작업 완료. 총 소요 시간: {end_time - start_time:.2f}초")
    logger.info("="*50)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        folder_path = sys.argv[1]
        if os.path.isdir(folder_path):
            if sys.platform == "win32":
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            asyncio.run(main(folder_path))
        else:
            print(f"오류: '{folder_path}'는 유효한 디렉토리가 아닙니다.")
    else:
        print("사용법: python 3_description_audio.py <DB 파일들이 있는 폴더 경로>")