import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import os
import re
import threading
import sqlite3
from typing import List, Dict, Tuple, Optional
import sys
from pathlib import Path
import subprocess

try:
    from pdf_processor import PDFProcessor
except ImportError:
    print("pdf_processor.py 파일이 필요합니다. 이 파일과 같은 폴더에 있어야 합니다.")
    sys.exit(1)

# ★★★★★ 사용자 설정 필요 ★★★★★
DB_BROWSER_PATH = "/Applications/DB Browser for SQLite.app/Contents/MacOS/DB Browser for SQLite"


class PDFBatchProcessor:
    def __init__(self, root):
        self.root = root
        self.root.title("PDF Batch Processor - DB 생성 도구")
        self.root.geometry("800x600")
        
        self.default_folder = os.path.expanduser("~/Desktop/Apps/qcjongmin/appauto/raw_DB/rawdbs/")
        self.selected_folder = ""
        
        self.total_files = 0
        self.processed_files = 0
        self.failed_files = []
        self.debug_issues = []
        self.question_number_issues = []
        self.first_db_last_number = None
        self.db_last_numbers = []
        
        self.issue_files_to_correct = []
        self.current_correction_index = -1
        self.opened_processes = []
        
        self.setup_ui()
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        folder_frame = ttk.LabelFrame(main_frame, text="폴더 선택", padding="5")
        folder_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        self.folder_button = ttk.Button(folder_frame, text="Folder Selected", command=self.select_folder)
        self.folder_button.grid(row=0, column=0, padx=(0, 10))
        
        self.folder_label = ttk.Label(folder_frame, text="폴더를 선택해주세요", foreground="gray")
        self.folder_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=2, pady=(0, 10))
        
        self.process_button = ttk.Button(control_frame, text="PDF -> DB 변환 시작", command=self.start_processing, state="disabled")
        self.process_button.pack(side=tk.LEFT, padx=(0, 10))
        
        self.stop_button = ttk.Button(control_frame, text="중단", command=self.stop_processing, state="disabled")
        self.stop_button.pack(side=tk.LEFT)
        
        progress_frame = ttk.LabelFrame(main_frame, text="진행상황", padding="5")
        progress_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(0, 10))
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, length=300)
        self.progress_bar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        
        self.progress_label = ttk.Label(progress_frame, text="대기 중...")
        self.progress_label.grid(row=1, column=0, columnspan=2, pady=(0, 5))
        
        log_frame = ttk.LabelFrame(main_frame, text="처리 로그", padding="5")
        log_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.correction_frame = ttk.LabelFrame(main_frame, text="디버그 수정 모드", padding="5")
        
        self.correction_label = ttk.Label(self.correction_frame, text="수정할 파일: 대기 중...", foreground="blue", font=("Arial", 10, "bold"))
        self.correction_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        
        self.next_button = ttk.Button(self.correction_frame, text="다음 이슈 파일 열기", command=self.next_correction_file)
        self.next_button.grid(row=0, column=1, padx=5, pady=5)
        
        self.finish_button = ttk.Button(self.correction_frame, text="수정 모드 종료", command=self.finish_correction_mode)
        self.finish_button.grid(row=0, column=2, padx=5, pady=5)
        
        
        # [추가] 카테고리 업데이트 버튼
        self.category_update_button = ttk.Button(main_frame, text="선택된 폴더의 모든 DB에 카테고리 정보 업데이트", command=self.start_category_update_process)
        self.category_update_button.grid(row=5, column=0, columnspan=2, pady=10, sticky=tk.W)

        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(3, weight=1)
        folder_frame.columnconfigure(1, weight=1)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.correction_frame.columnconfigure(0, weight=1)
        
        self.stop_flag = False

    def select_folder(self):
        initial_dir = self.default_folder if os.path.exists(self.default_folder) else "/"
        folder_path = filedialog.askdirectory(title="PDF 파일들이 있는 폴더를 선택하세요", initialdir=initial_dir)
        
        if folder_path:
            self.selected_folder = folder_path
            self.folder_label.config(text=folder_path, foreground="black")
            self.process_button.config(state="normal")
            self.log_message(f"폴더 선택됨: {folder_path}")
            pdf_files = self.get_pdf_files(folder_path)
            self.log_message(f"발견된 PDF 파일 수: {len(pdf_files)}개")
            
    def get_pdf_files(self, folder_path: str) -> List[str]:
        pdf_files = []
        pattern = re.compile(r"^.*\d{8}\(교사용\)\.pdf$")
        try:
            for filename in os.listdir(folder_path):
                if filename.lower().endswith('.pdf') and pattern.match(filename):
                    pdf_files.append(os.path.join(folder_path, filename))
        except Exception as e:
            self.log_message(f"폴더 스캔 중 오류: {e}")
        return sorted(pdf_files)
        
    def extract_db_name(self, pdf_filename: str) -> str:
        basename = os.path.basename(pdf_filename)
        match = re.search(r"(\d{8})\(교사용\)\.pdf$", basename)
        if match:
            return f"{match.group(1)}.db"
        return f"{Path(pdf_filename).stem}.db"
        
    def start_processing(self):
        if not self.selected_folder:
            messagebox.showerror("오류", "폴더를 먼저 선택해주세요.")
            return
            
        self.stop_flag = False
        self.failed_files = []
        self.debug_issues = []
        self.question_number_issues = []
        self.first_db_last_number = None
        self.db_last_numbers = []
        self.processed_files = 0
        
        self.process_button.config(state="disabled")
        self.stop_button.config(state="normal")
        self.folder_button.config(state="disabled")
        
        self.processing_thread = threading.Thread(target=self.process_all_pdfs)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
    def stop_processing(self):
        self.stop_flag = True
        self.log_message("처리 중단 요청됨...")

    def process_all_pdfs(self):
        try:
            pdf_files = self.get_pdf_files(self.selected_folder)
            self.total_files = len(pdf_files)
            
            if self.total_files == 0:
                self.log_message("처리할 PDF 파일이 없습니다.")
                self.reset_ui_state()
                return
                
            self.log_message(f"총 {self.total_files}개 PDF 파일 처리 시작")
            
            processor = PDFProcessor()
            
            for i, pdf_path in enumerate(pdf_files):
                if self.stop_flag:
                    self.log_message("사용자에 의해 처리가 중단되었습니다.")
                    break
                try:
                    self.process_single_pdf(processor, pdf_path, i + 1)
                except Exception as e:
                    error_msg = f"파일 처리 중 오류: {os.path.basename(pdf_path)} - {e}"
                    self.log_message(error_msg)
                    self.failed_files.append((pdf_path, str(e)))
                self.processed_files += 1
                self.update_progress()
                
            self.finalize_processing()
        except Exception as e:
            self.log_message(f"전체 처리 중 오류: {e}")
        finally:
            if not (self.debug_issues or self.question_number_issues):
                self.root.after(100, self.reset_ui_state)

    def process_single_pdf(self, processor: PDFProcessor, pdf_path: str, file_num: int):
        filename = os.path.basename(pdf_path)
        self.log_message(f"[{file_num}/{self.total_files}] 처리 시작: {filename}")
        
        db_name = self.extract_db_name(pdf_path)
        db_path = os.path.join(self.selected_folder, db_name)
        
        if os.path.exists(db_path):
            os.remove(db_path)
            self.log_message(f"기존 DB 파일 삭제: {db_name}")
            
        match = re.search(r"^(.*?)\d{8}\(교사용\)\.pdf$", filename)
        exam_category = match.group(1) if match else "Unknown"
        
        try:
            debug_result = processor.process_pdf(pdf_path, db_path, exam_category)
            
            if 'debug_output' in debug_result:
                debug_lines = debug_result['debug_output'].strip().split('\n')
                for line in debug_lines:
                    if line.strip():
                        self.log_message(f"    {line}")
            
            last_number = debug_result.get('last_question_number', 0)
            self.db_last_numbers.append({'filename': filename, 'db_name': db_name, 'last_number': last_number})
            
            if self.first_db_last_number is None:
                self.first_db_last_number = last_number
                self.log_message(f"    📊 기준 마지막 문제번호: {last_number}")
            else:
                if last_number != self.first_db_last_number:
                    self.question_number_issues.append({'filename': filename, 'db_name': db_name, 'expected': self.first_db_last_number, 'actual': last_number})
                    self.log_message(f"    ⚠️  문제번호 불일치: 기대값 {self.first_db_last_number}, 실제값 {last_number}")
                else:
                    self.log_message(f"    📊 마지막 문제번호: {last_number} (일치)")
            
            if debug_result['has_issues']:
                self.debug_issues.append({'filename': filename, 'db_name': db_name, 'debug_output': debug_result.get('debug_output', ''), 'issues': debug_result['issues']})
                self.log_message(f"⚠️  {filename}: 디버그 이슈 발견")
            else:
                self.log_message(f"✅ {filename}: 성공적으로 처리됨")
        except Exception as e:
            raise Exception(f"PDF 처리 실패: {e}")
            
    def update_progress(self):
        if self.total_files > 0:
            progress = (self.processed_files / self.total_files) * 100
            self.progress_var.set(progress)
            self.progress_label.config(text=f"{self.processed_files}/{self.total_files} 완료 ({progress:.1f}%)")
            
    def finalize_processing(self):
        """
        [수정] 처리 완료 후 최종 결과를 정리하고, 이슈 유무에 따라 다음 단계를 결정합니다.
        """
        self.log_message("\n" + "="*50)
        self.log_message("모든 PDF 처리 완료")
        self.log_message(f"성공: {self.processed_files - len(self.failed_files)}개, 실패: {len(self.failed_files)}개, 디버그 이슈: {len(self.debug_issues)}개, 문제번호 불일치: {len(self.question_number_issues)}개")
        
        # 이슈가 있는 파일 목록 생성 (수정 모드와 이름 변경에서 모두 사용)
        all_issues = self.debug_issues + self.question_number_issues
        unique_issues = {}
        for item in all_issues:
            unique_issues[item['filename']] = item
        self.issue_files_to_correct = list(unique_issues.values())

        # 실패한 파일 목록 표시
        if self.failed_files:
            self.log_message("\n실패한 파일들:")
            for pdf_path, error in self.failed_files:
                self.log_message(f"  - {os.path.basename(pdf_path)}: {error}")

        # [수정] 이슈 유무에 따라 분기
        if self.issue_files_to_correct:
            # 이슈가 있으면, 수정 모드 시작 (이름 변경은 수정 모드 종료 후 진행)
            self.start_correction_mode()
        else:
            # 이슈가 없으면, 바로 DB 이름 변경 후 완료 메시지 표시
            self.log_message("\n발견된 이슈가 없습니다. DB 파일명 변경을 시작합니다.")
            self.rename_db_files()
            messagebox.showinfo("처리 완료", "모든 파일이 이슈 없이 성공적으로 처리 및 변경되었습니다.")

    def rename_db_files(self) -> Dict[str, str]:
        """날짜 기준 역순으로 DB 파일명을 question1.db 등으로 변경하고 ExamSession을 업데이트합니다."""
        self.log_message("\n--- 데이터베이스 파일명 변경 및 ExamSession 업데이트 시작 ---")
        renamed_map = {}
        if not self.db_last_numbers:
            self.log_message("변경할 DB 파일이 없습니다.")
            return renamed_map

        sorted_dbs = sorted(self.db_last_numbers, key=lambda x: x['db_name'], reverse=True)
        
        for i, db_info in enumerate(sorted_dbs):
            session_num = i + 1
            old_db_name = db_info['db_name']
            new_db_name = f"question{session_num}.db"
            old_db_path = os.path.join(self.selected_folder, old_db_name)
            new_db_path = os.path.join(self.selected_folder, new_db_name)
            
            if os.path.exists(old_db_path):
                try:
                    os.rename(old_db_path, new_db_path)
                    self.log_message(f"  - 파일명 변경: '{old_db_name}' -> '{new_db_name}'")
                    renamed_map[old_db_name] = new_db_name
                    self.update_exam_session_in_db(new_db_path, session_num)
                except Exception as e:
                    self.log_message(f"  - '{old_db_name}' 처리 실패: {e}")
            else:
                self.log_message(f"  - 원본 DB 파일을 찾을 수 없음: '{old_db_name}'")
        
        self.log_message("--- 데이터베이스 파일명 변경 및 업데이트 완료 ---")
        return renamed_map

    def update_exam_session_in_db(self, db_path: str, session_number: int):
        """DB 파일의 모든 레코드에 ExamSession 번호를 업데이트합니다."""
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE questions SET ExamSession = ?", (session_number,))
            conn.commit()
            conn.close()
            self.log_message(f"    - '{os.path.basename(db_path)}'의 ExamSession을 '{session_number}'로 업데이트 완료")
        except Exception as e:
            self.log_message(f"    - '{os.path.basename(db_path)}'의 ExamSession 업데이트 실패: {e}")
    
    def start_correction_mode(self):
        if not self.issue_files_to_correct:
            self.log_message("수정할 디버그 이슈가 없습니다.")
            return

        if not DB_BROWSER_PATH or not os.path.exists(DB_BROWSER_PATH):
            messagebox.showerror("경고", f"DB 브라우저 경로가 잘못되었습니다.\n경로: {DB_BROWSER_PATH}\n\n스크립트 상단의 DB_BROWSER_PATH 변수를 수정해주세요.")
            self.log_message("DB_BROWSER_PATH가 유효하지 않아 수정 모드를 시작할 수 없습니다.")
            return

        self.log_message("\n--- 디버그 수정 모드를 시작합니다 ---")
        self.log_message(f"총 {len(self.issue_files_to_correct)}개의 파일에 대한 수정이 필요합니다.")
        
        self.correction_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        self.current_correction_index = -1
        self.next_correction_file()

    def next_correction_file(self):
        self.close_opened_processes()

        self.current_correction_index += 1
        if self.current_correction_index >= len(self.issue_files_to_correct):
            messagebox.showinfo("수정 완료", "모든 이슈 파일에 대한 검토가 완료되었습니다. DB 파일명 변경을 시작합니다.")
            self.finish_correction_mode()
            return
        
        issue_info = self.issue_files_to_correct[self.current_correction_index]
        filename = issue_info['filename']
        # [수정] 이름 변경 전의 원본 db_name을 사용해야 함
        db_name = self.extract_db_name(filename) 
        
        pdf_path = os.path.join(self.selected_folder, filename)
        db_path = os.path.join(self.selected_folder, db_name)

        self.correction_label.config(text=f"수정 중 ({self.current_correction_index + 1}/{len(self.issue_files_to_correct)}): {filename}")
        self.log_message(f"파일 열기: {filename}, {db_name}")

        self.open_files_for_correction(pdf_path, db_path)

    def open_files_for_correction(self, pdf_path: str, db_path: str):
        try:
            if sys.platform == "darwin":
                pdf_proc = subprocess.Popen(["open", pdf_path])
                self.opened_processes.append(pdf_proc)
            elif sys.platform == "win32":
                os.startfile(pdf_path)
            else:
                pdf_proc = subprocess.Popen(["xdg-open", pdf_path])
                self.opened_processes.append(pdf_proc)
            self.log_message(f"  - PDF 파일 열기 성공: {pdf_path}")
        except Exception as e:
            self.log_message(f"  - PDF 파일 열기 실패: {e}")

        if os.path.exists(db_path) and os.path.exists(DB_BROWSER_PATH):
            try:
                db_proc = subprocess.Popen([DB_BROWSER_PATH, db_path])
                self.opened_processes.append(db_proc)
                self.log_message(f"  - DB 파일 열기 성공: {db_path}")
            except Exception as e:
                self.log_message(f"  - DB 파일 열기 실패: {e}")
        else:
            self.log_message(f"  - DB 파일({db_path}) 또는 브라우저 경로({DB_BROWSER_PATH})를 찾을 수 없습니다.")

    def close_opened_processes(self):
        for proc in self.opened_processes:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception:
                pass
        self.opened_processes = []

    def finish_correction_mode(self):
        """[수정] 수정 모드를 종료하고, DB 파일명 변경을 트리거합니다."""
        self.close_opened_processes()
        self.correction_frame.grid_remove()
        self.correction_label.config(text="수정할 파일: 대기 중...")
        self.log_message("--- 디버그 수정 모드를 종료합니다 ---")

        # [수정] 여기서 DB 파일명 변경 실행
        self.rename_db_files()
        
        self.reset_ui_state()

    def reset_ui_state(self):
        self.process_button.config(state="normal" if self.selected_folder else "disabled")
        self.stop_button.config(state="disabled")
        self.folder_button.config(state="normal")
        self.progress_var.set(0)
        self.progress_label.config(text="대기 중...")
        
    def log_message(self, message: str):
        def update_log():
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
        self.root.after(0, update_log)
    def start_category_update_process(self):
        """UI의 버튼을 통해 카테고리 업데이트 프로세스를 시작합니다."""
        if not self.selected_folder:
            messagebox.showerror("오류", "먼저 폴더를 선택해주세요.")
            return

        self.log_message("\n" + "="*50)
        self.log_message("카테고리 정보 일괄 업데이트 시작...")

        # 비동기 실행을 위해 스레드 사용
        threading.Thread(target=self.run_category_update_logic).start()

    # in 0_Machine.py -> PDFBatchProcessor 클래스

# in 0_Machine.py

    def run_category_update_logic(self):
        """
        [수정] 실제 카테고리 업데이트 로직을 실행하며, 기준 PDF에서 시험명을 추출하여 전달합니다.
        """
        try:
            # 1. 가장 최신 PDF 파일 찾기
            pdf_files = self.get_pdf_files(self.selected_folder)
            if not pdf_files:
                self.log_message("오류: 처리할 PDF 파일을 찾을 수 없습니다.")
                messagebox.showerror("오류", "폴더에서 처리할 PDF 파일을 찾을 수 없습니다.")
                return

            master_pdf_path = sorted(pdf_files, reverse=True)[0]

            # [추가] 기준 PDF 파일명에서 시험명(카테고리명) 추출
            pdf_filename = os.path.basename(master_pdf_path)
            match = re.search(r"^(.*?)\d{8}\(교사용\)\.pdf$", pdf_filename)
            exam_name = match.group(1) if match else os.path.splitext(pdf_filename)[0]
            self.log_message(f"파일명 기반 시험명: '{exam_name}'")

            # 2. PDFProcessor의 카테고리 업데이트 함수 호출 (시험명 전달)
            processor = PDFProcessor()
            success, message = processor.update_all_db_categories(self.selected_folder, exam_name)
            
            self.log_message(message)
            
            if success:
                messagebox.showinfo("완료", "카테고리 정보 업데이트가 완료되었습니다.")
            else:
                # 오류 메시지는 message 변수 안에 포함되어 이미 로깅됨
                messagebox.showerror("오류", "카테고리 정보 업데이트 중 오류가 발생했습니다.")

        except Exception as e:
            error_msg = f"카테고리 업데이트 중 심각한 오류 발생: {e}"
            self.log_message(error_msg)
            messagebox.showerror("오류", error_msg)

def main():
    try:
        PDFProcessor()
    except NameError:
        print("오류: pdf_processor.py 또는 해당 클래스를 찾을 수 없습니다.")
        return
        
    root = tk.Tk()
    app = PDFBatchProcessor(root)
    root.mainloop()

if __name__ == "__main__":
    main()