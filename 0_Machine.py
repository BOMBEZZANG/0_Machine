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
        self.root.title("PDF/DB Batch Processor")
        self.root.geometry("800x650")
        
        self.default_folder = os.path.expanduser("~/Desktop/Apps/qcjongmin/appauto/raw_DB/rawdbs/")
        self.selected_folder = ""
        
        self.setup_ui()
        self.stop_flag = False
        self.reset_internal_state()

    def reset_internal_state(self):
        """내부 처리 상태 변수들을 초기화합니다."""
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
        
    def setup_ui(self):
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # --- UI 요소들 ---
        folder_frame = ttk.LabelFrame(main_frame, text="폴더 선택", padding="5")
        folder_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        self.folder_button = ttk.Button(folder_frame, text="Folder Selected", command=self.select_folder)
        self.folder_button.grid(row=0, column=0, padx=(0, 10))
        self.folder_label = ttk.Label(folder_frame, text="폴더를 선택해주세요", foreground="gray")
        self.folder_label.grid(row=0, column=1, sticky=(tk.W, tk.E))
        
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=2, pady=(0, 10), sticky=tk.W)
        self.process_button = ttk.Button(control_frame, text="1. PDF -> DB 변환", command=self.start_pdf_to_db_process, state="disabled")
        self.process_button.pack(side=tk.LEFT, padx=(0, 10))
        self.stop_button = ttk.Button(control_frame, text="중단", command=self.stop_processing, state="disabled")
        self.stop_button.pack(side=tk.LEFT, padx=(0, 5))
        
        button_frame_2 = ttk.Frame(main_frame)
        button_frame_2.grid(row=2, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)
        self.api_button = ttk.Button(button_frame_2, text="2. 해설 생성(API)", command=lambda: self.start_external_script_process('2_pro_api.py'), state="disabled")
        self.api_button.pack(side=tk.LEFT, padx=(0,10))
        
        self.audio_button = ttk.Button(button_frame_2, text="3. 음성 생성(TTS)", command=lambda: self.start_external_script_process('3_description_audio.py'), state="disabled")
        self.audio_button.pack(side=tk.LEFT)

        # [새로 추가된 버튼]
        button_frame_3 = ttk.Frame(main_frame)
        button_frame_3.grid(row=3, column=0, columnspan=2, pady=(10, 0), sticky=tk.W)
        self.studynote_button = ttk.Button(button_frame_3, text="4. Study Note 생성", command=lambda: self.start_external_script_process('4_StudyNote.py'), state="disabled")
        self.studynote_button.pack(side=tk.LEFT, padx=(0, 10))

        progress_frame = ttk.LabelFrame(main_frame, text="진행상황", padding="5")
        progress_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10,0)) # row 3 -> 4
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100, length=300)
        self.progress_bar.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 5))
        self.progress_label = ttk.Label(progress_frame, text="대기 중...")
        self.progress_label.grid(row=1, column=0, columnspan=2, pady=(0, 5))

        log_frame = ttk.LabelFrame(main_frame, text="처리 로그", padding="5")
        log_frame.grid(row=5, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10,0)) # row 4 -> 5
        self.log_text = scrolledtext.ScrolledText(log_frame, height=15, width=80)
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

        self.correction_frame = ttk.LabelFrame(main_frame, text="디버그 수정 모드", padding="5")
        self.correction_label = ttk.Label(self.correction_frame, text="수정할 파일: 대기 중...", foreground="blue")
        self.correction_label.grid(row=0, column=0, padx=5, pady=5, sticky=tk.W)
        self.next_button = ttk.Button(self.correction_frame, text="다음", command=self.next_correction_file)
        self.next_button.grid(row=0, column=1, padx=5, pady=5)
        self.finish_button = ttk.Button(self.correction_frame, text="종료", command=self.finish_correction_mode)
        self.finish_button.grid(row=0, column=2, padx=5, pady=5)
        
        self.root.columnconfigure(0, weight=1); self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1); main_frame.rowconfigure(5, weight=1) # row 4 -> 5
        folder_frame.columnconfigure(1, weight=1); log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1); self.correction_frame.columnconfigure(0, weight=1)

    def set_ui_for_processing(self, is_processing: bool):
        """UI 요소들의 상태를 일괄 변경합니다."""
        state = "disabled" if is_processing else "normal"
        self.folder_button.config(state=state)
        self.process_button.config(state=state)
        self.api_button.config(state=state)
        self.audio_button.config(state=state)
        self.studynote_button.config(state=state) # [추가] 새 버튼 상태 관리
        self.stop_button.config(state="normal" if is_processing else "disabled")
        if not self.selected_folder:
            self.process_button.config(state="disabled")
            self.api_button.config(state="disabled")
            self.audio_button.config(state="disabled")
            self.studynote_button.config(state="disabled") # [추가] 새 버튼 상태 관리

    def select_folder(self):
        initial_dir = self.default_folder if os.path.exists(self.default_folder) else "/"
        folder_path = filedialog.askdirectory(title="PDF 파일들이 있는 폴더를 선택하세요", initialdir=initial_dir)
        if folder_path:
            self.selected_folder = folder_path
            self.folder_label.config(text=folder_path, foreground="black")
            self.set_ui_for_processing(False)
            self.log_message(f"폴더 선택됨: {folder_path}")
    
    def start_external_script_process(self, script_name: str):
        """외부 스크립트 실행을 위한 범용 메소드"""
        if not self.selected_folder:
            messagebox.showerror("오류", "먼저 폴더를 선택해주세요.")
            return

        process_name = ""
        if "2_pro_api" in script_name: process_name = "2. 해설 생성(API)"
        elif "3_description_audio" in script_name: process_name = "3. 음성 생성(TTS)"
        elif "4_StudyNote" in script_name: process_name = "4. Study Note 생성" # [추가] 새 스크립트 이름
        
        self.log_message("\n" + "="*50)
        self.log_message(f"{process_name} 스크립트 실행 시작...")
        if "api" in script_name or "audio" in script_name or "StudyNote" in script_name:
            self.log_message("이 작업은 문제 수에 따라 수십 분 이상 소요될 수 있습니다.")

        self.set_ui_for_processing(True)
        threading.Thread(target=self.run_external_script_in_thread, args=(script_name,), daemon=True).start()

    def run_external_script_in_thread(self, script_name: str):
        """스레드에서 범용적으로 외부 스크립트를 실행하고 자동 연계"""
        return_code = -1
        try:
            script_path = os.path.join(os.path.dirname(__file__), script_name)
            if not os.path.exists(script_path):
                self.log_message(f"오류: {script_name} 파일을 찾을 수 없습니다.")
                self.root.after(100, lambda: self.set_ui_for_processing(False))
                return

            command = [sys.executable, script_path, self.selected_folder]
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8', bufsize=1)

            for line in iter(process.stdout.readline, ''):
                self.log_message(line.strip())

            process.stdout.close()
            return_code = process.wait()
            
            if return_code == 0:
                self.log_message(f"\n✅ {script_name} 스크립트 실행이 성공적으로 완료되었습니다.")
                if "2_pro_api" in script_name:
                    messagebox.showinfo("2단계 완료", "해설 생성(API)이 완료되었습니다.\n다음 단계를 진행해주세요.")
                elif "3_description_audio" in script_name:
                     messagebox.showinfo("3단계 완료", "음성 생성이 완료되었습니다.")
                elif "4_StudyNote" in script_name:
                     messagebox.showinfo("모든 작업 완료", "Study Note 관련 모든 프로세스가 완료되었습니다.")
            else:
                self.log_message(f"\n⚠️ {script_name} 스크립트 실행 중 오류가 발생했습니다. (Return Code: {return_code})")
                messagebox.showerror("오류", f"{script_name} 실행 중 오류가 발생했습니다. 로그를 확인해주세요.")

        except Exception as e:
            self.log_message(f"스크립트 실행 중 심각한 오류 발생: {e}")
        finally:
            self.root.after(100, lambda: self.set_ui_for_processing(False))
    
    def start_pdf_to_db_process(self):
        """1단계: PDF -> DB 변환 프로세스를 시작합니다."""
        if not self.selected_folder:
            messagebox.showerror("오류", "폴더를 먼저 선택해주세요.")
            return

        self.reset_internal_state()
        self.set_ui_for_processing(True)
        self.log_message("\n" + "="*50)
        self.log_message("1. PDF -> DB 변환 프로세스 시작...")
        
        threading.Thread(target=self.process_all_pdfs, daemon=True).start()
    
    def stop_processing(self):
        self.stop_flag = True
        self.log_message("처리 중단 요청됨...")

    def process_all_pdfs(self):
        try:
            pdf_files = self.get_pdf_files(self.selected_folder)
            self.total_files = len(pdf_files)
            
            if self.total_files == 0:
                self.log_message("처리할 PDF 파일이 없습니다.")
                self.root.after(100, lambda: self.set_ui_for_processing(False))
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
            self.root.after(100, lambda: self.set_ui_for_processing(False))

    def process_single_pdf(self, processor: PDFProcessor, pdf_path: str, file_num: int):
        filename = os.path.basename(pdf_path)
        self.log_message(f"[{file_num}/{self.total_files}] 처리 시작: {filename}")
        
        db_name = self.extract_db_name(pdf_path)
        db_path = os.path.join(self.selected_folder, db_name)
        
        if os.path.exists(db_path):
            os.remove(db_path)
            
        match = re.search(r"^(.*?)\d{8}\(교사용\)\.pdf$", filename)
        exam_category = match.group(1) if match else "Unknown"
        
        debug_result = processor.process_pdf(pdf_path, db_path, exam_category)
        
        last_number = debug_result.get('last_question_number', 0)
        self.db_last_numbers.append({'filename': filename, 'db_name': db_name, 'last_number': last_number})
        
        if self.first_db_last_number is None:
            self.first_db_last_number = last_number
        elif last_number != self.first_db_last_number:
            self.question_number_issues.append({'filename': filename, 'db_name': db_name, 'expected': self.first_db_last_number, 'actual': last_number})

        if debug_result['has_issues']:
            self.debug_issues.append({'filename': filename, 'db_name': db_name, 'debug_output': debug_result.get('debug_output', ''), 'issues': debug_result['issues']})
        else:
            self.log_message(f"✅ {filename}: 성공적으로 처리됨")
            
    def update_progress(self):
        if self.total_files > 0:
            progress = (self.processed_files / self.total_files) * 100
            self.progress_var.set(progress)
            self.progress_label.config(text=f"{self.processed_files}/{self.total_files} 완료 ({progress:.1f}%)")
            
    def finalize_processing(self):
        self.log_message("\n" + "="*50)
        self.log_message("모든 PDF 처리 완료")
        
        all_issues = self.debug_issues + self.question_number_issues
        unique_issues = {item['filename']: item for item in all_issues}
        self.issue_files_to_correct = list(unique_issues.values())

        if self.failed_files:
            self.log_message("\n실패한 파일들:")
            for pdf_path, error in self.failed_files:
                self.log_message(f"  - {os.path.basename(pdf_path)}: {error}")

        if self.issue_files_to_correct:
            self.start_correction_mode()
        else:
            self.log_message("\n발견된 이슈가 없습니다. DB 파일명 변경 및 카테고리 업데이트를 시작합니다.")
            self.rename_db_files()
            self.run_category_update_logic()

    def rename_db_files(self) -> Dict[str, str]:
        self.log_message("\n--- 데이터베이스 파일명 변경 및 ExamSession 업데이트 시작 ---")
        renamed_map = {}
        if not self.db_last_numbers:
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
                    renamed_map[old_db_name] = new_db_name
                    self.update_exam_session_in_db(new_db_path, session_num)
                except Exception as e:
                    self.log_message(f"  - '{old_db_name}' 처리 실패: {e}")
        
        self.log_message("--- 데이터베이스 파일명 변경 및 업데이트 완료 ---")
        return renamed_map

    def update_exam_session_in_db(self, db_path: str, session_number: int):
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("UPDATE questions SET ExamSession = ?", (str(session_number),))
            conn.commit()
            conn.close()
        except Exception as e:
            self.log_message(f"    - '{os.path.basename(db_path)}'의 ExamSession 업데이트 실패: {e}")

    def run_category_update_logic(self):
        try:
            processor = PDFProcessor()
            all_pdfs = processor.get_pdf_files_info(self.selected_folder)
            if not all_pdfs:
                messagebox.showerror("오류", "폴더에서 처리할 PDF 파일을 찾을 수 없습니다.")
                self.set_ui_for_processing(False)
                return

            master_pdf_path = all_pdfs[-1]['full_path']
            match = re.search(r"^(.*?)\d{8}\(교사용\)\.pdf$", os.path.basename(master_pdf_path))
            exam_name = match.group(1) if match else "Unknown"

            success, message = processor.update_all_db_categories(self.selected_folder, exam_name)
            
            self.log_message(message)
            
            if success:
                 self.log_message("\n✅ 1단계(PDF 변환 및 카테고리 업데이트) 완료. 다음 단계를 진행해주세요.")
                 self.set_ui_for_processing(False)
            else:
                messagebox.showerror("오류", "카테고리 정보 업데이트 중 오류가 발생했습니다.")
                self.set_ui_for_processing(False)
        except Exception as e:
            self.log_message(f"카테고리 업데이트 중 심각한 오류 발생: {e}")
            self.set_ui_for_processing(False)
            
    def start_correction_mode(self):
        if not self.issue_files_to_correct:
            return

        if not DB_BROWSER_PATH or not os.path.exists(DB_BROWSER_PATH):
            messagebox.showerror("경고", f"DB 브라우저 경로가 잘못되었습니다.")
            return

        self.log_message("\n--- 디버그 수정 모드를 시작합니다 ---")
        self.correction_frame.grid(row=6, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10) # row 5 -> 6
        self.current_correction_index = -1
        self.next_correction_file()

    def next_correction_file(self):
        self.close_opened_processes()
        self.current_correction_index += 1
        if self.current_correction_index >= len(self.issue_files_to_correct):
            messagebox.showinfo("수정 완료", "모든 이슈 파일 검토 완료. 최종 단계를 시작합니다.")
            self.finish_correction_mode()
            return
        
        issue_info = self.issue_files_to_correct[self.current_correction_index]
        filename = issue_info['filename']
        db_name = self.extract_db_name(filename) 
        pdf_path = os.path.join(self.selected_folder, filename)
        db_path = os.path.join(self.selected_folder, db_name)

        self.correction_label.config(text=f"수정 중 ({self.current_correction_index + 1}/{len(self.issue_files_to_correct)}): {filename}")
        self.open_files_for_correction(pdf_path, db_path)

    def open_files_for_correction(self, pdf_path: str, db_path: str):
        try:
            proc = subprocess.Popen(["open", pdf_path] if sys.platform == "darwin" else ["xdg-open", pdf_path])
            self.opened_processes.append(proc)
        except Exception as e:
            self.log_message(f"  - PDF 파일 열기 실패: {e}")

        if os.path.exists(db_path) and os.path.exists(DB_BROWSER_PATH):
            try:
                proc = subprocess.Popen([DB_BROWSER_PATH, db_path])
                self.opened_processes.append(proc)
            except Exception as e:
                self.log_message(f"  - DB 파일 열기 실패: {e}")

    def close_opened_processes(self):
        for proc in self.opened_processes:
            try:
                proc.terminate()
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
            except Exception: pass
        self.opened_processes = []

    def finish_correction_mode(self):
        self.close_opened_processes()
        self.correction_frame.grid_remove()
        self.log_message("--- 디버그 수정 모드를 종료합니다 ---")
        self.rename_db_files()
        self.run_category_update_logic()

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
        return f"{match.group(1)}.db" if match else f"{Path(pdf_filename).stem}.db"
        
    def log_message(self, message: str):
        def update_log():
            self.log_text.insert(tk.END, message + "\n")
            self.log_text.see(tk.END)
        self.root.after(0, update_log)
    
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