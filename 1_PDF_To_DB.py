import fitz  # PyMuPDF
import re
import os
import sqlite3
from typing import List, Dict, Tuple, Optional
import datetime # [추가] 날짜 처리를 위해 추가

# in 1_PDF_To_DB.py

def setup_database_original(db_path: str):
    """
    원본 스키마에 따라 SQLite 데이터베이스와 'questions' 테이블을 설정합니다.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # [수정] 새로운 칼럼들 추가
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS questions (
            Question_id INTEGER PRIMARY KEY AUTOINCREMENT,
            Question_Number TEXT,
            Big_Question TEXT,
            Big_Question_Special BLOB,
            Question BLOB,
            Option1 BLOB,
            Option2 BLOB,
            Option3 BLOB,
            Option4 BLOB,
            Correct_Option INTEGER,
            Category TEXT,
            ExamSession TEXT,
            Answer_description TEXT,
            audio TEXT,
            Date_information TEXT
        )
    ''')

    conn.commit()
    conn.close()


class QuestionMarker:
    """문제 시작점 정보를 담는 클래스"""
    def __init__(self, page_num: int, question_num: int, bbox: fitz.Rect, block_index: int):
        self.page_num = page_num
        self.question_num = question_num
        self.bbox = bbox
        self.block_index = block_index


class QuestionRange:
    """문제의 범위 정보를 담는 클래스"""
    def __init__(self, start_marker: QuestionMarker, end_marker: Optional[QuestionMarker] = None):
        self.start_marker = start_marker
        self.end_marker = end_marker
        self.start_page = start_marker.page_num
        self.end_page = end_marker.page_num if end_marker else start_marker.page_num


def map_all_question_starts(doc: fitz.Document) -> List[QuestionMarker]:
    """전체 문서에서 문제 시작점들을 매핑합니다."""
    question_start_pattern = re.compile(r"^\s*([1-9]\d*)\.(?!\d)\s*(.*)")
    question_markers = []

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]

        for block_index, block in enumerate(blocks):
            if not block.get('lines') or not block['lines'][0].get('spans'):
                continue

            first_line_spans = block['lines'][0]['spans']
            if not first_line_spans:
                continue

            first_line_text = "".join(span['text'] for span in first_line_spans).strip()

            if re.match(r'^\s*\d+과목\s*:', first_line_text):
                continue

            q_match = question_start_pattern.match(first_line_text)

            if q_match:
                question_num = int(q_match.group(1))
                bbox = fitz.Rect(block['bbox'])
                marker = QuestionMarker(page_num, question_num, bbox, block_index)
                question_markers.append(marker)

    question_markers.sort(key=lambda x: x.question_num)
    return question_markers


def determine_question_ranges(question_markers: List[QuestionMarker], doc: fitz.Document) -> List[QuestionRange]:
    """각 문제의 범위를 페이지 경계를 고려하여 결정합니다."""
    question_ranges = []

    for i, current_marker in enumerate(question_markers):
        next_marker = question_markers[i + 1] if i + 1 < len(question_markers) else None
        question_range = QuestionRange(current_marker, next_marker)
        if next_marker is None:
            # 마지막 문제인 경우, 문서의 마지막 페이지까지 범위를 확장
            question_range.end_page = len(doc) -1

        question_ranges.append(question_range)

    return question_ranges


def get_blocks_in_range(doc: fitz.Document, question_range: QuestionRange) -> List[dict]:
    """특정 문제 범위에 속하는 모든 블록들을 수집합니다."""
    blocks_in_range = []

    for page_num in range(question_range.start_page, question_range.end_page + 1):
        page = doc.load_page(page_num)
        blocks = page.get_text("dict", flags=fitz.TEXTFLAGS_TEXT)["blocks"]

        for block_index, block in enumerate(blocks):
            if not block.get('lines'):
                continue

            if page_num == question_range.start_page:
                if block_index < question_range.start_marker.block_index:
                    continue

            if (question_range.end_marker and
                    page_num == question_range.end_page and
                    block_index >= question_range.end_marker.block_index):
                break

            block_with_page = block.copy()
            block_with_page['page_num'] = page_num
            blocks_in_range.append(block_with_page)

    return blocks_in_range


def get_all_images(doc: fitz.Document) -> List[dict]:
    """전체 문서에서 모든 이미지들을 수집하여 중앙에서 관리합니다."""
    all_images = []
    image_id_counter = 0

    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        page_images = page.get_images(full=True)

        for img_index, img in enumerate(page_images):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_bbox = page.get_image_bbox(img)

                image_data = {
                    'id': image_id_counter,
                    'page_num': page_num,
                    'bbox': image_bbox,
                    'data': image_bytes,
                    'assigned': False
                }
                all_images.append(image_data)
                image_id_counter += 1
            except Exception as e:
                pass

    return all_images




def process_question_range(doc: fitz.Document, question_range: QuestionRange, category_to_remove: Optional[str], all_images: List[Dict]) -> dict:
    """
    (1차 배정) 특정 범위의 문제를 처리하며, 마지막 문제의 파싱 중단 로직을 포함합니다.
    [업데이트] 마지막 문제 파싱 시, 컬럼 위치를 확인하여 다른 컬럼의 텍스트가 추가되는 것을 방지합니다.
    """
    question_start_pattern = re.compile(r"^\s*([1-9]\d*)\.\s*(.*)")
    option_splitter_pattern = re.compile(r'([①②③④❶❷❸❹])')
    option_map = {'①': 0, '②': 1, '③': 2, '④': 3, '❶': 0, '❷': 1, '❸': 2, '❹': 3}
    correct_option_markers = "❶❷❸❹"

    blocks_in_range = get_blocks_in_range(doc, question_range)
    images_in_range = [
        img for img in all_images
        if question_range.start_page <= img['page_num'] <= question_range.end_page
    ]

    q_data = {
        "Big_Question": "", "Big_Question_Special": None, "Question": None,
        "Options": ["", "", "", ""], "Correct_option_index": None,
        "question_bbox": None, "options_bboxes": [None] * 4,
        "option_marker_bboxes": [None] * 4,
        "question_number_marker_bbox": None,
        "page_num": question_range.start_page
    }

    if not blocks_in_range:
        return {}

    is_last_question = question_range.end_marker is None

    def clean_text(text: str) -> str:
        cleaned = re.sub(r'◐.*?◑', '', text)
        cleaned = cleaned.replace('최강 자격증 기출문제 전자문제집 CBT : www.comcbt.com', '')
        cleaned = cleaned.replace('전자문제집 CBT : www.comcbt.com', '')
        if category_to_remove:
            cleaned = cleaned.replace(category_to_remove, '')
        return cleaned.strip()

    def build_formatted_text(spans: List[Dict]) -> str:
        line_builder = []
        for s in spans:
            text = s['text']
            if s['flags'] & 1: line_builder.append(f"<sup>{text}</sup>")
            elif s['flags'] & 2: line_builder.append(f"<sub>{text}</sub>")
            else: line_builder.append(text)
        return "".join(line_builder)

    # 페이지 너비의 중간 지점을 계산
    page = doc.load_page(question_range.start_page)
    column_midpoint = page.rect.width / 2

    first_block = blocks_in_range[0]
    first_line_spans = first_block['lines'][0]['spans'] if first_block.get('lines') else []
    first_line_formatted_text = build_formatted_text(first_line_spans)
    q_match = question_start_pattern.match(first_line_formatted_text.strip())

    if not q_match:
        first_line_plain_text = "".join(s['text'] for s in first_line_spans).strip()
        q_match = question_start_pattern.match(first_line_plain_text)
        if not q_match:
            return {}

    q_num_str = q_match.group(1)
    q_data["question_number"] = q_num_str

    first_line = first_block['lines'][0]
    q_num_marker_bbox = next((fitz.Rect(span['bbox']) for span in first_line['spans'] if q_num_str in span['text']), None)
    q_data["question_number_marker_bbox"] = q_num_marker_bbox

    full_question_text = clean_text(q_match.group(2))
    question_only_bbox = fitz.Rect(first_block['bbox'])
    is_option_part = False
    last_option_index = None
    stop_parsing = False
    last_option_marker_bbox = None  # 마지막으로 처리한 옵션 마커의 bbox 저장

    for block in blocks_in_range:
        # 페이지가 바뀌면 column_midpoint 업데이트
        if 'page_num' in block and block.get('page_num') != page.number:
            page = doc.load_page(block['page_num'])
            column_midpoint = page.rect.width / 2

        for line in block.get('lines', []):
            line_text_plain = "".join([s['text'] for s in line.get('spans', [])]).strip()

            if is_last_question and is_option_part and ("전자문제집 CBT" in line_text_plain or "www.comcbt.com" in line_text_plain):
                stop_parsing = True
                break

            line_text_formatted = build_formatted_text(line.get('spans', []))
            if re.match(r'^\s*\d+과목\s*:', line_text_plain): continue
            cleaned_line_text = clean_text(line_text_formatted).strip()
            if not cleaned_line_text: continue

            split_parts = option_splitter_pattern.split(cleaned_line_text)

            if len(split_parts) > 1:
                is_option_part = True
                for i in range(1, len(split_parts), 2):
                    marker, text = split_parts[i], split_parts[i + 1].strip() if (i + 1) < len(split_parts) else ""
                    if marker in option_map:
                        opt_idx = option_map[marker]
                        q_data["Options"][opt_idx] += text
                        last_option_index = opt_idx
                        if q_data["options_bboxes"][opt_idx] is None: q_data["options_bboxes"][opt_idx] = fitz.Rect(line['bbox'])
                        else: q_data["options_bboxes"][opt_idx].include_rect(line['bbox'])
                        for span in line.get('spans', []):
                           if marker in span['text']:
                               current_marker_bbox = fitz.Rect(span['bbox'])
                               q_data["option_marker_bboxes"][opt_idx] = current_marker_bbox
                               last_option_marker_bbox = current_marker_bbox 
                               break
                        if marker in correct_option_markers: q_data["Correct_option_index"] = opt_idx
            
            elif is_option_part and last_option_index is not None and last_option_marker_bbox is not None:
                line_bbox = fitz.Rect(line['bbox'])
                
                # 현재 라인이 마지막 옵션 마커와 같은 컬럼에 있는지 확인
                marker_is_left = last_option_marker_bbox.x1 < column_midpoint
                line_is_left = line_bbox.x1 < column_midpoint
                
                if marker_is_left == line_is_left:
                    q_data["Options"][last_option_index] += " " + cleaned_line_text
                    if q_data["options_bboxes"][last_option_index]: 
                        q_data["options_bboxes"][last_option_index].include_rect(line_bbox)
            
            elif not is_option_part and not line_text_plain.startswith(q_num_str):
                full_question_text += " " + cleaned_line_text
                question_only_bbox.include_rect(block['bbox'])
        
        if stop_parsing:
            break

    q_data["Big_Question"] = " ".join(full_question_text.split())
    q_data["question_bbox"] = question_only_bbox

    def find_best_image(image_list: List[Dict], marker_bbox: fitz.Rect) -> Optional[Dict]:
        candidate_image, min_dist = None, float('inf')
        TOLERANCE_Y = 5 

        for img in image_list:
            if img['assigned']:
                continue

            has_overlap = max(marker_bbox.y0, img['bbox'].y0) < min(marker_bbox.y1, img['bbox'].y1)
            gap_y = img['bbox'].y0 - marker_bbox.y1
            is_within_tolerance = (gap_y > 0 and gap_y <= TOLERANCE_Y)
            is_vertically_aligned = has_overlap or is_within_tolerance
            is_right_of_marker = img['bbox'].x1 > marker_bbox.x1
            
            page = doc.load_page(img['page_num']) 
            img_column_midpoint = page.rect.width / 2
            
            marker_is_left = marker_bbox.x1 < img_column_midpoint
            image_is_left = img['bbox'].x1 < img_column_midpoint
            in_same_column = (marker_is_left == image_is_left)

            if is_vertically_aligned and is_right_of_marker and in_same_column:
                dist = img['bbox'].x0 - marker_bbox.x1
                if dist < min_dist:
                    min_dist, candidate_image = dist, img
        
        return candidate_image

    for i in range(4):
        marker_bbox = q_data["option_marker_bboxes"][i]
        if not marker_bbox or (isinstance(q_data["Options"][i], str) and q_data["Options"][i].strip()):
            continue
        if isinstance(q_data["Options"][i], bytes):
            continue

        same_page_images = [img for img in images_in_range if img['page_num'] == q_data['page_num']]
        best_image = find_best_image(same_page_images, marker_bbox)

        if best_image:
            best_image['assigned'] = True
            q_data["Options"][i] = best_image['data']
            continue

        other_page_images = [img for img in images_in_range if img['page_num'] != q_data['page_num']]
        best_image = find_best_image(other_page_images, marker_bbox)
        
        if best_image:
            best_image['assigned'] = True
            q_data["Options"][i] = best_image['data']

    opt1_marker_bbox = q_data["option_marker_bboxes"][0]
    question_bbox = q_data["question_bbox"]
    if opt1_marker_bbox and question_bbox:
        candidate_image, min_dist = None, float('inf')
        images_to_check = sorted(images_in_range, key=lambda img: img['page_num'] != q_data['page_num'])
        
        for img in images_to_check:
            if img['assigned']: continue
            is_vertically_correct = img['bbox'].y1 <= opt1_marker_bbox.y0 and img['bbox'].y0 >= question_bbox.y1
            is_horizontally_aligned = max(img['bbox'].x0, question_bbox.x0) < min(img['bbox'].x1, question_bbox.x1)
            if is_vertically_correct and is_horizontally_aligned:
                dist = img['bbox'].y0 - question_bbox.y1
                if dist < min_dist: min_dist, candidate_image = dist, img
        if candidate_image:
            candidate_image['assigned'] = True
            q_data["Question"] = candidate_image['data']

    return q_data

def is_ignorable_header_footer_text(text: str, exam_name: Optional[str]) -> bool:
    cleaned = text
    cleaned = re.sub(r'◐.*?◑', '', cleaned).strip()
    cleaned = cleaned.replace('최강 자격증 기출문제 전자문제집 CBT : www.comcbt.com', '').strip()
    cleaned = cleaned.replace('전자문제집 CBT : www.comcbt.com', '').strip()
    if exam_name:
        cleaned = cleaned.replace(exam_name, '').strip()
    if re.match(r'^\s*\d+과목\s*:', cleaned):
        cleaned = ""
    return not bool(cleaned)


def assign_cross_column_images(doc: fitz.Document, all_questions: List[Dict], all_images: List[Dict], exam_name: Optional[str]):
    """
    (2차 배정) 다른 단(column)이나 다음 페이지에 있는 이미지를 배정합니다.
    """
    unassigned_images = [img for img in all_images if not img['assigned']]
    if not unassigned_images:
        return

    page_width = doc[0].rect.width
    column_midpoint = page_width / 2

    for image in unassigned_images:
        page_num = image['page_num']
        img_bbox = image['bbox']

        is_topmost = True
        page = doc.load_page(page_num)
        blocks = page.get_text("dict")["blocks"]
        for other_img in all_images:
            if other_img['id'] == image['id'] or other_img['page_num'] != page_num:
                continue
            if (other_img['bbox'].x0 < column_midpoint) == (img_bbox.x0 < column_midpoint) and other_img['bbox'].y1 < img_bbox.y0:
                is_topmost = False
                break
        if not is_topmost:
            continue

        for block in blocks:
            block_bbox = fitz.Rect(block['bbox'])
            if (block_bbox.x0 < column_midpoint) == (img_bbox.x0 < column_midpoint) and block_bbox.y1 < img_bbox.y0:
                block_text = "".join(span['text'] for line in block.get('lines',[]) for span in line.get('spans',[]))
                if not is_ignorable_header_footer_text(block_text, exam_name):
                    is_topmost = False
                    break
        if not is_topmost:
            continue

        img_col_is_left = img_bbox.x0 < column_midpoint

        if not img_col_is_left:
            target_question, vertical_reference_question = None, None
            same_page_left_q = [q for q in all_questions if q.get('page_num') == page_num and q.get('question_bbox') and q['question_bbox'].x1 < column_midpoint]
            if same_page_left_q:
                target_question = max(same_page_left_q, key=lambda q: q['question_bbox'].y1)
            
            same_page_right_q = [q for q in all_questions if q.get('page_num') == page_num and q.get('question_bbox') and q['question_bbox'].x0 > column_midpoint]
            if same_page_right_q:
                vertical_reference_question = min(same_page_right_q, key=lambda q: q['question_bbox'].y0)

            if not target_question or not vertical_reference_question:
                continue

            first_opt_marker_bbox = vertical_reference_question.get('option_marker_bboxes', [None]*4)[0]
            if not first_opt_marker_bbox:
                continue

            if img_bbox.y1 < first_opt_marker_bbox.y0 and target_question.get('Question') is None:
                target_question['Question'] = image['data']
                image['assigned'] = True

        else:
            if page_num == 0:
                continue
            
            prev_page_q = [q for q in all_questions if q.get('page_num') == page_num - 1 and q.get('question_number')]
            if not prev_page_q:
                continue

            potential_target = max(prev_page_q, key=lambda q: int(q['question_number']))
            first_opt_marker_bbox = potential_target.get('option_marker_bboxes', [None]*4)[0]
            
            if not first_opt_marker_bbox:
                continue

            if img_bbox.y1 < first_opt_marker_bbox.y0:
                if potential_target.get('Question') is None:
                    potential_target['Question'] = image['data']
                    image['assigned'] = True


def assign_bottom_column_images(doc: fitz.Document, all_questions: List[Dict], all_images: List[Dict], exam_name: Optional[str]):
    unassigned_images = [img for img in all_images if not img['assigned']]
    if not unassigned_images:
        return

    page_width = doc[0].rect.width
    column_midpoint = page_width / 2
    page_blocks_cache = {p: doc.load_page(p).get_text("dict")["blocks"] for p in set(img['page_num'] for img in unassigned_images)}

    for image in unassigned_images:
        img_bbox, page_num = image['bbox'], image['page_num']
        img_col_is_left = img_bbox.x0 < column_midpoint

        is_bottommost = True
        for other_img in all_images:
            if other_img['id'] == image['id'] or other_img['page_num'] != page_num: continue
            other_img_col_is_left = other_img['bbox'].x0 < column_midpoint
            if other_img_col_is_left == img_col_is_left and other_img['bbox'].y0 > img_bbox.y1:
                is_bottommost = False
                break
        if not is_bottommost:
            continue

        for block in page_blocks_cache.get(page_num, []):
            block_bbox = fitz.Rect(block['bbox'])
            block_col_is_left = block_bbox.x0 < column_midpoint
            if block_col_is_left == img_col_is_left and block_bbox.y0 > img_bbox.y1:
                block_text = "".join(span['text'] for line in block.get('lines', []) for span in line.get('spans', []))
                if not is_ignorable_header_footer_text(block_text, exam_name):
                    is_bottommost = False
                    break
        if not is_bottommost:
            continue

        candidate_questions = []
        for q in all_questions:
            q_bbox = q.get('question_bbox')
            if not q_bbox or q.get('page_num') != page_num: continue
            q_col_is_left = q_bbox.x0 < column_midpoint
            if q_col_is_left == img_col_is_left and q_bbox.y1 < img_bbox.y0:
                candidate_questions.append(q)
        if not candidate_questions:
            continue

        target_question = max(candidate_questions, key=lambda q: q['question_bbox'].y1)

        if target_question.get('Question') is None:
            target_question['Question'] = image['data']
            image['assigned'] = True


def assign_images_below_options(all_questions: List[Dict], all_images: List[Dict]):
    """
    (4차 배정) 선택지 마크 아래 이미지 배정을 시작합니다.
    """
    unassigned_images = [img for img in all_images if not img['assigned']]
    if not unassigned_images:
        return

    sorted_questions = sorted(all_questions, key=lambda q: int(q.get('question_number', 0)))
    num_questions = len(sorted_questions)

    for i, current_q in enumerate(sorted_questions):
        for opt_idx in range(4):
            option_marker_bbox = current_q.get('option_marker_bboxes', [None]*4)[opt_idx]
            if not option_marker_bbox:
                continue

            option_content = current_q['Options'][opt_idx]
            if isinstance(option_content, bytes) or (isinstance(option_content, str) and option_content.strip()):
                continue

            bottom_boundary_bbox = None
            if opt_idx < 3:
                next_marker_bbox = current_q.get('option_marker_bboxes', [None]*4)[opt_idx + 1]
                if next_marker_bbox:
                     bottom_boundary_bbox = next_marker_bbox
            
            if bottom_boundary_bbox is None and i + 1 < num_questions:
                next_q = sorted_questions[i + 1]
                bottom_boundary_bbox = next_q.get('question_number_marker_bbox')

            if not bottom_boundary_bbox:
                continue

            for image in unassigned_images:
                if image['assigned'] or image['page_num'] != current_q['page_num']:
                    continue

                img_bbox = image['bbox']
                is_below_option = img_bbox.y0 > option_marker_bbox.y0
                is_above_boundary = img_bbox.y1 < bottom_boundary_bbox.y0
                
                if is_below_option and is_above_boundary:
                    current_q['Options'][opt_idx] = image['data']
                    image['assigned'] = True
                    break


def assign_special_question_images(all_questions: List[Dict], all_images: List[Dict]):
    """
    (5차 배정) 문제 텍스트 영역과 겹치고 선택지 1번 위에 있는 이미지를 Big_Question_Special로 배정합니다.
    """
    unassigned_images = [img for img in all_images if not img['assigned']]
    if not unassigned_images:
        return

    for question in all_questions:
        if question.get('Big_Question_Special') is not None:
            continue

        question_bbox = question.get('question_bbox')
        opt1_marker_bbox = question.get('option_marker_bboxes', [None]*4)[0]

        if not question_bbox or not opt1_marker_bbox:
            continue

        for image in unassigned_images:
            if image['assigned']:
                continue

            if image['page_num'] != question['page_num']:
                continue

            img_bbox = image['bbox']

            is_overlapping = img_bbox.intersects(question_bbox)
            is_above_option1 = img_bbox.y1 < opt1_marker_bbox.y0

            if is_overlapping and is_above_option1:
                question['Big_Question_Special'] = image['data']
                image['assigned'] = True
                break


def assign_bottom_of_page_special_images(doc: fitz.Document, all_questions: List[Dict], all_images: List[Dict]):
    """
    (6차 배정) 페이지 하단에 위치한 문제에 대해 Big_Question_Special 이미지를 배정합니다.
    """
    unassigned_images = [img for img in all_images if not img['assigned']]
    if not unassigned_images:
        return

    page_heights = {i: doc[i].rect.height for i in range(len(doc))}

    for image in unassigned_images:
        if image['assigned']:
            continue

        for question in all_questions:
            if question.get('Big_Question_Special') is not None:
                continue
            if image['page_num'] != question.get('page_num'):
                continue
            
            question_bbox = question.get('question_bbox')
            if not question_bbox:
                continue
            
            is_overlapping = image['bbox'].intersects(question_bbox)

            page_height = page_heights.get(question['page_num'])
            if not page_height:
                continue
            
            is_at_bottom = question_bbox.y1 > (page_height * 0.70)

            if is_overlapping and is_at_bottom:
                question['Big_Question_Special'] = image['data']
                image['assigned'] = True
                break


def final_assignment_debug_check(db_path: str, all_images: List[Dict]):
    """
    모든 배정이 끝난 후, DB의 빈 항목과 미배정 이미지를 최종적으로 확인하여
    디버깅 정보를 출력합니다.
    """
    print("\n" + "="*25 + " 최종 할당 디버깅 시작 " + "="*25)

    def format_bbox(bbox: Optional[fitz.Rect]) -> str:
        """BBox 객체를 읽기 좋은 문자열로 포맷팅합니다."""
        if bbox and isinstance(bbox, fitz.Rect):
            return f"Rect({bbox.x0:.2f}, {bbox.y0:.2f}, {bbox.x1:.2f}, {bbox.y1:.2f})"
        return "좌표 없음"

    # =================================================================
    # [복원] 1. DB 전체를 검색하여 비어있는 항목 찾기
    # =================================================================
    print("\n[1] 데이터베이스의 빈 항목 확인")
    issues = []
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        any_empty_found = False

        # 비어있는 Big_Question 확인
        cursor.execute("SELECT Question_id FROM questions WHERE Big_Question IS NULL OR Big_Question = ''")
        empty_bq = cursor.fetchall()
        if empty_bq:
            any_empty_found = True
            issue_msg = f"  - Big_Question이 비어있는 문제 ID: {[row[0] for row in empty_bq]}"
            print(issue_msg)
            issues.append(issue_msg)

        # 비어있는 Option 1~4 확인
        for i in range(1, 5):
            option_col = f"Option{i}"
            query = f"SELECT Question_id, {option_col} FROM questions WHERE {option_col} IS NULL OR (TYPEOF({option_col}) = 'text' AND {option_col} = '')"
            cursor.execute(query)
            empty_opts = cursor.fetchall()
            
            empty_text_opts = [row[0] for row in empty_opts if not isinstance(row[1], bytes)]

            if empty_text_opts:
                any_empty_found = True
                issue_msg = f"  - {option_col}이 비어있는 문제 ID (텍스트/이미지 모두 없음): {empty_text_opts}"
                print(issue_msg)
                issues.append(issue_msg)

        if not any_empty_found:
            print("  - 확인된 모든 문제의 Big_Question과 Option 필드에 값이 채워져 있습니다.")
        
        conn.close()
    except Exception as e:
        error_msg = f"  - 데이터베이스 확인 중 오류 발생: {e}"
        print(error_msg)
        issues.append(error_msg)

    # =================================================================
    # [복원] 2. 최종적으로 배정되지 않은 이미지 확인
    # =================================================================
    print("\n[2] 최종 미배정 이미지 확인")
    unassigned_images = [img for img in all_images if not img['assigned']]
    if not unassigned_images:
        print("  - 모든 이미지가 성공적으로 배정되었습니다.")
    else:
        issue_msg = f"  - 총 {len(unassigned_images)}개의 이미지가 배정되지 않았습니다:"
        print(issue_msg)
        issues.append(issue_msg)
        for img in unassigned_images:
            img_issue_msg = f"    - 이미지 ID: {img['id']}, 페이지: {img['page_num'] + 1}, 좌표: {format_bbox(img['bbox'])}"
            print(img_issue_msg)
            issues.append(img_issue_msg)

    # =================================================================
    # [유지] 3. 마지막 문제번호 확인
    # =================================================================
    print("\n[3] 마지막 문제번호 확인")
    last_question_number = 0
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 가장 높은 Question_id를 가진 행의 Question_Number를 직접 조회
        cursor.execute("SELECT Question_Number FROM questions WHERE Question_id = (SELECT MAX(Question_id) FROM questions)")
        result = cursor.fetchone()
        
        if result and result[0]:
            last_question_number = int(result[0])
            print(f"  - 마지막 문제번호: {last_question_number}")
        else:
            print("  - 데이터베이스에서 마지막 문제번호를 찾을 수 없습니다.")
        
        conn.close()
        
    except Exception as e:
        error_msg = f"  - 마지막 문제번호 확인 중 오류 발생: {e}"
        print(error_msg)
        issues.append(error_msg)

    print("\n" + "="*27 + " 최종 할당 디버깅 완료 " + "="*27)
    
    return {
        'has_issues': bool(issues),
        'issues': issues,
        'last_question_number': last_question_number
    }

# [수정] save_questions_to_db_original 함수 시그니처 변경
def save_questions_to_db_original(db_path: str, db_name: str, questions_data: List[dict], category: str):
    """
    수정된 'questions' 테이블 스키마에 맞게 데이터를 저장합니다.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # [추가] db_name에서 Date_information 생성
    date_info_formatted = ""
    try:
        date_str = os.path.basename(db_name).replace('.db', '')
        if len(date_str) == 8 and date_str.isdigit():
            dt = datetime.datetime.strptime(date_str, '%Y%m%d')
            date_info_formatted = dt.strftime('%Y년 %m월')
    except Exception:
        date_info_formatted = "" # 파싱 실패 시 빈 값

    for q in questions_data:
        if not q.get("question_number"):
            continue

        options = q.get("Options", ["", "", "", ""])
        db_options = [(val if val else None) for val in options]

        correct_option_num = q.get("Correct_option_index")
        if correct_option_num is not None:
            correct_option_num += 1

        # [수정] INSERT 문에 새로운 칼럼과 값 추가
        cursor.execute('''
            INSERT INTO questions (
                Question_Number, Big_Question, Big_Question_Special, Question,
                Option1, Option2, Option3, Option4,
                Correct_Option, Category,
                ExamSession, Answer_description, audio, Date_information
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            q.get("question_number"),
            q.get("Big_Question") if q.get("Big_Question") else None,
            q.get("Big_Question_Special"),
            q.get("Question"),
            db_options[0],
            db_options[1],
            db_options[2],
            db_options[3],
            correct_option_num,
            category,
            None,  # ExamSession
            None,  # Answer_description
            None,  # audio
            date_info_formatted # Date_information
        ))

    conn.commit()
    conn.close()


def get_last_question_number_from_db(db_path: str) -> int:
    """
    DB 파일에서 마지막 문제번호를 추출하는 별도 함수
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT Question_Number FROM questions WHERE Question_id = (SELECT MAX(Question_id) FROM questions)")
        result = cursor.fetchone()
        
        conn.close()
        
        if result and result[0]:
            return int(result[0])
        else:
            return 0
        
    except Exception as e:
        print(f"마지막 문제번호 확인 중 오류: {e}")
        return 0


def parse_exam_pdf_with_cross_page_support(pdf_path: str, exam_name_from_filename: Optional[str]) -> Tuple[List[Dict], List[Dict]]:
    """
    PDF에서 텍스트와 이미지를 추출하고, 다단계 이미지 배정 규칙을 순차적으로 적용합니다.
    """
    if not os.path.exists(pdf_path):
        print(f"오류: 파일 경로를 찾을 수 없습니다 -> {pdf_path}")
        return [], []
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"PDF 파일을 여는 중 오류가 발생했습니다: {e}")
        return [], []

    all_images = get_all_images(doc)
    question_markers = map_all_question_starts(doc)
    if not question_markers:
        doc.close()
        return [], all_images

    question_ranges = determine_question_ranges(question_markers, doc)

    all_questions = []
    for question_range in question_ranges:
        question_data = process_question_range(doc, question_range, exam_name_from_filename, all_images)
        if question_data.get("question_number"):
            all_questions.append(question_data)

    assign_cross_column_images(doc, all_questions, all_images, exam_name_from_filename)
    assign_bottom_column_images(doc, all_questions, all_images, exam_name_from_filename)
    assign_special_question_images(all_questions, all_images)
    assign_images_below_options(all_questions, all_images)
    assign_bottom_of_page_special_images(doc, all_questions, all_images)

    doc.close()

    return all_questions, all_images

def save_questions_to_db_original(db_path: str, db_name: str, questions_data: List[dict], category: str):
    """
    수정된 'questions' 테이블 스키마에 맞게 데이터를 저장합니다.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # db_name에서 Date_information 생성
    date_info_formatted = ""
    try:
        date_str = os.path.basename(db_name).replace('.db', '')
        if len(date_str) == 8 and date_str.isdigit():
            dt = datetime.datetime.strptime(date_str, '%Y%m%d')
            date_info_formatted = dt.strftime('%Y년 %m월')
    except Exception:
        date_info_formatted = "" # 파싱 실패 시 빈 값

    for q in questions_data:
        if not q.get("question_number"):
            continue

        options = q.get("Options", ["", "", "", ""])
        db_options = [(val if val else None) for val in options]

        correct_option_num = q.get("Correct_option_index")
        if correct_option_num is not None:
            correct_option_num += 1

        # INSERT 문에 새로운 칼럼과 값 추가
        cursor.execute('''
            INSERT INTO questions (
                Question_Number, Big_Question, Big_Question_Special, Question,
                Option1, Option2, Option3, Option4,
                Correct_Option, Category,
                ExamSession, Answer_description, audio, Date_information
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            q.get("question_number"),
            q.get("Big_Question") if q.get("Big_Question") else None,
            q.get("Big_Question_Special"),
            q.get("Question"),
            db_options[0],
            db_options[1],
            db_options[2],
            db_options[3],
            correct_option_num,
            category,
            None,  # ExamSession
            None,  # Answer_description
            None,  # audio
            date_info_formatted # Date_information
        ))

    conn.commit()
    conn.close()


def get_last_question_number_from_db(db_path: str) -> int:
    """
    DB 파일에서 마지막 문제번호를 추출하는 별도 함수
    """
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 가장 높은 Question_id를 가진 행의 Question_Number를 직접 조회
        cursor.execute("SELECT Question_Number FROM questions WHERE Question_id = (SELECT MAX(Question_id) FROM questions)")
        result = cursor.fetchone()
        
        conn.close()
        
        if result and result[0]:
            return int(result[0])
        else:
            return 0
        
    except Exception as e:
        print(f"마지막 문제번호 확인 중 오류: {e}")
        return 0
    
# 1_PDF_To_DB.py 파일 맨 마지막에 추가

def find_nearest_question_above(marker_bbox: fitz.Rect, marker_page_num: int, all_questions: List[QuestionMarker], doc: fitz.Document) -> Optional[QuestionMarker]:
    """
    주어진 마커 위치 바로 위에 있는 문제 번호를 복잡한 레이아웃을 고려하여 찾습니다.
    """
    column_midpoint = doc[marker_page_num].rect.width / 2
    
    # 1. 같은 페이지, 같은 단에서 바로 위에 있는 문제 찾기
    marker_is_left = marker_bbox.x0 < column_midpoint
    candidates = []
    for q in all_questions:
        if q.page_num == marker_page_num:
            q_is_left = q.bbox.x0 < column_midpoint
            if marker_is_left == q_is_left and q.bbox.y1 < marker_bbox.y0:
                candidates.append(q)
    if candidates:
        return max(candidates, key=lambda c: c.bbox.y1) # 가장 가까운(y값이 가장 큰) 문제

    # 2. 페이지/단 경계 케이스 처리
    # 2-1. 마커가 왼쪽 단 최상단 근처에 있는 경우 -> 이전 페이지 마지막 문제
    if marker_is_left and marker_bbox.y0 < 100: # 100은 임의의 임계값
        prev_page_questions = [q for q in all_questions if q.page_num == marker_page_num - 1]
        if prev_page_questions:
            return max(prev_page_questions, key=lambda q: q.question_num)
            
    # 2-2. 마커가 오른쪽 단 최상단 근처에 있는 경우 -> 같은 페이지 왼쪽 단 마지막 문제
    if not marker_is_left and marker_bbox.y0 < 100:
        left_column_questions = [q for q in all_questions if q.page_num == marker_page_num and q.bbox.x0 < column_midpoint]
        if left_column_questions:
            return max(left_column_questions, key=lambda q: q.question_num)
            
    # 추가적인 복잡한 규칙은 여기에 계속 구현 가능...
    
    return None


def extract_category_ranges(pdf_path: str) -> List[Dict]:
    """
    [디버깅 강화] 기준 PDF 파일에서 'N과목: 과목명'을 찾아 각 과목의 문제 번호 범위를 반환합니다.
    """
    print("\n--- 카테고리 추출 디버깅 시작 ---")
    print(f"기준 파일: {os.path.basename(pdf_path)}")
    
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[실패] PDF 파일을 열 수 없습니다: {e}")
        print("--- 카테고리 추출 디버깅 종료 ---")
        return []

    # --- 디버깅 체크포인트 1: 문제 번호 마커 찾기 ---
    all_question_markers = map_all_question_starts(doc)
    if not all_question_markers:
        print("[실패] 이 PDF에서 문제 번호(예: '1.', '21.')를 하나도 찾지 못했습니다.")
        doc.close()
        print("--- 카테고리 추출 디버깅 종료 ---")
        return []
    
    last_q_num_in_doc = all_question_markers[-1].question_num
    print(f"[성공] 총 {len(all_question_markers)}개의 문제 번호를 찾았습니다. (첫 번호: {all_question_markers[0].question_num}, 마지막 번호: {last_q_num_in_doc})")

    # --- 디버깅 체크포인트 2: 과목명 마커 찾기 ---
    category_markers = []
    cat_pattern = re.compile(r"(\d)\s*과목\s*:\s*(.+)")

    for page_num in range(len(doc)):
        # get_text("text")는 간단한 텍스트 추출에 더 효과적일 수 있습니다.
        page_text = doc[page_num].get_text("text")
        for line in page_text.split('\n'):
            match = cat_pattern.match(line.strip())
            if match:
                # 과목명을 찾으면, 해당 텍스트의 위치(bbox)도 찾습니다.
                text_instances = doc[page_num].search_for(line.strip())
                if text_instances:
                    category_markers.append({
                        "num": int(match.group(1)),
                        "name": match.group(2).strip(),
                        "page_num": page_num,
                        "bbox": text_instances[0] # 첫 번째 발견된 위치의 bbox 사용
                    })

    if not category_markers:
        print("[실패] PDF에서 'N과목: 과목명' 패턴의 텍스트를 하나도 찾지 못했습니다.")
        doc.close()
        print("--- 카테고리 추출 디버깅 종료 ---")
        return []
        
    print(f"[성공] 총 {len(category_markers)}개의 과목 마커를 찾았습니다:")
    category_markers.sort(key=lambda x: x['num']) # 과목 번호 순으로 정렬
    for marker in category_markers:
        print(f"  - {marker['num']}과목: '{marker['name']}' (페이지: {marker['page_num'] + 1})")

    # --- 디버깅 체크포인트 3: 카테고리 범위 계산 ---
    category_ranges = []
    for i, cat_marker in enumerate(category_markers):
        current_cat_name = f"{cat_marker['num']}과목: {cat_marker['name']}"
        start_q_num = 1
        
        if i > 0:
            # 이전 과목의 끝 번호 + 1을 현재 과목의 시작 번호로 설정
            prev_cat_end_num = category_ranges[i-1]['end_q_num']
            start_q_num = prev_cat_end_num + 1

        end_q_num = last_q_num_in_doc
        if i + 1 < len(category_markers): # 다음 과목이 있다면
            next_cat_marker = category_markers[i+1]
            q_marker_before_next_cat = find_nearest_question_above(next_cat_marker['bbox'], next_cat_marker['page_num'], all_question_markers, doc)
            
            if q_marker_before_next_cat:
                end_q_num = q_marker_before_next_cat.question_num
                print(f"  - '{next_cat_marker['name']}' 앞 문제({end_q_num}번)를 기준으로 '{cat_marker['name']}'의 끝 범위를 설정했습니다.")
            else:
                print(f"  - 경고: '{next_cat_marker['name']}' 위의 문제 번호를 찾지 못해, 끝 범위를 문서의 마지막 번호로 설정합니다.")

        category_ranges.append({
            "name": current_cat_name,
            "start_q_num": start_q_num,
            "end_q_num": end_q_num
        })
        
    doc.close()
    print("--- 카테고리 추출 디버깅 종료 ---")
    return category_ranges

# in 1_PDF_To_DB.py

def apply_categories_to_db(db_path: str, category_ranges: List[Dict], exam_name: str):
    """
    [수정] 주어진 DB 파일에 접속하여, 세 가지 규칙에 따라 카테고리명을 가공하고 업데이트합니다.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB 파일을 찾을 수 없습니다: {db_path}")

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    try:
        for cat_range in category_ranges:
            original_cat_name = cat_range['name']
            start_q = cat_range['start_q_num']
            end_q = cat_range['end_q_num']

            # --- [추가] 카테고리명 가공 로직 ---
            final_cat_name = original_cat_name
            
            # 규칙 1: "N과목: " 접두사 제거
            match = re.match(r"\d+\s*과목\s*:\s*(.+)", original_cat_name)
            if match:
                final_cat_name = match.group(1).strip()

            # 규칙 2 & 3: 특정 이름일 경우 시험명으로 대체
            if final_cat_name in ["과목 구분 없음", "임의 구분"]:
                final_cat_name = exam_name
            # ------------------------------------

            query = "UPDATE questions SET Category = ? WHERE CAST(Question_Number AS INTEGER) BETWEEN ? AND ?"
            cursor.execute(query, (final_cat_name, start_q, end_q))
        
        conn.commit()
    
    except Exception as e:
        conn.rollback()
        raise e
    
    finally:
        conn.close()
        
        
# 독립적인 테스트를 위한 main 실행 블록
if __name__ == '__main__':
    # 이 파일을 직접 실행할 경우, 아래 코드가 실행됩니다.
    # 사용법: python 1_PDF_To_DB.py <PDF파일_경로>
    import sys
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        
        if not os.path.exists(pdf_path):
            print(f"오류: 파일을 찾을 수 없습니다 - {pdf_path}")
        else:
            # 테스트용 변수 설정
            db_name = f"{os.path.splitext(os.path.basename(pdf_path))[0]}_test.db"
            db_path = os.path.join(os.path.dirname(pdf_path), db_name)
            exam_category = "Test Category"
            exam_name_from_filename = "TestExam"

            print(f"테스트 시작: {pdf_path}")
            
            # 데이터베이스 설정
            setup_database_original(db_path)
            
            # 파싱 실행
            extracted_questions, all_images = parse_exam_pdf_with_cross_page_support(pdf_path, exam_name_from_filename)
            
            if extracted_questions:
                # DB에 저장
                save_questions_to_db_original(db_path, db_name, extracted_questions, exam_category)
                print(f"성공: 총 {len(extracted_questions)}개의 문제를 추출하여 '{db_name}'에 저장했습니다.")
                
                # 최종 디버그 체크
                final_assignment_debug_check(db_path, all_images)
            else:
                print("오류: PDF에서 문제를 추출하지 못했습니다.")
    else:
        print("사용법: python 1_PDF_To_DB.py <PDF파일_경로>")