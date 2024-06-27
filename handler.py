import os
import pathlib
from collections import defaultdict
import runpod
from pdf2image import convert_from_path
from runpod.serverless.utils import rp_download
import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd
from ultralytics import YOLO

from pre_process import Preprocess
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load the model into memory
table = YOLO("table-detection.pt")
model = YOLO("best-yolov-india-us.pt")

# Define the class names
class_names = {
    0: 'table',
    1: 'table column',
    2: 'table column header',
    3: 'table projected row header',
    4: 'table row',
    5: 'table spanning cell'
}

# Initialize PaddleOCR reader
ocr = PaddleOCR(lang='en', use_gpu=True)


def get_cell_coordinates_by_row(table_data):
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    cell_coordinates = []
    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})
        row_cells.sort(key=lambda x: x['column'][0])
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    cell_coordinates.sort(key=lambda x: x['row'][1])
    return cell_coordinates


def apply_ocr(cell_coordinates, cropped_table):
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []
        for cell_idx, cell in enumerate(row["cells"]):
            cell_bbox = cell["cell"]
            cell_image = cropped_table.crop(cell_bbox)
            cell_image_np = np.array(cell_image)
            result = ocr.ocr(cell_image_np, cls=True)
            if result and result[0]:
                text = " ".join([line[-1][0] for line in result[0] if line and line[-1]])
                row_text.append(text)
            else:
                row_text.append("")
        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)
        data[idx] = row_text
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data
    return data


def handler(job):
    job_input = job.get('input', {})

    if not job_input.get("file_path", False):
        return {
            "error": "Input is missing the 'file_path' key. Please include a file_path and retry your request."
        }
    page_num = job_input.get('page_num', 5)
    file_path = job_input.get("file_path")
    # file_path = rp_download.file(file_path).get('file_path')
    file_extension = pathlib.Path(file_path).suffix.strip().lower()
    if not file_extension.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.pdf')):
        return {"error": "Provide a valid file type"}

    images = convert_from_path(file_path, dpi=250, fmt="jpeg", first_page=0, last_page=page_num)

    with pd.ExcelWriter('output.xlsx') as writer:
        for page_index, image in enumerate(images):
            results = table(image, conf=0.5)

            if len(results) == 0:
                return {"error": f"No tables detected in the image on page {page_index + 1}."}

            detected_table = results[0]
            if len(detected_table.boxes.xyxy) == 0:
                return {"error": f"No tables detected in the image on page {page_index + 1}."}

            # Assuming there's only one table detection, take the first detection
            table_bbox = detected_table.boxes.xyxy[0].tolist()

            # Crop the table from the image
            cropped_table = image.crop(table_bbox)

            # Run the model on the cropped table
            results = model(cropped_table, conf=0.3)
            detected_boxes = defaultdict(list)
            for result in results:
                for cls, confidence, bbox in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
                    class_name = class_names.get(int(cls), "Unknown")
                    bbox_list = bbox.tolist()
                    detected_boxes[class_name].append((confidence.item(), bbox_list))

            nms_handler = Preprocess(iou_threshold=0.4)
            nms_results = []
            for class_name, boxes in detected_boxes.items():
                nms_boxes = nms_handler.apply_nms(boxes)
                for box in nms_boxes:
                    nms_results.append({'label': class_name, 'score': box[0], 'bbox': box[1]})

            cell_coordinates = get_cell_coordinates_by_row(nms_results)
            ocr_data = apply_ocr(cell_coordinates, cropped_table)

            df = pd.DataFrame.from_dict(ocr_data, orient='index')
            df = df.rename(columns=df.iloc[0]).drop(df.index[0])

            # Save the DataFrame to an Excel file with the sheet name as the page number
            df.to_excel(writer, sheet_name=f'Page_{page_index + 1}', index=False)

    return {"refresh_worker": True, "job_results": 'output.xlsx'}


if __name__ == "__main__":
    job = {
        "input": {
            "file_path": "sbi.pdf",
            "page_num": 1

        }
    }
    result = handler(job)
    print(result)

#runpod.serverless.start({"handler": handler})
