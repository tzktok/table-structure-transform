import numpy as np
from paddleocr import PaddleOCR
from PIL import Image
from tqdm.auto import tqdm
import pandas as pd
from ultralytics import YOLO
import os
from collections import defaultdict
from pre_process import Preprocess

# Load YOLO model
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
cropped_table = Image.open("test21.jpg")
# Run batched inference on an image
results = model(cropped_table,conf=0.3)


# Create a dictionary to store detected bounding boxes per class
detected_boxes = defaultdict(list)

# Process results
for result in results:
    for cls, confidence, bbox in zip(result.boxes.cls, result.boxes.conf, result.boxes.xyxy):
        class_name = class_names.get(int(cls), "Unknown")
        bbox_list = bbox.tolist()  # Convert tensor to list
        detected_boxes[class_name].append((confidence.item(), bbox_list))

nms_handler = Preprocess(iou_threshold=0.4)

nms_results = []
for class_name, boxes in detected_boxes.items():
    nms_boxes = nms_handler.apply_nms(boxes)
    for box in nms_boxes:
        nms_results.append({
            'label': class_name,
            'score': box[0],
            'bbox': box[1]
        })

# Define the function to get cell coordinates by row
def get_cell_coordinates_by_row(table_data):
    # Extract rows and columns
    rows = [entry for entry in table_data if entry['label'] == 'table row']
    columns = [entry for entry in table_data if entry['label'] == 'table column']

    # Sort rows and columns by their Y and X coordinates, respectively
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    # Function to find cell coordinates
    def find_cell_coordinates(row, column):
        cell_bbox = [column['bbox'][0], row['bbox'][1], column['bbox'][2], row['bbox'][3]]
        return cell_bbox

    # Generate cell coordinates and count cells in each row
    cell_coordinates = []

    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({'column': column['bbox'], 'cell': cell_bbox})

        # Sort cells in the row by X coordinate
        row_cells.sort(key=lambda x: x['column'][0])

        # Append row information to cell_coordinates
        cell_coordinates.append({'row': row['bbox'], 'cells': row_cells, 'cell_count': len(row_cells)})

    # Sort rows from top to bottom
    cell_coordinates.sort(key=lambda x: x['row'][1])

    return cell_coordinates


# Pass the final results into the function
cell_coordinates = get_cell_coordinates_by_row(nms_results)

# Initialize PaddleOCR reader
ocr = PaddleOCR(lang='en',use_gpu=True)
def apply_ocr(cell_coordinates):

    # Let's OCR row by row
    data = dict()
    max_num_columns = 0
    for idx, row in enumerate(tqdm(cell_coordinates)):
        row_text = []
        for cell_idx, cell in enumerate(row["cells"]):
            # Crop cell out of image
            cell_bbox = cell["cell"]
            cell_image = cropped_table.crop(cell_bbox)

            # Convert cell image to numpy array
            cell_image_np = np.array(cell_image)

            # Apply OCR
            result = ocr.ocr(cell_image_np, cls=True)
            if result and result[0]:
                # Extract text from OCR result
                text = " ".join([line[-1][0] for line in result[0] if line and line[-1]])
                row_text.append(text)
            else:
                row_text.append("")

        if len(row_text) > max_num_columns:
            max_num_columns = len(row_text)

        data[idx] = row_text

    print("Max number of columns:", max_num_columns)

    # Pad rows which don't have max_num_columns elements
    # to make sure all rows have the same number of columns
    for row, row_data in data.copy().items():
        if len(row_data) != max_num_columns:
            row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
        data[row] = row_data

    return data

# Apply OCR to the cell coordinates
ocr_data = apply_ocr(cell_coordinates)

# Convert the OCR results to a DataFrame
df = pd.DataFrame.from_dict(ocr_data, orient='index')
df = df.rename(columns=df.iloc[0]).drop(df.index[0])
print(df)

# Save the DataFrame to an Excel file
df.to_excel("output.xlsx", index=False)

# Print the OCR results
for row, row_data in ocr_data.items():
    print(row_data)
