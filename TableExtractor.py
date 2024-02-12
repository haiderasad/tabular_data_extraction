import torch
from torchvision import transforms
from transformers import AutoModelForObjectDetection
import easyocr
from PIL import ImageDraw
import numpy as np
import csv
import pandas as pd

class OCRProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Define transforms
        self.detection_transform = transforms.Compose([
            self.MaxResize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.structure_transform = transforms.Compose([
            self.MaxResize(1000),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        # Load models
        self.model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm").to(self.device)
        self.structure_model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition-v1.1-all").to(self.device)
        
        # Load EasyOCR reader
        self.reader = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
    
    class MaxResize(object):
        def __init__(self, max_size=800):
            self.max_size = max_size

        def __call__(self, image):
            width, height = image.size
            current_max_size = max(width, height)
            scale = self.max_size / current_max_size
            resized_image = image.resize((int(round(scale*width)), int(round(scale*height))))
            return resized_image
    
    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self,x):
        x_c, y_c, w, h = x.unbind(-1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)
    
    def rescale_bboxes(self,out_bbox, size):
        width, height = size
        boxes = self.box_cxcywh_to_xyxy(out_bbox)
        boxes = boxes * torch.tensor([width, height, width, height], dtype=torch.float32)
        return boxes
    
    def outputs_to_objects(self,outputs, img_size, id2label):
        m = outputs.logits.softmax(-1).max(-1)
        pred_labels = list(m.indices.detach().cpu().numpy())[0]
        pred_scores = list(m.values.detach().cpu().numpy())[0]
        pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
        pred_bboxes = [elem.tolist() for elem in self.rescale_bboxes(pred_bboxes, img_size)]

        objects = []
        for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
            class_label = id2label[int(label)]
            if not class_label == 'no object':
                objects.append({'label': class_label, 'score': float(score),
                                'bbox': [float(elem) for elem in bbox]})

        return objects
    
    
    def detect_and_crop_table(self,image):
        # prepare image for the model
        pixel_values = self.detection_transform(image).unsqueeze(0).to(self.device)

        # forward pass
        with torch.no_grad():
            outputs = self.model(pixel_values)

        # postprocess to get detected tables
        id2label = self.model.config.id2label
        id2label[len(self.model.config.id2label)] = "no object"
        detected_tables = self.outputs_to_objects(outputs, image.size, id2label)

        # crop first detected table out of image
        cropped_table = image.crop(detected_tables[0]["bbox"])

        return cropped_table
    
    def recognize_table(self,image):
        # prepare image for the model
        # pixel_values = structure_processor(images=image, return_tensors="pt").pixel_values
        pixel_values = self.structure_transform(image).unsqueeze(0).to(self.device)

        # forward pass
        with torch.no_grad():
            outputs = self.structure_model(pixel_values)

        # postprocess to get individual elements
        id2label = self.structure_model.config.id2label
        id2label[len(self.structure_model.config.id2label)] = "no object"
        cells = self.outputs_to_objects(outputs, image.size, id2label)

        # visualize cells on cropped table
        draw = ImageDraw.Draw(image)

        for cell in cells:
            draw.rectangle(cell["bbox"], outline="red")
            
        return image, cells
    
    def get_cell_coordinates_by_row(self,table_data):
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
    

    def apply_ocr(self,cell_coordinates, cropped_table):
        # let's OCR row by row
        data = dict()
        max_num_columns = 0
        for idx, row in enumerate(cell_coordinates):
            row_text = []
            for cell in row["cells"]:
                # crop cell out of image
                cell_image = np.array(cropped_table.crop(cell["cell"]))
                # apply OCR
                result = self.reader.readtext(np.array(cell_image))
                if len(result) > 0:
                    text = " ".join([x[1] for x in result])
                    row_text.append(text)

            if len(row_text) > max_num_columns:
                max_num_columns = len(row_text)
            
            data[str(idx)] = row_text

        # pad rows which don't have max_num_columns elements
        # to make sure all rows have the same number of columns
        for idx, row_data in data.copy().items():
            if len(row_data) != max_num_columns:
                row_data = row_data + ["" for _ in range(max_num_columns - len(row_data))]
            data[str(idx)] = row_data

        # write to csv
        with open('output.csv','w') as result_file:
            wr = csv.writer(result_file, dialect='excel')
        
            for row, row_text in data.items():
                wr.writerow(row_text)

        # return as Pandas dataframe
        df = pd.read_csv('output.csv')

        return df, data
    
    def process_pdf(self,image):
        
        if image.mode == 'RGBA':
            image = image.convert('RGB')
            
        cropped_table = self.detect_and_crop_table(image)

        image, cells = self.recognize_table(cropped_table)

        cell_coordinates = self.get_cell_coordinates_by_row(cells)

        df, data = self.apply_ocr(cell_coordinates, image)

        return image, self.convert_to_json_structure(data)

    def convert_to_json_structure(self,data_dict):
        
        # Extract headers
        headers = data_dict['0']
        # Create a list to hold the row dictionaries
        json_list = []
        
        # Iterate over the rows, starting from the first row after the headers
        for key in sorted(data_dict.keys())[1:]:  # Skip the headers
            row_data = data_dict[key]
            # Create a dictionary for the row ensuring the order is maintained
            row_dict = {headers[i]: row_data[i] for i in range(len(headers))}
            # Append the row dictionary to the list
            json_list.append(row_dict)
        
        return json_list