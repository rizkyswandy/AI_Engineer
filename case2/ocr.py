import easyocr

reader = easyocr.Reader(['en'], gpu=False)

def detect_image(image_path):
    result = reader.readtext(image_path)
    
    return result
    
if __name__ == "__main__":
    image_path = '../invoice.png'  
    ocr_result = detect_image(image_path)
    
    full_text = ""
    
    for detection in ocr_result:
        text = detection[1]
        confidence = detection[2]
        full_text += text + " "
    
    print(f"Full text detected: {full_text.strip()}")
    