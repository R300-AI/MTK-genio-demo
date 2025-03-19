import numpy as np
import cv2

def convert_to_binary(image_data, handeler = 0):
    binary_file = f"./bin/input{handeler}.bin"
    image_data.astype(np.float32).tofile(binary_file)
    return binary_file
    
def conert_to_numpy(output_handlers_with_shape, dtype = np.float32):
    outputs = []
    for output_handler in output_handlers_with_shape:
        data = np.fromfile(output_handler, dtype=dtype)
        outputs.append(np.reshape(data, output_handlers_with_shape[output_handler]))
    return np.array(outputs)[0]
