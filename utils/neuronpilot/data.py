import numpy as np
import cv2

def convert_to_binary(image_data, dtype = np.float32, handeler = 0):
    #image_data = cv2.resize(rgb_image, input_shape, interpolation=cv2.INTER_AREA)
    #image_data = image_data.astype(dtype)
    #image_data = np.expand_dims(image_data, axis=0)   #add batch_size dimension: (1, height, width, 3)
    binary_file = f"input{handeler}.bin"
    with open(binary_file, "wb") as f:
        f.write(image_data.tobytes())
    return binary_file
    
def conert_to_numpy(output_handlers_with_shape, dtype = np.float32):
    outputs = []
    for output_handler in output_handlers_with_shape:
        with open(output_handler, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.float32)
            print(data.shape)
            data = np.reshape(data, output_handlers_with_shape[output_handler])
            outputs.append(data)
    return np.array(data)
