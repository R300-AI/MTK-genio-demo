import numpy as np
import tflite_runtime.interpreter as tflite

#require python 3.9
# Load TFLite model and allocate tensors.
# (if you are using the complete tensorflow package you can find load_delegate in tf.experimental.load_delegate)
armnn_delegate = tflite.load_delegate( library="/home/ubuntu/armnn/ArmNN-linux-aarch64/libarmnnDelegate.so",
                                       options={"backends": "CpuAcc,GpuAcc,CpuRef", "logging-severity":"info"})
# Delegates/Executes all operations supported by Arm NN to/with Arm NN
interpreter = tflite.Interpreter(model_path="/home/ubuntu/armnn/delegate/python/test/test_data/mock_model.tflite", 
                                 experimental_delegates=[armnn_delegate])
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']
input_data = np.array(np.random.random_sample(input_shape), dtype=np.uint8)
interpreter.set_tensor(input_details[0]['index'], input_data)

interpreter.invoke()

# Print out result
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)
