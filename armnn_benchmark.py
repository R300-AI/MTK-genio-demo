import argparse, time
import tensorflow as tf

parser = argparse.ArgumentParser()

parser.add_argument("-m", "--tflite_model", type=str, help="Path to .tflite")
parser.add_argument("-d", "--device", default='GpuAcc', choices=['CpuAcc', 'GpuAcc'], type=str, help="Device name for acceleration")
parser.add_argument("-t", "--iteration", default=10, type=int, help="Test How Many Times?")
args = parser.parse_args()

# Initialize Interpreter
Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate
armnn_delegate = load_delegate( library="/home/ubuntu/armnn/ArmNN-linux-aarch64/libarmnnDelegate.so",
                                       options={"backends": args.device, "logging-severity":"info"})
interpreter = Interpreter(model_path=args.tflite_model, 
                          experimental_delegates=[armnn_delegate])
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Forward Propagation Speed
inputs = np.random.rand(*input_details[0]['shape']).astype(input_details[0]['dtype'])
for _ in range(args.iteration):
  t = time.time()
  interpreter.set_tensor(input_details[0]['index'], inputs)
  interpreter.invoke()
  outputs = interpreter.get_tensor(output_details[0]['index'])
  
  print("Inference Speed:", (time.time() - t) * 1000, "ms")
print('Finished')
