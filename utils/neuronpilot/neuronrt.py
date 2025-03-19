import subprocess

def predict(input_handlers, output_handlers, compiled_dla_model, boost_value = 100):
    commands = ["neuronrt",  
                "-m",  "hw",  
                "-a",  compiled_dla_model,
                "-c",  "1",         # Repeat the inference <num> times. It can be used for profiling.
                "-b",  str(boost_value),     
                "-i",  ' -i '.join(input_handlers),  
                "-o",  ' -o '.join(output_handlers)]
                      
    p = subprocess.Popen(commands, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
