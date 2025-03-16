                #delegate = {"Linux": "libedgetpu.so.1", "Darwin": "libedgetpu.1.dylib", "Windows": "edgetpu.dll"}[
                #    platform.system()
                #]
                #interpreter = Interpreter(
                #    model_path=w,
                #    experimental_delegates=[load_delegate(delegate, options={"device": device})],
                #)
                armnn_delegate = load_delegate(
                    library="/home/ubuntu/armnn/ArmNN-linux-aarch64/libarmnnDelegate.so",
                    options={"backends": 'GpuAcc', "logging-severity":"info"}
                )
                interpreter = Interpreter(
                    model_path=args.tflite_model, 
                    experimental_delegates=[armnn_delegate]
                )    
