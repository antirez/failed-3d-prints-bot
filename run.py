import onnxruntime as ort
import onnx
from PIL import Image
import numpy as np
import time

def load_image_and_preprocess(image_path):
    # Load the image
    image = Image.open(image_path)

    # Resize the image to the required size (416x416)
    resized_image = image.resize((416, 416))

    # Convert the image to RGB (3 channels)
    if resized_image.mode != 'RGB':
        resized_image = resized_image.convert('RGB')

    # Convert the image to a numpy array and normalize it
    image_array = np.array(resized_image).astype(np.float32) / 255.0

    # Change the shape of the numpy array to match the input shape of the model ([1, 3, 416, 416])
    # This involves moving the color channel to the second dimension and adding an extra dimension at the start
    input_tensor = np.transpose(image_array, (2, 0, 1))
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor

def process_outputs(outputs, image_width=416, image_height=416, threshold=0.1):
    # Extract the boxes and confidence scores from the outputs
    boxes = outputs[0][0]
    confs = outputs[1][0]

    # Iterate over boxes and confidences
    for i, box in enumerate(boxes):
        # Extract the confidence score
        conf = confs[i][0]

        # Check if the confidence score is greater than the threshold
        if conf > threshold:
            # Extract and adjust bounding box coordinates
            x_min, y_min, x_max, y_max = box[0]
            x_min_pixel = int(x_min * image_width)
            y_min_pixel = int(y_min * image_height)
            x_max_pixel = int(x_max * image_width)
            y_max_pixel = int(y_max * image_height)

            # Print the adjusted bounding box and its confidence score
            print(f"Box: {x_min_pixel}, {y_min_pixel}, {x_max_pixel}, {y_max_pixel}, Confidence: {conf}")

# Load the model and create InferenceSession
model_path = "./model-weights-5a6b1be1fa.onnx"

# Print details about the inputs
model = onnx.load(model_path)
print("Model Inputs:")
for input in model.graph.input:
    print("Name:", input.name)
    # The shape of the input tensor
    print("Shape:", [dim.dim_value for dim in input.type.tensor_type.shape.dim])
    # Data type of the input tensor
    print("Data type:", input.type.tensor_type.elem_type)

print("\n")
print("Model Outputs:")
for output in model.graph.output:
    print("Name:", output.name)
    # The shape of the input tensor
    print("Shape:", [dim.dim_value for dim in output.type.tensor_type.shape.dim])
    # Data type of the input tensor
    print("Data type:", output.type.tensor_type.elem_type)

session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
# Load and preprocess the input image inputTensor
inputTensor = load_image_and_preprocess("test.png")

# Run inference
start_time = time.time()
outputs = session.run(None, {"input": inputTensor})
end_time = time.time()
duration_time = (end_time - start_time)*1000
print("Running time in ms: ", duration_time)

process_outputs(outputs)
