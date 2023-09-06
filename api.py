from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# Load the TensorFlow Lite model
model_path = 'model.tflite'
interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()

# Load labels
labels_path = 'labels.txt'
with open(labels_path, 'r') as file:
    labels = [line.strip() for line in file.readlines()]

def classify_image(image_data):
    # Preprocess the image (you may need to adjust this based on your model requirements)
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]['shape']
    image = tf.image.decode_image(image_data)
    image = tf.image.resize(image, (input_shape[1], input_shape[2]))
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  # Normalize the image

    # Perform inference
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    output_details = interpreter.get_output_details()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Post-process the results
    class_index = np.argmax(output)
    confidence = output[0][class_index]

    return labels[class_index], confidence

@app.route('/classify', methods=['POST'])
def classify():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400

        image = request.files['image'].read()
        class_name, confidence = classify_image(image)
        confidence_percentage = round(float(confidence) * 100, 3)  # Convert to percentage and round
        return jsonify({'class_name': class_name, 'confidence': confidence_percentage}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False,host='0.0.0.0')
