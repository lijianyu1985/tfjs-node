const tf = require('@tensorflow/tfjs-node');
// const tf = require('@tensorflow/tfjs-node-gpu');
const fs = require('fs');

let objectDetectionModel;

async function loadModel() {
    // Warm up the model
    if (!objectDetectionModel) {
        // Load the TensorFlow SavedModel through tfjs-node API. You can find more
        // details in the API documentation:
        // https://js.tensorflow.org/api_node/1.3.1/#node.loadSavedModel
        objectDetectionModel = await tf.node.loadSavedModel(
            './attributes_model', ['serve'], 'serving_default');
    }
    // const tempTensor = tf.zeros([1, 2, 2, 3]).toFloat();
    // objectDetectionModel.predict(tempTensor);
    predict();
}

function getPhotoTensor(pathPhoto) {
    const f = fs.readFileSync(pathPhoto);
    const t = tf.node.decodeImage(new Uint8Array(f));
    return t;
  }

async function predict() {

    const startTime = tf.util.now();
    let imageTensor = (getPhotoTensor('./000006.jpg'));
    imageTensor = tf.image.resizeBilinear(imageTensor, [256,128])
    imageTensor = tf.reverse(imageTensor, -1);

    const input = imageTensor.div(255.0).expandDims(0).toFloat();
    // Feed the image tensor into the model for inference.
    let s = tf.util.now();
    objectDetectionModel.predict(input);
    console.log( tf.util.now()-s)
    s = tf.util.now();
    objectDetectionModel.predict(input);
    console.log( tf.util.now()-s)
    s = tf.util.now();
    objectDetectionModel.predict(input);
    console.log( tf.util.now()-s)
    s = tf.util.now();
    objectDetectionModel.predict(input);
    console.log( tf.util.now()-s)
    s = tf.util.now();
    objectDetectionModel.predict(input);
    console.log( tf.util.now()-s)
    s = tf.util.now();
    objectDetectionModel.predict(input);
    console.log( tf.util.now()-s)
    s = tf.util.now();
    objectDetectionModel.predict(input);
    console.log( tf.util.now()-s)
    s = tf.util.now();
    objectDetectionModel.predict(input);
    console.log( tf.util.now()-s)
    s = tf.util.now();
    objectDetectionModel.predict(input);
    console.log( tf.util.now()-s)
    s = tf.util.now();
    objectDetectionModel.predict(input);
    let outputTensor = objectDetectionModel.predict(input);
    // Parse the model output to get meaningful result(get detection class and
    // object location).
    const result = await outputTensor.arraySync();

    const endTime = tf.util.now();
    console.log({
        result: result,
        inferenceTime: endTime - startTime
      })
}

loadModel();

