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
            './saved_model', ['serve'], 'serving_default');
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
    let imageTensor = (getPhotoTensor('./image.jpeg'));
    imageTensor = tf.image.resizeBilinear(imageTensor, [640,640])
    const input = imageTensor.div(255.0).expandDims(0).toFloat();
    // Feed the image tensor into the model for inference.
    let s = tf.util.now();
    objectDetectionModel.predict({'x': input});
    console.log( tf.util.now()-s)
    s = tf.util.now();
    objectDetectionModel.predict({'x': input});
    console.log( tf.util.now()-s)
    s = tf.util.now();
    objectDetectionModel.predict({'x': input});
    console.log( tf.util.now()-s)
    s = tf.util.now();
    objectDetectionModel.predict({'x': input});
    console.log( tf.util.now()-s)
    s = tf.util.now();
    objectDetectionModel.predict({'x': input});
    console.log( tf.util.now()-s)
    s = tf.util.now();
    objectDetectionModel.predict({'x': input});
    console.log( tf.util.now()-s)
    s = tf.util.now();
    objectDetectionModel.predict({'x': input});
    console.log( tf.util.now()-s)
    s = tf.util.now();
    objectDetectionModel.predict({'x': input});
    console.log( tf.util.now()-s)
    s = tf.util.now();
    objectDetectionModel.predict({'x': input});
    console.log( tf.util.now()-s)
    s = tf.util.now();
    objectDetectionModel.predict({'x': input});
    let outputTensor = objectDetectionModel.predict({'x': input});
    // Parse the model output to get meaningful result(get detection class and
    // object location).
    const scores = await outputTensor.output_1.arraySync();
    const boxes = await outputTensor.output_0.arraySync();
    const names = await outputTensor.output_2.arraySync();
    const nums = await outputTensor.output_3.arraySync()[0];

    const endTime = tf.util.now();
    outputTensor.output_0.dispose();
    outputTensor.output_1.dispose();
    outputTensor.output_2.dispose();
    outputTensor.output_3.dispose();
    const detectedBoxes = [];
    const detectedNames = [];
    for (let i = 0; i < scores[0].length; i++) {
      if (scores[0][i] > 0.3) {
        detectedBoxes.push(boxes[0][i]);
        detectedNames.push([names[0][i]]);
      }
    }
    console.log({
        boxes: detectedBoxes,
        names: detectedNames,
        inferenceTime: endTime - startTime
      })
}

loadModel();

//https://pan.baidu.com/s/1GZzVSrdUc2XvCBM3Hog7Ug
//saku

