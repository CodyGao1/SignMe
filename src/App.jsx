import React, { useState, useEffect, useRef } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import Loader from "./components/loader";
import { Webcam } from "./utils/webcam";
import { renderBoxes } from "./utils/renderBox";
import { non_max_suppression } from "./utils/nonMaxSuppression";
import "./style/App.css";

function shortenedCol(arrayofarray, indexlist) {
  return arrayofarray.map(array => indexlist.map(idx => array[idx]));
}

function mapIdToLetter(id) {
  return id === 25 ? ' ' : String.fromCharCode(65 + id); // Map 0-24 to A-Y, 25 to ' '
}

function mostCommon(arr) {
  // The most common detection calculation is simplified and improved
  const frequencyMap = {};
  arr.forEach((item) => {
    frequencyMap[item] = (frequencyMap[item] || 0) + 1;
  });
  return Object.keys(frequencyMap).reduce((a, b) => frequencyMap[a] > frequencyMap[b] ? a : b);
}


const App = () => {
  const [loading, setLoading] = useState(true);
  const [outputText, setOutputText] = useState('');
  const [detectionFrequency, setDetectionFrequency] = useState({});
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const webcam = new Webcam();
  const modelName = "ASL";
  const threshold = 0.85;
  const detectionThreshold = 30;

  const detectFrame = async (model) => {
    const model_dim = [512, 512];
    tf.engine().startScope();
    const input = tf.tidy(() => {
      const img = tf.browser.fromPixels(videoRef.current)
        .resizeBilinear([model_dim[0], model_dim[1]])
        .div(tf.scalar(255))
        .transpose([2, 0, 1])
        .expandDims(0);
      return img;
    });

    const res = model.execute(input);
    const predictions = res.arraySync();

    var detections = non_max_suppression(predictions[0]);
    const boxes = shortenedCol(detections, [0,1,2,3]);
    const scores = shortenedCol(detections, [4]);
    const class_detect = shortenedCol(detections, [5]);

    if (class_detect.length > 0) {
      const detectedClass = class_detect[0][0];
      setDetectionBuffer((prevBuffer) => [...prevBuffer, detectedClass]);
    }

    renderBoxes(canvasRef, threshold, boxes, scores, class_detect);
    tf.dispose(res);
    tf.dispose(input);

    requestAnimationFrame(() => detectFrame(model));
    tf.engine().endScope();
  };

  useEffect(() => {
    tf.loadGraphModel(`${window.location.origin}/${modelName}_web_model/model.json`)
      .then(model => {
        setLoading(false);
        webcam.open(videoRef, () => detectFrame(model));
      });
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      if (detectionBuffer.length > 0) {
        const mostCommonDetection = mostCommon(detectionBuffer);
        if (mostCommonDetection !== undefined) {
          const frequency = detectionBuffer.filter(x => x === mostCommonDetection).length;
          if (frequency > detectionThreshold) {
            const newLetter = mapIdToLetter(parseInt(mostCommonDetection, 10));
            setOutputText(currentText => currentText + newLetter);
          }
        }
        setDetectionBuffer([]); // Clear the buffer after processing
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [detectionFrequency]);

  const clearOutput = () => {
    setOutputText('');
    setDetectionFrequency({});
  };

  return (
    <div className="App">
      <h2>SignMe</h2>
      {loading ? (
        <Loader>Loading model...</Loader>
      ) : (
        <>
          <div className="content">
            <video autoPlay playsInline muted ref={videoRef} id="frame" />
            <canvas width={512} height={512} ref={canvasRef} />
          </div>
          
          <div className="output-area">
            {outputText}
          </div>
          
          {outputText && (
            <button onClick={clearOutput} className="clear-button">
              Clear
            </button>
          )}
        </>
      )}
    </div>
  );
};

export default App;
