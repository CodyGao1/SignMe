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
  if (id === 25) return ' '; // Map 25 to a blank space
  return String.fromCharCode(65 + id); // Map 0-24 to A-Y
}

const App = () => {
  const [loading, setLoading] = useState({ loading: true, progress: 0 });
  const [classHistory, setClassHistory] = useState([]);
  const [outputText, setOutputText] = useState('');
  const latestDetectionRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const webcam = new Webcam();
  const modelName = "ASL";
  const threshold = 0.85;

  const detectFrame = async (model) => {
    const model_dim = [512, 512];
    tf.engine().startScope();
    const input = tf.tidy(() => {
      const img = tf.image
                .resizeBilinear(tf.browser.fromPixels(videoRef.current), model_dim)
                .div(255.0)
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
        latestDetectionRef.current = class_detect[0][0];
    }

    renderBoxes(canvasRef, threshold, boxes, scores, class_detect);
    tf.dispose(res);
    tf.dispose(input);

    requestAnimationFrame(() => detectFrame(model));
    tf.engine().endScope();
  };

  useEffect(() => {
    const interval = setInterval(() => {
        if (latestDetectionRef.current !== null && latestDetectionRef.current !== 25) {
            setClassHistory(currentHistory => [...currentHistory, latestDetectionRef.current]);
            setOutputText(currentText => currentText + mapIdToLetter(latestDetectionRef.current));
            latestDetectionRef.current = null;
        }
    }, 2000); // Update history and output text every 2 seconds

    return () => clearInterval(interval);
  }, []);

  const clearOutput = () => {
    setOutputText('');
    setClassHistory([]);
  };

  return (
    <div className="App">
      <h2>Object Detection Using YOLOv7 & Tensorflow.js</h2>
      {loading.loading ? (
        <Loader>Loading model... {(loading.progress * 100).toFixed(2)}%</Loader>
      ) : (
        <p> </p>
      )}

      <div className="content">
        <video autoPlay playsInline muted ref={videoRef} id="frame" />
        <canvas width={512} height={512} ref={canvasRef} />
      </div>
      
      <div className="output-area" style={{ background: 'transparent', color: 'black', fontSize: '24px', whiteSpace: 'pre-wrap' }}>
        {outputText}
      </div>
      
      <button onClick={clearOutput}>Clear</button>
    </div>
  );
};

export default App;
