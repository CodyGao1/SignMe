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
  const [loading, setLoading] = useState(true);
  const [progress, setProgress] = useState(0);
  const [outputText, setOutputText] = useState('');
  const latestDetectionRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const webcam = new Webcam();
  const modelName = "ASL";
  const threshold = 0.85;

  const detectFrame = async (model) => {
    // Detection logic
  };

  useEffect(() => {
    // Model loading and detection setup logic
    tf.loadGraphModel(`${window.location.origin}/${modelName}_web_model/model.json`, {
      onProgress: (fractions) => {
        setProgress(fractions);
      },
    }).then(async (model) => {
      const dummyInput = tf.ones(model.inputs[0].shape);
      await model.executeAsync(dummyInput).then((warmupResult) => {
        tf.dispose(warmupResult);
        tf.dispose(dummyInput);

        setLoading(false);
        webcam.open(videoRef, () => detectFrame(model));
      });
    });
  }, []);

  useEffect(() => {
    // Interval for updating output text logic
  }, []);

  const clearOutput = () => {
    setOutputText('');
  };

  return (
    <div className="App">
      <h2>Object Detection Using YOLOv7 & Tensorflow.js</h2>
      {loading ? (
        <Loader>Loading model... {(progress * 100).toFixed(2)}%</Loader>
      ) : (
        <>
          <div className="content">
            <video autoPlay playsInline muted ref={videoRef} id="frame" />
            <canvas width={512} height={512} ref={canvasRef} />
          </div>
          
          <div className="output-area">
            {outputText}
          </div>
          
          {outputText && <button onClick={clearOutput} className="clear-button">Clear</button>}
        </>
      )}
    </div>
  );
};

export default App;
