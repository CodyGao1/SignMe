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
  const [outputText, setOutputText] = useState('');
  const [intervalDuration, setIntervalDuration] = useState(2000); // New state for interval duration
  const latestDetectionRef = useRef(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const webcam = new Webcam();
  const modelName = "ASL";
  const threshold = 0.85;

  const detectFrame = async (model) => {
    // detection logic...
  };

  useEffect(() => {
    // model loading logic...
  }, []);

  useEffect(() => {
    const interval = setInterval(() => {
      if (latestDetectionRef.current !== null) {
        const newLetter = mapIdToLetter(latestDetectionRef.current);
        setOutputText(currentText => currentText + newLetter);
        latestDetectionRef.current = null;
      }
    }, intervalDuration);
    return () => clearInterval(interval);
  }, [intervalDuration]);

  const clearOutput = () => {
    setOutputText('');
  };

  return (
    <div className="App">
      <h2>SignMe</h2>
      {loading ? (
        <div>
          <Loader />
          <p>Loading model...</p>
        </div>
      ) : (
        <>
          <div className="content">
            <video autoPlay playsInline muted ref={videoRef} id="frame" />
            <canvas width={512} height={512} ref={canvasRef} />
          </div>
          
          <div className="output-area">
            {outputText}
          </div>
          
          <div className="slider-container">
            <input
              type="range"
              min="500"
              max="5000"
              value={intervalDuration}
              onChange={(e) => setIntervalDuration(Number(e.target.value))}
              className="slider"
            />
            <p>Interval: {intervalDuration} ms</p>
          </div>

          {outputText && <button onClick={clearOutput} className="clear-button">Clear</button>}
        </>
      )}
    </div>
  );
};

export default App;
