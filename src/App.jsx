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

const App = () => {
  const [loading, setLoading] = useState({ loading: true, progress: 0 });
  const [classHistory, setClassHistory] = useState([]);
  const [latestDetection, setLatestDetection] = useState(null);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const webcam = new Webcam();
  const modelName = "ASL";
  const threshold = 0.50;

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

    if (class_detect.length > 0 && class_detect[0] !== 25) { // Check for non-blank detection
      setLatestDetection(class_detect[0]);
    }

    renderBoxes(canvasRef, threshold, boxes, scores, class_detect);
    tf.dispose(res);
    tf.dispose(input);

    requestAnimationFrame(() => detectFrame(model));
    tf.engine().endScope();
  };

  useEffect(() => {
    const interval = setInterval(() => {
      if (latestDetection !== null) {
        setClassHistory(currentHistory => [...currentHistory, latestDetection]);
        console.log(classHistory);
        setLatestDetection(null); // Reset latest detection
      }
    }, 2000); // Update history every 2 seconds

    return () => clearInterval(interval);
  }, [latestDetection]);

  useEffect(() => {
    tf.loadGraphModel(`${window.location.origin}/${modelName}_web_model/model.json`, {
      onProgress: (fractions) => {
        setLoading({ loading: true, progress: fractions });
      },
    }).then(async (yolov7) => {
      const dummyInput = tf.ones(yolov7.inputs[0].shape);
      await yolov7.executeAsync(dummyInput).then((warmupResult) => {
        tf.dispose(warmupResult);
        tf.dispose(dummyInput);

        setLoading({ loading: false, progress: 1 });
        webcam.open(videoRef, () => detectFrame(yolov7));
      });
    });
  }, []);

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
    </div>
  );
};

export default App;
