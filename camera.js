import * as posenet from '@tensorflow-models/posenet';
import Stats from 'stats.js';

import { drawKeypoints, drawSkeleton } from './demo_util';

const videoWidth = 600;
const videoHeight = 500;
const stats = new Stats();

function isAndroid() {
  return /Android/i.test(navigator.userAgent);
}

function isiOS() {
  return /iPhone|iPad|iPod/i.test(navigator.userAgent);
}

function isMobile() {
  return isAndroid() || isiOS();
}

async function setupCamera() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    throw new Error(
      'Browser API navigator.mediaDevices.getUserMedia not available');
  }

  const video = document.getElementById('video');
  video.width = videoWidth;
  video.height = videoHeight;

  const mobile = isMobile();
  const stream = await navigator.mediaDevices.getUserMedia({
    'audio': false,
    'video': {
      facingMode: 'user',
      width: mobile ? undefined : videoWidth,
      height: mobile ? undefined : videoHeight,
    },
  });
  video.srcObject = stream;

  return new Promise((resolve) => {
    video.onloadedmetadata = () => {
      resolve(video);
    };
  });
}

async function loadVideo() {
  const video = await setupCamera();
  video.play();

  return video;
}

const guiState = {
  algorithm: 'multi-pose',
  input: {
    mobileNetArchitecture: isMobile() ? '0.50' : '0.75',
    outputStride: 16,
    imageScaleFactor: 0.5,
  },
  singlePoseDetection: {
    minPoseConfidence: 0.1,
    minPartConfidence: 0.5,
  },
  multiPoseDetection: {
    maxPoseDetections: 5,
    minPoseConfidence: 0.15,
    minPartConfidence: 0.1,
    nmsRadius: 30.0,
  },
  output: {
    showVideo: true,
    showSkeleton: true,
    showPoints: true,
    showBoundingBox: false,
  },
  net: null,
};

function detectPoseInRealTime(video, net) {
  const canvas = document.getElementById('output');
  const ctx = canvas.getContext('2d');

  const flipHorizontal = true;

  canvas.width = videoWidth;
  canvas.height = videoHeight;

  async function poseDetectionFrame() {

    // Begin monitoring code for frames per second
    stats.begin();

    // Scale an image down to a certain factor. Too large of an image will slow
    // down the GPU
    const imageScaleFactor = guiState.input.imageScaleFactor;
    const outputStride = +guiState.input.outputStride;

    let poses = [];
    let minPoseConfidence;
    let minPartConfidence;

    const pose = await net.estimateSinglePose(
      video, imageScaleFactor, flipHorizontal, outputStride);
    poses.push(pose);

    ctx.clearRect(0, 0, videoWidth, videoHeight);

    ctx.save();
    ctx.scale(-1, 1);
    ctx.translate(-videoWidth, 0);
    ctx.drawImage(video, 0, 0, videoWidth, videoHeight);
    ctx.restore();

    poses.forEach(({ keypoints }) => {
      drawKeypoints(keypoints, minPartConfidence, ctx);
      drawSkeleton(keypoints, minPartConfidence, ctx);
    });

    stats.end();

    requestAnimationFrame(poseDetectionFrame);
  }

  poseDetectionFrame();
}

export async function bindPage() {
  const net = await posenet.load(0.75);

  document.getElementById('loading').style.display = 'none';
  document.getElementById('main').style.display = 'block';

  let video;

  try {
    video = await loadVideo();
  } catch (e) {
    let info = document.getElementById('info');
    info.textContent = 'this browser does not support video capture,' +
      'or this device does not have a camera';
    info.style.display = 'block';
    throw e;
  }
  detectPoseInRealTime(video, net);
}

navigator.getUserMedia = navigator.getUserMedia ||
  navigator.webkitGetUserMedia || navigator.mozGetUserMedia;
bindPage();
