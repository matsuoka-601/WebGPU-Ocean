import { PrefixSumKernel } from 'webgpu-radix-sort';
import { mat4 } from 'wgpu-matrix'

import { Camera } from './camera'
import { mlsmpmParticleStructSize, MLSMPMSimulator } from './mls-mpm/mls-mpm'
import { SPHSimulator, sphParticleStructSize } from './sph/sph';
import { renderUniformsViews, renderUniformsValues, numParticlesMax } from './common'
import { FluidRenderer } from './render/fluidRender'

/// <reference types="@webgpu/types" />


async function init() {
  const canvas: HTMLCanvasElement = document.querySelector('canvas')!

  if (!navigator.gpu) {
    alert("WebGPU is not supported on your browser.");
    throw new Error()
  }

  const adapter = await navigator.gpu.requestAdapter()

  if (!adapter) {
    alert("Adapter is not available.");
    throw new Error()
  }

  const device = await adapter.requestDevice()

  const context = canvas.getContext('webgpu') as GPUCanvasContext

  if (!context) {
    throw new Error()
  }

  // 共通
  // const { devicePixelRatio } = window
  // let devicePixelRatio  = 3.0;
  let devicePixelRatio  = 0.7;
  canvas.width = devicePixelRatio * canvas.clientWidth
  canvas.height = devicePixelRatio * canvas.clientHeight

  const presentationFormat = navigator.gpu.getPreferredCanvasFormat()

  context.configure({
    device,
    format: presentationFormat,
  })

  return { canvas, device, presentationFormat, context }
}

async function main() {
  const { canvas, device, presentationFormat, context } = await init();

  context.configure({
    device,
    format: presentationFormat,
    alphaMode: 'premultiplied',
  })

  let cubemapTexture: GPUTexture;
  {
    // The order of the array layers is [+X, -X, +Y, -Y, +Z, -Z]
    const imgSrcs = [
      'cubemap/posx.png',
      'cubemap/negx.png',
      'cubemap/posy.png',
      'cubemap/negy.png',
      'cubemap/posz.png',
      'cubemap/negz.png',
    ];
    const promises = imgSrcs.map(async (src) => {
      const response = await fetch(src);
      return createImageBitmap(await response.blob());
    });
    const imageBitmaps = await Promise.all(promises);

    cubemapTexture = device.createTexture({
      dimension: '2d',
      // Create a 2d array texture.
      // Assume each image has the same size.
      size: [imageBitmaps[0].width, imageBitmaps[0].height, 6],
      format: 'rgba8unorm',
      usage:
        GPUTextureUsage.TEXTURE_BINDING |
        GPUTextureUsage.COPY_DST |
        GPUTextureUsage.RENDER_ATTACHMENT,
    });

    for (let i = 0; i < imageBitmaps.length; i++) {
      const imageBitmap = imageBitmaps[i];
      device.queue.copyExternalImageToTexture(
        { source: imageBitmap },
        { texture: cubemapTexture, origin: [0, 0, i] },
        [imageBitmap.width, imageBitmap.height]
      );
    }
  }
  const cubemapTextureView = cubemapTexture.createView({
    dimension: 'cube',
  });

  // uniform buffer を作る
  renderUniformsViews.texel_size.set([1.0 / canvas.width, 1.0 / canvas.height]);

  // storage buffer を作る
  const maxParticleStructSize = Math.max(mlsmpmParticleStructSize, sphParticleStructSize)
  const particleBuffer = device.createBuffer({
    label: 'particles buffer', 
    size: maxParticleStructSize * numParticlesMax, // TODO : max をとる
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  const posvelBuffer = device.createBuffer({
    label: 'position buffer', 
    size: 32 * numParticlesMax,  // 32 = 2 x vec3f + padding
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  const renderUniformBuffer = device.createBuffer({
    label: 'filter uniform buffer', 
    size: renderUniformsValues.byteLength, 
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })


  // レンダリングのパイプライン

  // let distanceParamsIndex = 1; // 20000 
  // const distanceParams = [
  //   { MIN_DISTANCE: 100, MAX_DISTANCE: 100, INIT_DISTANCE: 100 }, // 10000
  //   { MIN_DISTANCE: 60, MAX_DISTANCE: 100, INIT_DISTANCE: 80 }, // 10000
  //   { MIN_DISTANCE: 100, MAX_DISTANCE: 100, INIT_DISTANCE: 100 }, // 30000
  //   { MIN_DISTANCE: 100, MAX_DISTANCE: 100, INIT_DISTANCE: 100 }, // 40000
  //   { MIN_DISTANCE: 100, MAX_DISTANCE: 100, INIT_DISTANCE: 100 }, // 100000
  // ]
  // let currentDistance = distanceParams[distanceParamsIndex].INIT_DISTANCE; 
  // let currentDistance = 30; 

  // let numParticleParams = [40000, 70000, 120000, 200000]
  // let initBoxSizes = [[35, 30, 45], [40, 30, 60], [45, 40, 60], [50, 50, 70]]
  // let initDistances = [60, 70, 80, 100]
  let numParticleParams = [10000, 20000, 30000, 40000]
  let initBoxSizes = [[0.7, 2.0, 0.7], [1.0, 2.0, 1.0], [1.2, 2.0, 1.2], [1.4, 2.0, 1.4]]
  let initDistances = [1.6, 2.1, 2.3, 2.7]

  const canvasElement = document.getElementById("fluidCanvas") as HTMLCanvasElement;
  // const radius = 0.04; // どれくらいがいいかな
  // const diameter = 2 * radius;
  // const initDistance = 2.5
  // let initBoxSize = [1.0, 2.0, 1.0]
  // let realBoxSize = [...initBoxSize];
  // const sphSimulator = new SPHSimulator(particleBuffer, posvelBuffer, initBoxSize, diameter, device)
  // const camera = new Camera(canvasElement, initDistance, [0, -initBoxSize[1], 0], fov);

  let fov = 90 * Math.PI / 180
  let radius = 0.6 // どれくらいがいいかな
  let diameter = 2 * radius
  const mlsmpmSimulator = new MLSMPMSimulator(particleBuffer, posvelBuffer, diameter, device)
  radius = 0.04
  diameter = 2 * radius
  const sphSimulator = new SPHSimulator(particleBuffer, posvelBuffer, diameter, device)
  const camera = new Camera(canvasElement);

  
  const renderer = new FluidRenderer(device, canvas, presentationFormat, 
    radius, fov, posvelBuffer, renderUniformBuffer, cubemapTextureView)

  // ボタン押下の監視
  let form = document.getElementById('number-button') as HTMLFormElement;
  let pressed = false;
  let pressedButton = ""
  form.addEventListener('change', function(event) {
    const target = event.target as HTMLInputElement;
    if (target?.name === 'options') {
      pressed = true;
      pressedButton = target.value;
    }
  });

  // デバイスロストの監視
  let errorLog = document.getElementById('error-reason') as HTMLSpanElement;
  errorLog.textContent = "";
  device.lost.then(info => {
    const reason = info.reason ? `reason: ${info.reason}` : 'unknown reason';
    errorLog.textContent = reason;
  });

  // はじめは mls-mpm
  const initDistance = initDistances[1]
  let initBoxSize = initBoxSizes[1]
  let realBoxSize = [...initBoxSize];
  sphSimulator.reset(numParticleParams[1], initBoxSizes[1])
  camera.reset(canvasElement, initDistance, [0, -initBoxSize[1], 0], fov)

  let ballFl = false;
  let t = 0;
  let boxWidthRatio = 1.
  async function frame() {
    t += 0.01;
    const start = performance.now();

    if (pressed) { 
      const paramsIdx = parseInt(pressedButton);
      initBoxSize = initBoxSizes[paramsIdx]
      realBoxSize = [...initBoxSize]
      // mlsmpmSimulator.reset(numParticleParams[paramsIdx], initBoxSize)
      // camera.reset(canvasElement, initDistances[paramsIdx], [initBoxSize[0] / 2, initBoxSize[1] / 4, initBoxSize[2] / 2], fov)
      sphSimulator.reset(numParticleParams[paramsIdx], initBoxSize)
      camera.reset(canvasElement, initDistances[paramsIdx], [0, -initBoxSize[1], 0], fov)
      let slider = document.getElementById("slider") as HTMLInputElement;
      slider.value = "100";
      pressed = false;
    } 

    // ボックスサイズの変更
    const slider = document.getElementById("slider") as HTMLInputElement;
    const sliderValue = document.getElementById("slider-value") as HTMLSpanElement;
    const particle = document.getElementById("particle") as HTMLInputElement;
    let curBoxWidthRatio = parseInt(slider.value) / 200 + 0.5;
    const minClosingSpeed = -0.007;
    const dVal = Math.max(curBoxWidthRatio - boxWidthRatio, minClosingSpeed);
    boxWidthRatio += dVal;

    // 行列の更新
    // realBoxSize[2] = initBoxSize[2] * boxWidthRatio;
    // mlsmpmSimulator.changeBoxSize(realBoxSize)
    device.queue.writeBuffer(renderUniformBuffer, 0, renderUniformsValues); // これもなくしたい

    const commandEncoder = device.createCommandEncoder()

    // 計算のためのパス
    sphSimulator.execute(commandEncoder)

    // レンダリングのためのパス
    if (!ballFl) {
      renderer.execute(context, commandEncoder, sphSimulator.numParticles)
    } else {
      // const ballPassEncoder = commandEncoder.beginRenderPass(ballPassDescriptor);
      // ballPassEncoder.setBindGroup(0, ballBindGroup);
      // ballPassEncoder.setPipeline(ballPipeline);
      // ballPassEncoder.draw(6, mlsmpmSimulator.numParticles);
      // ballPassEncoder.end();
    }

    device.queue.submit([commandEncoder.finish()])
    const end = performance.now();
    // console.log(`js: ${(end - start).toFixed(1)}ms`);

    requestAnimationFrame(frame)
  } 
  requestAnimationFrame(frame)
}

main()