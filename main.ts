import shader from './render/shader.wgsl'
import show from './render/show.wgsl'
import filter from './render/bilateral.wgsl'
import fluid from './render/fluid.wgsl'
import vertex from './render/vertex.wgsl'
import thickness from './render/thickness.wgsl'
import gaussian from './render/gaussian.wgsl'
import ball from './render/ball.wgsl'


import { PrefixSumKernel } from 'webgpu-radix-sort';
import { mat4 } from 'wgpu-matrix'

import { Camera } from './camera'
import { MLSMPMSimulator } from './mls-mpm/mls-mpm'
import { renderUniformsViews, renderUniformsValues, numParticlesMax, particleStructSize } from './common'
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
  let devicePixelRatio  = 1.0;
  canvas.width = devicePixelRatio * canvas.clientWidth
  canvas.height = devicePixelRatio * canvas.clientHeight

  const presentationFormat = navigator.gpu.getPreferredCanvasFormat()

  context.configure({
    device,
    format: presentationFormat,
  })

  return { canvas, device, presentationFormat, context }
}

const radius = 0.6; // どれくらいがいいかな
const diameter = 2 * radius;

async function main() {
  const { canvas, device, presentationFormat, context } = await init();

  context.configure({
    device,
    format: presentationFormat,
    alphaMode: 'premultiplied',
  })

  const vertexModule = device.createShaderModule({ code: vertex })
  const shaderModule = device.createShaderModule({ code: shader })
  const showModule = device.createShaderModule({ code: show })
  const filterModule = device.createShaderModule({ code: filter })
  const fluidModule = device.createShaderModule({ code: fluid })
  const ballModule = device.createShaderModule({ code: ball })
  const thicknessModule = device.createShaderModule({ code: thickness })
  const thicknessFilterModule = device.createShaderModule({ code: gaussian })

  // レンダリングパイプライン
  const circlePipeline = device.createRenderPipeline({
    label: 'circles pipeline', 
    layout: 'auto', 
    vertex: { module: shaderModule },
    fragment: {
      module: shaderModule,
      targets: [
        {
          format: 'r32float',
        },
      ],
    },
    primitive: {
      topology: 'triangle-list', // ここだけプリミティブに応じて変える
      // cullMode: 'back', 
    },
    depthStencil: {
      depthWriteEnabled: true, // enable depth test
      depthCompare: 'less',
      format: 'depth32float'
    }
  })
  const ballPipeline = device.createRenderPipeline({
    label: 'ball pipeline', 
    layout: 'auto', 
    vertex: { module: ballModule }, 
    fragment: {
      module: ballModule, 
      targets: [
        {
          format: presentationFormat, 
        }
      ]
    }, 
    primitive: {
      topology: 'triangle-list', // ここだけプリミティブに応じて変える
      // cullMode: 'back', 
    },
    depthStencil: {
      depthWriteEnabled: true, // enable depth test
      depthCompare: 'less',
      format: 'depth32float'
    }
  })

  const fov = 60 * Math.PI / 180;

  const screenConstants = {
    'screenHeight': canvas.height, 
    'screenWidth': canvas.width, 
  }
  // TODO : filter size を設定できるようにする
  const filterConstants = {
    'depth_threshold' : radius * 10, 
    'max_filter_size' : 100, 
    'projected_particle_constant' : (10 * diameter * 0.05 * (canvas.height / 2)) / Math.tan(fov / 2), 
  }
  const filterPipeline = device.createRenderPipeline({
    label: 'filter pipeline', 
    layout: 'auto', 
    vertex: { 
      module: vertexModule,  
      constants: screenConstants
    },
    fragment: {
      module: filterModule, 
      constants: filterConstants, 
      targets: [
        {
          format: 'r32float',
        },
      ],
    },
    primitive: {
      topology: 'triangle-list', // ここだけプリミティブに応じて変える
      // cullMode: 'back', 
    },
  });
  const fluidPipeline = device.createRenderPipeline({
    label: 'fluid rendering pipeline', 
    layout: 'auto', 
    vertex: { 
      module: vertexModule,  
      constants: screenConstants
    }, 
    fragment: {
      module: fluidModule, 
      targets: [
        {
          format: presentationFormat
        }
      ],
    }, 
    primitive: {
      topology: 'triangle-list', // ここだけプリミティブに応じて変える
      cullMode: 'none'
    },
  });
  const thicknessPipeline = device.createRenderPipeline({
    label: 'thickness pipeline', 
    layout: 'auto', 
    vertex: { 
      module: thicknessModule,  
    }, 
    fragment: {
      module: thicknessModule, 
      targets: [
        {
          format: 'r16float',
          writeMask: GPUColorWrite.RED,
          blend: {
            color: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
            alpha: { operation: 'add', srcFactor: 'one', dstFactor: 'one' },
          }
        }
      ],
    }, 
    primitive: {
      topology: 'triangle-list', // ここだけプリミティブに応じて変える
      cullMode: 'none'
    },
  });
  const thicknessFilterPipeline = device.createRenderPipeline({
    label: 'thickness filter pipeline', 
    layout: 'auto', 
    vertex: { 
      module: vertexModule,  
      constants: screenConstants
    },
    fragment: {
      module: thicknessFilterModule,
      targets: [
        {
          format: 'r16float',
        },
      ],
    },
    primitive: {
      topology: 'triangle-list', // ここだけプリミティブに応じて変える
      // cullMode: 'back', 
    },
  });
  const showPipeline = device.createRenderPipeline({
    label: 'show pipeline', 
    layout: 'auto', 
    vertex: { module: vertexModule, constants: screenConstants }, 
    fragment: {
      module: showModule, 
      targets: [
        {
          format: presentationFormat
        }
      ]
    }, 
    primitive: {
      topology: 'triangle-list', // ここだけプリミティブに応じて変える
      cullMode: 'none'
    },
  })
  // 計算のためのパイプライン

  // テクスチャ・サンプラの作成
  const renderTargetTexture = device.createTexture({
    label: 'depth map texture', 
    size: [canvas.width, canvas.height, 1],
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    format: 'r32float',
  });
  const renderTargetTextureView = renderTargetTexture.createView();
  const tmpTargetTexture = device.createTexture({
    label: 'temporary texture', 
    size: [canvas.width, canvas.height, 1],
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    format: 'r32float',
  });
  const tmpTargetTextureView = tmpTargetTexture.createView();
  const thicknessTexture = device.createTexture({
    label: 'thickness map texture', 
    size: [canvas.width, canvas.height, 1],
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    format: 'r16float',
  });
  const thicknessTextureView = thicknessTexture.createView();
  const tmpThicknessTexture = device.createTexture({
    label: 'temporary thickness map texture', 
    size: [canvas.width, canvas.height, 1],
    usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING,
    format: 'r16float',
  });
  const tmpThicknessTextureView = tmpThicknessTexture.createView();
  const depthTexture = device.createTexture({
    size: [canvas.width, canvas.height, 1],
    format: 'depth32float',
    usage: GPUTextureUsage.RENDER_ATTACHMENT,
  })
  const depthTextureView = depthTexture.createView()

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

    console.log(imageBitmaps[0].width, imageBitmaps[0].height);

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

  const sampler = device.createSampler({
    magFilter: 'linear', 
    minFilter: 'linear'
  });


  // uniform buffer を作る
  renderUniformsViews.texel_size.set([1.0 / canvas.width, 1.0 / canvas.height]);


  const filterXUniformsValues = new ArrayBuffer(8);
  const filterYUniformsValues = new ArrayBuffer(8);
  const filterXUniformsViews = { blur_dir: new Float32Array(filterXUniformsValues) };
  const filterYUniformsViews = { blur_dir: new Float32Array(filterYUniformsValues) };
  filterXUniformsViews.blur_dir.set([1.0, 0.0]);
  filterYUniformsViews.blur_dir.set([0.0, 1.0]);

  // storage buffer を作る
  const particleBuffer = device.createBuffer({
    label: 'particles buffer', 
    size: particleStructSize * numParticlesMax, 
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })


  const filterXUniformBuffer = device.createBuffer({
    label: 'filter uniform buffer', 
    size: filterXUniformsValues.byteLength, 
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const filterYUniformBuffer = device.createBuffer({
    label: 'filter uniform buffer', 
    size: filterYUniformsValues.byteLength, 
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const renderUniformBuffer = device.createBuffer({
    label: 'filter uniform buffer', 
    size: renderUniformsValues.byteLength, 
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })

  device.queue.writeBuffer(filterXUniformBuffer, 0, filterXUniformsValues);
  device.queue.writeBuffer(filterYUniformBuffer, 0, filterYUniformsValues);


  // レンダリングのパイプライン

  let distanceParamsIndex = 1; // 20000 
  const distanceParams = [
    { MIN_DISTANCE: 100, MAX_DISTANCE: 100, INIT_DISTANCE: 100 }, // 10000
    { MIN_DISTANCE: 60, MAX_DISTANCE: 100, INIT_DISTANCE: 80 }, // 10000
    { MIN_DISTANCE: 100, MAX_DISTANCE: 100, INIT_DISTANCE: 100 }, // 30000
    { MIN_DISTANCE: 100, MAX_DISTANCE: 100, INIT_DISTANCE: 100 }, // 40000
    { MIN_DISTANCE: 100, MAX_DISTANCE: 100, INIT_DISTANCE: 100 }, // 100000
  ]
  let currentDistance = distanceParams[distanceParamsIndex].INIT_DISTANCE; 
  // let currentDistance = 30; 

  const canvasElement = document.getElementById("fluidCanvas") as HTMLCanvasElement;
  const initDistance = 80
  let initBoxSize = [40, 40, 80];
  let realBoxSize = [...initBoxSize];
  const camera = new Camera(canvasElement, initDistance, [initBoxSize[0] / 2, initBoxSize[1] / 4, initBoxSize[2] / 2], fov);
  const mlsmpmSimulator = new MLSMPMSimulator(particleBuffer, initBoxSize, diameter, device)
  const renderer = new FluidRenderer(device, canvas, presentationFormat, 
    radius, fov, particleBuffer, renderUniformBuffer, cubemapTextureView)

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

  // // let boxSizeKey = ["10000", "20000", "30000", "40000", "100000"]
  // let boxSizes = [
  //   { xHalf: 0.7, yHalf: 2.0, zHalf: 0.7 }, 
  //   { xHalf: 1.0, yHalf: 2.0, zHalf: 1.0 }, 
  //   { xHalf: 1.2, yHalf: 2.0, zHalf: 1.2 }, 
  //   { xHalf: 1.4, yHalf: 2.0, zHalf: 1.4 }, 
  //   { xHalf: 1.0, yHalf: 2.0, zHalf: 2.0 }
  // ];

  
  // let environment = {
  //   boxSize: boxSizes[1], 
  //   numParticles: 20000, 
  // } 

  // デバイスロストの監視
  let errorLog = document.getElementById('error-reason') as HTMLSpanElement;
  errorLog.textContent = "";
  device.lost.then(info => {
    const reason = info.reason ? `reason: ${info.reason}` : 'unknown reason';
    errorLog.textContent = reason;
  });


  let ballFl = false;
  let t = 0;
  let boxWidthRatio = 1.
  async function frame() {
    t += 0.01;
    const start = performance.now();

    // if (pressed) { 
    //   distanceParamsIndex = boxSizeKey.indexOf(pressedButton);
    //   environment.boxSize = boxSizes[distanceParamsIndex];
    //   environment.numParticles = parseInt(pressedButton);
    //   currentXtheta = Math.PI / 4;
    //   currentYtheta = -Math.PI / 12;
    //   const particlesData = init_dambreak(constants.grid_res);
    //   device.queue.writeBuffer(particlesBuffer, 0, particlesData);
    //   currentDistance = distanceParams[distanceParamsIndex].INIT_DISTANCE;
    //   let slider = document.getElementById("slider") as HTMLInputElement;
    //   slider.value = "100";

    //   console.log(distanceParams[distanceParamsIndex]);
      
    //   pressed = false;
    // }



    // ボックスサイズの変更
    const slider = document.getElementById("slider") as HTMLInputElement;
    const sliderValue = document.getElementById("slider-value") as HTMLSpanElement;
    const particle = document.getElementById("particle") as HTMLInputElement;
    let curBoxWidthRatio = parseInt(slider.value) / 200 + 0.5;
    const minClosingSpeed = -0.007;
    const dVal = Math.max(curBoxWidthRatio - boxWidthRatio, minClosingSpeed);
    boxWidthRatio += dVal;
    // sliderValue.textContent = curBoxWidthRatio.toFixed(2);
    // realBoxSizeViews.xHalf.set([environment.boxSize.xHalf]);
    // realBoxSizeViews.yHalf.set([environment.boxSize.yHalf]);
    // realBoxSizeViews.zHalf.set([environment.boxSize.zHalf * boxWidthRatio]);
    // device.queue.writeBuffer(realBoxSizeBuffer, 0, realBoxSizeValues);

    // 行列の更新
    realBoxSize[2] = initBoxSize[2] * boxWidthRatio;
    mlsmpmSimulator.changeBoxSize(realBoxSize)
    device.queue.writeBuffer(renderUniformBuffer, 0, renderUniformsValues); // これもなくしたい

    const commandEncoder = device.createCommandEncoder()

    // 計算のためのパス
    mlsmpmSimulator.execute(commandEncoder)

    // レンダリングのためのパス
    if (!ballFl) {
      renderer.execute(context, commandEncoder, mlsmpmSimulator.numParticles)
    } else {
      // const ballPassEncoder = commandEncoder.beginRenderPass(ballPassDescriptor);
      // ballPassEncoder.setBindGroup(0, ballBindGroup);
      // ballPassEncoder.setPipeline(ballPipeline);
      // ballPassEncoder.draw(6, mlsmpmSimulator.numParticles);
      // ballPassEncoder.end();
    }

    // const showPassEncoder = commandEncoder.beginRenderPass(showPassDescriptor);
    // showPassEncoder.setBindGroup(0, showBindGroup);
    // showPassEncoder.setPipeline(showPipeline);
    // showPassEncoder.draw(6);
    // showPassEncoder.end();

    device.queue.submit([commandEncoder.finish()])
    const end = performance.now();
    // console.log(`js: ${(end - start).toFixed(1)}ms`);

    requestAnimationFrame(frame)
  } 
  requestAnimationFrame(frame)
}

main()