// shader の import には ?raw が必要
import density from './compute/density.wgsl'
import force from './compute/force.wgsl'
import integrate from './compute/integrate.wgsl'

import shader from './render/shader.wgsl'
import show from './render/show.wgsl'
import filter from './render/bilateral.wgsl'
import fluid from './render/fluid.wgsl'
import vertex from './render/vertex.wgsl'
import thickness from './render/thickness.wgsl'
import gaussian from './render/gaussian.wgsl'
import ball from './render/ball.wgsl'

import gridClear from './compute/grid/gridClear.wgsl'
import gridBuild from './compute/grid/gridBuild.wgsl'
import reorderParticles from './compute/grid/reorderParticles.wgsl'

import { PrefixSumKernel } from 'webgpu-radix-sort';
import { mat4 } from 'wgpu-matrix'

/// <reference types="@webgpu/types" />

const kernelRadius = 0.07;
const xHalf = 0.7;
const yHalf = 2.0;
const zHalf = 0.7;
function init_dambreak(n: number) {
  let particles = new ArrayBuffer(64 * n);
  var cnt = 0;
  const DIST_FACTOR = 0.4


  for (var y = -yHalf; cnt < n; y += DIST_FACTOR * kernelRadius) {
      for (var x = -0.95 * xHalf; x < 0.95 * xHalf && cnt < n; x += DIST_FACTOR * kernelRadius) {
          for (var z = -0.95 * zHalf; z < 0. && cnt < n; z += DIST_FACTOR * kernelRadius) {
              let jitter = 0.0001 * Math.random();
              const offset = 64 * cnt;
              const particleViews = {
                position: new Float32Array(particles, offset, 3),
                velocity: new Float32Array(particles, offset + 16, 3),
                force: new Float32Array(particles, offset + 32, 3),
                density: new Float32Array(particles, offset + 44, 1),
                nearDensity: new Float32Array(particles, offset + 48, 1),
              };
              particleViews.position.set([x + jitter, y, z]);
              cnt++;
          }
      }
  }
  return particles;
}


async function init() {
  const canvas: HTMLCanvasElement = document.querySelector('canvas')!

  const adapter = await navigator.gpu.requestAdapter()

  if (!adapter) {
    throw new Error()
  }

  const device = await adapter.requestDevice()

  const context = canvas.getContext('webgpu') as GPUCanvasContext

  if (!context) {
    throw new Error()
  }

  // 共通
  const { devicePixelRatio } = window
  // let devicePixelRatio  = 3.0;
  canvas.width = devicePixelRatio * canvas.clientWidth
  canvas.height = devicePixelRatio * canvas.clientHeight

  const presentationFormat = navigator.gpu.getPreferredCanvasFormat()

  context.configure({
    device,
    format: presentationFormat,
  })

  return { canvas, device, presentationFormat, context }
}

function computeViewPosFromUVDepth(
  uv: number[], 
  depth: number,
  projection: Float32Array, 
  inv_projection: Float32Array, 
) {
  var ndc = [uv[0] * 2 - 1, 1 - uv[1] * 2, 0, 1];
  ndc[2] = -projection[4 * 2 + 2] + projection[4 * 3 + 2] / depth;
  var eye_pos = mat4.multiply(inv_projection, ndc);
  var w = eye_pos[3];
  return [eye_pos[0] / w, eye_pos[1] / w, eye_pos[2] / w];
}

function init_camera(canvas: HTMLCanvasElement, fov: number) {
  const aspect = canvas.clientWidth / canvas.clientHeight
  const projection = mat4.perspective(fov, aspect, 0.1, 50)
  const view = mat4.lookAt(
    [0, 0, 5.5], // position
    [0, 0, 0], // target
    [0, 1, 0], // up
  )
  // mat4.rotateY(view, 1.0, view); // 初期値
  const v = new Float32Array(4);
  v[0] = 1.5;
  v[1] = 1.0;
  v[2] = 2.;
  v[3] = 1.0;
  const camera = mat4.multiply(view, v);
  const depth = Math.abs(camera[2]);
  const clip = mat4.multiply(projection, camera);
  const ndc = [clip[0] / clip[3], clip[1] / clip[3], clip[2] / clip[3]];
  console.log("camera: ", camera); // カメラ座標は，[1, 1, -0.2] である．
  console.log("ndc: ", ndc);
  // console.log(-projection[4 * 2 + 2] + projection[4 * 3 + 2] / depth);

  // uv と depth から，カメラ座標（[0, 1, -0.2]）を復元できるか？
  const uv = [(ndc[0] + 1) / 2., (1 - ndc[1]) / 2]
  const inv_projection = mat4.inverse(projection);
  const constructed_camera = computeViewPosFromUVDepth(uv, depth, projection, inv_projection);
  console.log(constructed_camera);

  return { projection, view } 
}

function recalculateView(r: number, xRotate: number, yRotate: number, target: number[]) {
  var mat = mat4.identity();
  mat4.translate(mat, target, mat);
  mat4.rotateY(mat, yRotate, mat);
  mat4.rotateX(mat, xRotate, mat);
  mat4.translate(mat, [0, 0, r], mat);
  var position = mat4.multiply(mat, [0, 0, 0, 1]);

  const view = mat4.lookAt(
    [position[0], position[1], position[2]], // position
    target, // target
    [0, 1, 0], // up
  )
  return view;
}

const radius = 0.025;
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
  const densityModule = device.createShaderModule({ code: density })
  const forceModule = device.createShaderModule({ code: force })
  const integrateModule = device.createShaderModule({ code: integrate })
  const gridBuildModule = device.createShaderModule({ code: gridBuild })
  const gridClearModule = device.createShaderModule({ code: gridClear })
  const reorderParticlesModule = device.createShaderModule({ code: reorderParticles })

  const constants = {
    kernelRadius: kernelRadius, 
    kernelRadiusPow2: Math.pow(kernelRadius, 2), 
    kernelRadiusPow4: Math.pow(kernelRadius, 4), 
    kernelRadiusPow5: Math.pow(kernelRadius, 5), 
    kernelRadiusPow6: Math.pow(kernelRadius, 6), 
    kernelRadiusPow9: Math.pow(kernelRadius, 9), 
    stiffness: 15.0, 
    nearStiffness : 1.,   
    mass: 1.0, 
    restDensity: 40000, 
    viscosity: 300000, 
    dt: 0.008, 
    xHalf: xHalf, 
    yHalf: yHalf, 
    zHalf: zHalf, 
  }
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

  const fov = 90 * Math.PI / 180;
  const { projection, view } = init_camera(canvas, fov);



  const screenConstants = {
    'screenHeight': canvas.height, 
    'screenWidth': canvas.width, 
  }
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
  const cellSize = 1.0 * kernelRadius; 
  const xLen = 2.0 * xHalf;
  const yLen = 2.0 * yHalf;
  const zLen = 2.0 * zHalf;
  const sentinel = 4 * cellSize;
  const xGrids = Math.ceil((xLen + sentinel) / cellSize);
  const yGrids = Math.ceil((yLen + sentinel) / cellSize);
  const zGrids = Math.ceil((zLen + sentinel) / cellSize);
  const gridCount = xGrids * yGrids * zGrids;
  const offset = sentinel / 2;
  const sizeConstants = {
    'xHalf' : xHalf, 
    'yHalf' : yHalf, 
    'zHalf' : zHalf, 
    'gridCount': gridCount, 
    'xGrids': xGrids, 
    'yGrids': yGrids, 
    'cellSize' : cellSize, 
    'offset' : offset
  }
  const gridClearPipeline = device.createComputePipeline({
    label: "grid clear pipeline", 
    layout: 'auto', 
    compute: {
      module: gridClearModule, 
    }
  })
  const gridBuildPipeline = device.createComputePipeline({
    label: "grid build pipeline", 
    layout: 'auto', 
    compute: {
      module: gridBuildModule, 
      constants: sizeConstants, 
    }
  })
  const reorderPipeline = device.createComputePipeline({
    label: "reorder pipeline", 
    layout: 'auto', 
    compute: {
      module: reorderParticlesModule, 
      constants: sizeConstants, 
    }
  })
  const densityPipeline = device.createComputePipeline({
    label: "density pipeline", 
    layout: 'auto', 
    compute: {
      module: densityModule, 
      constants: {
        'kernelRadius': constants.kernelRadius, 
        'kernelRadiusPow2': constants.kernelRadiusPow2, 
        'kernelRadiusPow5': constants.kernelRadiusPow5, 
        'kernelRadiusPow6': constants.kernelRadiusPow6, 
        'mass': constants.mass, 
        'xHalf' : xHalf, 
        'yHalf' : yHalf, 
        'zHalf' : zHalf, 
        'xGrids': xGrids, 
        'yGrids': yGrids, 
        'zGrids': zGrids, 
        'cellSize' : cellSize, 
        'offset' : offset
      }, 
    }
  });
  const forcePipeline = device.createComputePipeline({
    label: "force pipeline", 
    layout: 'auto', 
    compute: {
      module: forceModule, 
      constants: {
        'kernelRadius': constants.kernelRadius, 
        'kernelRadiusPow2': constants.kernelRadiusPow2, 
        'kernelRadiusPow5': constants.kernelRadiusPow5, 
        'kernelRadiusPow6': constants.kernelRadiusPow6, 
        'kernelRadiusPow9': constants.kernelRadiusPow9, 
        'mass': constants.mass, 
        'stiffness': constants.stiffness, 
        'nearStiffness': constants.nearStiffness, 
        'viscosity': constants.viscosity, 
        'restDensity': constants.restDensity, 
        'xHalf' : xHalf, 
        'yHalf' : yHalf, 
        'zHalf' : zHalf, 
        'xGrids': xGrids, 
        'yGrids': yGrids, 
        'zGrids': zGrids, 
        'cellSize' : cellSize, 
        'offset' : offset
      }, 
    }
  });
  const integratePipeline = device.createComputePipeline({
    label: "integrate pipeline", 
    layout: 'auto', 
    compute: {
      module: integrateModule, 
      constants: {
        'dt': constants.dt, 
      }, 
    }
  });

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
      'cubemap/posx.jpg',
      'cubemap/negx.jpg',
      'cubemap/posy.jpg',
      'cubemap/negy.jpg',
      'cubemap/posz.jpg',
      'cubemap/negz.jpg',
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


  const numParticles = 20000
  const particlesData = init_dambreak(numParticles)


  // uniform buffer を作る
  const uniformsValues = new ArrayBuffer(144);
  const uniformsViews = {
    size: new Float32Array(uniformsValues, 0, 1),
    view_matrix: new Float32Array(uniformsValues, 16, 16),
    projection_matrix: new Float32Array(uniformsValues, 80, 16),
  };

  const filterXUniformsValues = new ArrayBuffer(8);
  const filterYUniformsValues = new ArrayBuffer(8);
  const filterXUniformsViews = { blur_dir: new Float32Array(filterXUniformsValues) };
  const filterYUniformsViews = { blur_dir: new Float32Array(filterYUniformsValues) };
  filterXUniformsViews.blur_dir.set([1.0, 0.0]);
  filterYUniformsViews.blur_dir.set([0.0, 1.0]);

  const fluidUniformsValues = new ArrayBuffer(272);
  const fluidUniformsViews = {
    texel_size: new Float32Array(fluidUniformsValues, 0, 2),
    inv_projection_matrix: new Float32Array(fluidUniformsValues, 16, 16),
    projection_matrix: new Float32Array(fluidUniformsValues, 80, 16),
    view_matrix: new Float32Array(fluidUniformsValues, 144, 16), 
    inv_view_matrix: new Float32Array(fluidUniformsValues, 208, 16),
  };  
  fluidUniformsViews.texel_size.set([1.0 / canvas.width, 1.0 / canvas.height]);
  fluidUniformsViews.projection_matrix.set(projection);
  const inv_projection = mat4.identity();
  const inv_view = mat4.identity();
  mat4.inverse(projection, inv_projection);
  mat4.inverse(view, inv_view);
  fluidUniformsViews.inv_projection_matrix.set(inv_projection);
  fluidUniformsViews.inv_view_matrix.set(inv_view);

  const realBoxSizeValues = new ArrayBuffer(12);
  const realBoxSizeViews = {
    xHalf: new Float32Array(realBoxSizeValues, 0, 1),
    yHalf: new Float32Array(realBoxSizeValues, 4, 1),
    zHalf: new Float32Array(realBoxSizeValues, 8, 1),
  };
  realBoxSizeViews.xHalf.set([xHalf]);
  realBoxSizeViews.yHalf.set([yHalf]);
  realBoxSizeViews.zHalf.set([zHalf]);

  // storage buffer を作る
  const particlesBuffer = device.createBuffer({
    label: 'particles buffer', 
    size: particlesData.byteLength, 
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  const cellParticleCountBuffer = device.createBuffer({ // 累積和はここに保存
    label: 'cell particle count buffer', 
    size: 4 * (gridCount + 1),  // 1 要素余分にとっておく
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  const targetParticlesBuffer = device.createBuffer({
    label: 'target particles buffer', 
    size: particlesData.byteLength, 
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  const particleCellOffsetBuffer = device.createBuffer({
    label: 'particle cell offset buffer', 
    size: 4 * numParticles,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(particlesBuffer, 0, particlesData)


  const uniformBuffer = device.createBuffer({
    label: 'uniform buffer', 
    size: uniformsValues.byteLength,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
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
  const fluidUniformBuffer = device.createBuffer({
    label: 'filter uniform buffer', 
    size: fluidUniformsValues.byteLength, 
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  const realBoxSizeBuffer = device.createBuffer({
    label: 'real box size buffer', 
    size: realBoxSizeValues.byteLength, 
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(filterXUniformBuffer, 0, filterXUniformsValues);
  device.queue.writeBuffer(filterYUniformBuffer, 0, filterYUniformsValues);
  device.queue.writeBuffer(fluidUniformBuffer, 0, fluidUniformsValues);
  device.queue.writeBuffer(realBoxSizeBuffer, 0, realBoxSizeValues);


  // 計算の bindGroup
  const gridClearBindGroup = device.createBindGroup({
      layout: gridClearPipeline.getBindGroupLayout(0), 
      entries: [
        { binding: 0, resource: { buffer: cellParticleCountBuffer }}, // 累積和をクリア
      ],  
  })
  const gridBuildBindGroup = device.createBindGroup({
    layout: gridBuildPipeline.getBindGroupLayout(0), 
    entries: [
      { binding: 0, resource: { buffer: cellParticleCountBuffer }}, 
      { binding: 1, resource: { buffer: particleCellOffsetBuffer }}, 
      { binding: 2, resource: { buffer: particlesBuffer }}, 
    ],  
  })
  const reorderBindGroup = device.createBindGroup({
    layout: reorderPipeline.getBindGroupLayout(0), 
    entries: [
      { binding: 0, resource: { buffer: particlesBuffer }}, 
      { binding: 1, resource: { buffer: targetParticlesBuffer }}, 
      { binding: 2, resource: { buffer: cellParticleCountBuffer }}, 
      { binding: 3, resource: { buffer: particleCellOffsetBuffer }}, 
    ]
  })
  const densityBindGroup = device.createBindGroup({
    layout: densityPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particlesBuffer }},
      { binding: 1, resource: { buffer: targetParticlesBuffer }},
      { binding: 2, resource: { buffer: cellParticleCountBuffer }},
    ],
  })
  const forceBindGroup = device.createBindGroup({
    layout: forcePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particlesBuffer }},
      { binding: 1, resource: { buffer: targetParticlesBuffer }},
      { binding: 2, resource: { buffer: cellParticleCountBuffer }},
    ],
  })
  const integrateBindGroup = device.createBindGroup({
    layout: integratePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particlesBuffer }},
      { binding: 1, resource: { buffer: realBoxSizeBuffer }},
    ],
  })

  // レンダリングのパイプライン
  const ballBindGroup = device.createBindGroup({
    label: 'ball bind group', 
    layout: ballPipeline.getBindGroupLayout(0),  
    entries: [
      { binding: 0, resource: { buffer: particlesBuffer }},
      { binding: 1, resource: { buffer: uniformBuffer }},
    ]
  })
  const circleBindGroup = device.createBindGroup({
    label: 'circle bind group', 
    layout: circlePipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particlesBuffer }},
      { binding: 1, resource: { buffer: uniformBuffer }},
    ],
  })
  const showBindGroup = device.createBindGroup({
    label: 'show bind group', 
    layout: showPipeline.getBindGroupLayout(0),
    entries: [
      // { binding: 0, resource: sampler },
      { binding: 1, resource: thicknessTextureView },
    ],
  })
  const filterBindGroups : GPUBindGroup[] = [
    device.createBindGroup({
      label: 'filterX bind group', 
      layout: filterPipeline.getBindGroupLayout(0),
      entries: [
        // { binding: 0, resource: sampler },
        { binding: 1, resource: renderTargetTextureView }, // 元の領域から読み込む
        { binding: 2, resource: { buffer: filterXUniformBuffer } },
      ],
    }), 
    device.createBindGroup({
      label: 'filterY bind group', 
      layout: filterPipeline.getBindGroupLayout(0),
      entries: [
        // { binding: 0, resource: sampler },
        { binding: 1, resource: tmpTargetTextureView }, // 一時領域から読み込む
        { binding: 2, resource: { buffer: filterYUniformBuffer }}
      ],
    })
  ];
  const fluidBindGroup = device.createBindGroup({
    label: 'fluid bind group', 
    layout: fluidPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: sampler },
      { binding: 1, resource: renderTargetTextureView },
      { binding: 2, resource: { buffer: fluidUniformBuffer } },
      { binding: 3, resource: thicknessTextureView },
      { binding: 4, resource: cubemapTextureView }, 
    ],
  })
  const thicknessBindGroup = device.createBindGroup({
    label: 'thickness bind group', 
    layout: thicknessPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particlesBuffer }},
      { binding: 1, resource: { buffer: uniformBuffer }},
    ],
  })
  const thicknessFilterBindGroups = [
    device.createBindGroup({
      label: 'thickness filterX bind group', 
      layout: thicknessFilterPipeline.getBindGroupLayout(0),
      entries: [
        // { binding: 0, resource: sampler },
        { binding: 1, resource: thicknessTextureView }, // 1 回目のパスはもとのテクスチャから
        { binding: 2, resource: { buffer: filterXUniformBuffer } }, 
      ],
    }), 
    device.createBindGroup({
      label: 'thickness filterY bind group', 
      layout: thicknessFilterPipeline.getBindGroupLayout(0),
      entries: [
        // { binding: 0, resource: sampler },
        { binding: 1, resource: tmpThicknessTextureView }, // 2 回目のパスは一時テクスチャから
        { binding: 2, resource: { buffer: filterYUniformBuffer } }, 
      ],
    }), 
  ]


  let isDragging = false;
  let prevX = 0;
  let prevY = 0;
  let deltaX = 0;
  let deltaY = 0;
  let currentXtheta = Math.PI / 4;
  let currentYtheta = -Math.PI / 12;
  const SENSITIVITY = 0.005;
  const MIN_YTHETA = -0.99 * Math.PI / 2.;
  const MAX_YTHETA = 0;
  let boxWidth = 1.0;

  const distanceParamsIndex = 0;
  const distanceParams = [
    {
      MIN_DISTANCE: 1.2, 
      MAX_DISTANCE: 3.0, 
      INIT_DISTANCE: 1.8
    }
  ]
  const distanceParam = distanceParams[distanceParamsIndex];
  let currentDistance = distanceParam.INIT_DISTANCE;

  const canvasElement = document.getElementById("fluidCanvas") as HTMLCanvasElement;

  canvasElement.addEventListener("mousedown", (event: MouseEvent) => {
    isDragging = true;
    prevX = event.clientX;
    prevY = event.clientY;
  });
  canvasElement.addEventListener("wheel", (event: WheelEvent) => {
    event.preventDefault();
    var scrollDelta = event.deltaY;
    currentDistance += ((scrollDelta > 0) ? 1 : -1) * 0.05;
    if (currentDistance < distanceParam.MIN_DISTANCE) currentDistance = distanceParam.MIN_DISTANCE;
    if (currentDistance > distanceParam.MAX_DISTANCE) currentDistance = distanceParam.MAX_DISTANCE;  
  })
  document.addEventListener("mousemove", (event: MouseEvent) => {
    if (isDragging) {
      const currentX = event.clientX;
      const currentY = event.clientY;
      const deltaX = prevX - currentX;
      const deltaY = prevY - currentY;
      currentXtheta += SENSITIVITY * deltaX;
      currentYtheta += SENSITIVITY * deltaY;
      if (currentYtheta > MAX_YTHETA) {
        currentYtheta = MAX_YTHETA
      }
      if (currentYtheta < MIN_YTHETA) {
        currentYtheta = MIN_YTHETA
      }
      prevX = currentX;
      prevY = currentY;
      console.log("Dragging... Delta:", deltaX, deltaY);
    }
  });
  document.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;
      console.log("Drag ended");
    }
  });


  // let t = -Math.PI / 2 * 0.98;
  let t = 0;
  let sign = 1;
  async function frame() {
    t += 0.05;
    // if (t < -Math.PI / 2) {
    //   sign *= -1;
    // } 
    // if (t > -Math.PI / 3) {
    //   sign *= -1;
    // }
    const start = performance.now();

    const circlePassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: renderTargetTextureView,
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    }

    const ballPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.7, g: 0.7, b: 0.7, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
      depthStencilAttachment: {
        view: depthTexture.createView(),
        depthClearValue: 1.0,
        depthLoadOp: 'clear',
        depthStoreOp: 'store',
      },
    }

    const filterPassDescriptors: GPURenderPassDescriptor[] = [
      {
        colorAttachments: [
          {
            view: tmpTargetTextureView, // 一時領域へ書き込み
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      }, 
      {
        colorAttachments: [
          {
            view: renderTargetTextureView, // Y のパスはもとに戻す
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      }
    ]
    const showPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    }
    const fluidPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: context.getCurrentTexture().createView(),
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    }
    const thicknessPassDescriptor: GPURenderPassDescriptor = {
      colorAttachments: [
        {
          view: thicknessTextureView, // 変える
          clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    }
    const thicknessFilterPassDescriptors: GPURenderPassDescriptor[] = [
      {
        colorAttachments: [
          {
            view: tmpThicknessTextureView, // 一時領域へ書き込み
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      }, 
      {
        colorAttachments: [
          {
            view: thicknessTextureView, // Y のパスはもとに戻す
            clearValue: { r: 0.0, g: 0.0, b: 0.0, a: 1.0 },
            loadOp: 'clear',
            storeOp: 'store',
          },
        ],
      }
    ]


    // 行列の更新
    uniformsViews.size.set([diameter]);
    uniformsViews.projection_matrix.set(projection);
    const view = recalculateView(currentDistance, currentYtheta, currentXtheta, [0., -yHalf, 0.]);
    uniformsViews.view_matrix.set(view);
    fluidUniformsViews.view_matrix.set(view);
    mat4.inverse(view, inv_view);
    fluidUniformsViews.inv_view_matrix.set(inv_view); // Don't forget!!!!
    device.queue.writeBuffer(uniformBuffer, 0, uniformsValues);
    device.queue.writeBuffer(fluidUniformBuffer, 0, fluidUniformsValues);
    // ボックスサイズの変更
    const slider = document.getElementById("slider") as HTMLInputElement;
    const sliderValue = document.getElementById("slider-value") as HTMLSpanElement;
    const particle = document.getElementById("particle") as HTMLInputElement;
    let curBoxWidth = parseInt(slider.value) / 200 + 0.5;
    const minClosingSpeed = -0.01;
    const dVal = Math.max(curBoxWidth - boxWidth, minClosingSpeed);
    boxWidth += dVal;
    sliderValue.textContent = curBoxWidth.toFixed(2);
    realBoxSizeViews.zHalf.set([zHalf * boxWidth]);
    device.queue.writeBuffer(realBoxSizeBuffer, 0, realBoxSizeValues);

    const commandEncoder = device.createCommandEncoder()

    // 計算のためのパス
    const computePass = commandEncoder.beginComputePass();
    for (let i = 0; i < 1; i++) { // ここは 2 であるべき
      computePass.setBindGroup(0, gridClearBindGroup);
      computePass.setPipeline(gridClearPipeline);
      computePass.dispatchWorkgroups(Math.ceil((gridCount + 1) / 64)) // これは gridCount だよな？
      computePass.setBindGroup(0, gridBuildBindGroup);
      computePass.setPipeline(gridBuildPipeline);
      computePass.dispatchWorkgroups(Math.ceil(numParticles / 64)) 

      const prefixSumKernel = new PrefixSumKernel({
        device: device, data: cellParticleCountBuffer, count: gridCount + 1
      })
      prefixSumKernel.dispatch(computePass);

      computePass.setBindGroup(0, reorderBindGroup);
      computePass.setPipeline(reorderPipeline)
      computePass.dispatchWorkgroups(Math.ceil(numParticles / 64))

      computePass.setBindGroup(0, densityBindGroup)
      computePass.setPipeline(densityPipeline)
      computePass.dispatchWorkgroups(Math.ceil(numParticles / 64))
      // この reorder をしないと，ソートした密度がゼロのままになる 
      computePass.setBindGroup(0, reorderBindGroup);
      computePass.setPipeline(reorderPipeline)
      computePass.dispatchWorkgroups(Math.ceil(numParticles / 64))
      computePass.setBindGroup(0, forceBindGroup)
      computePass.setPipeline(forcePipeline)
      computePass.dispatchWorkgroups(Math.ceil(numParticles / 64)) 
      computePass.setBindGroup(0, integrateBindGroup)
      computePass.setPipeline(integratePipeline)
      computePass.dispatchWorkgroups(Math.ceil(numParticles / 64)) 
    }
    computePass.end()

    // レンダリングのためのパス
    if (!particle.checked) {
      const circlePassEncoder = commandEncoder.beginRenderPass(circlePassDescriptor);
      circlePassEncoder.setBindGroup(0, circleBindGroup);
      circlePassEncoder.setPipeline(circlePipeline);
      circlePassEncoder.draw(6, numParticles);
      circlePassEncoder.end();
      for (var iter = 0; iter < 6; iter++) {
        const filterPassEncoder = commandEncoder.beginRenderPass(filterPassDescriptors[iter % 2]);
        filterPassEncoder.setBindGroup(0, filterBindGroups[iter % 2]);
        filterPassEncoder.setPipeline(filterPipeline);
        filterPassEncoder.draw(6);
        filterPassEncoder.end();  
      }
  
      const thicknessPassEncoder = commandEncoder.beginRenderPass(thicknessPassDescriptor);
      thicknessPassEncoder.setBindGroup(0, thicknessBindGroup);
      thicknessPassEncoder.setPipeline(thicknessPipeline);
      thicknessPassEncoder.draw(6, numParticles);
      thicknessPassEncoder.end();
  
      for (var iter = 0; iter < 4; iter++) { // 多いか？
        const thicknessFilterPassEncoder = commandEncoder.beginRenderPass(thicknessFilterPassDescriptors[iter % 2]);
        thicknessFilterPassEncoder.setBindGroup(0, thicknessFilterBindGroups[iter % 2]);
        thicknessFilterPassEncoder.setPipeline(thicknessFilterPipeline);
        thicknessFilterPassEncoder.draw(6);
        thicknessFilterPassEncoder.end(); 
      }

      const fluidPassEncoder = commandEncoder.beginRenderPass(fluidPassDescriptor);
      fluidPassEncoder.setBindGroup(0, fluidBindGroup);
      fluidPassEncoder.setPipeline(fluidPipeline);
      fluidPassEncoder.draw(6);
      fluidPassEncoder.end();
    } else {
      const ballPassEncoder = commandEncoder.beginRenderPass(ballPassDescriptor);
      ballPassEncoder.setBindGroup(0, ballBindGroup);
      ballPassEncoder.setPipeline(ballPipeline);
      ballPassEncoder.draw(6, numParticles);
      ballPassEncoder.end();
    }


    device.queue.submit([commandEncoder.finish()])
    const end = performance.now();
    console.log(`js: ${(end - start).toFixed(1)}ms`);

    requestAnimationFrame(frame)
  } 

  requestAnimationFrame(frame)
}

main()
