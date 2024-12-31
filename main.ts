import shader from './render/shader.wgsl'
import show from './render/show.wgsl'
import filter from './render/bilateral.wgsl'
import fluid from './render/fluid.wgsl'
import vertex from './render/vertex.wgsl'
import thickness from './render/thickness.wgsl'
import gaussian from './render/gaussian.wgsl'
import ball from './render/ball.wgsl'

import clearGrid from './mls-mpm/clearGrid.wgsl';
import p2g_1 from './mls-mpm/p2g_1.wgsl';
import p2g_2 from './mls-mpm/p2g_2.wgsl';
import updateGrid from './mls-mpm/updateGrid.wgsl';
import g2p from './mls-mpm/g2p.wgsl';

import { PrefixSumKernel } from 'webgpu-radix-sort';
import { mat4 } from 'wgpu-matrix'

/// <reference types="@webgpu/types" />

const numParticlesMax = 400000;
const particleStructSize = 112;
const cellStructSize = 16;
const max_x_grids = 64;
const max_y_grids = 64;
const max_z_grids = 64;
let numParticles = 0;
function init_dambreak(init_box_size: number[]) {
  let particlesBuf = new ArrayBuffer(particleStructSize * numParticlesMax);
  const spacing = 0.65;

  for (let j = 3; j < init_box_size[1] * 0.6; j += spacing) {
    for (let i = 3; i < init_box_size[0] - 5; i += spacing) {
      for (let k = 3; k < init_box_size[2] / 2; k += spacing) {
        const offset = particleStructSize * numParticles;
        const particleViews = {
          position: new Float32Array(particlesBuf, offset + 0, 3),
          v: new Float32Array(particlesBuf, offset + 16, 3),
          C: new Float32Array(particlesBuf, offset + 32, 12),
          force: new Float32Array(particlesBuf, offset + 80, 3),
          density: new Float32Array(particlesBuf, offset + 92, 1),
          nearDensity: new Float32Array(particlesBuf, offset + 96, 1),
        };
        const jitter = 2.0 * Math.random();
        particleViews.position.set([i + jitter, j + jitter, k + jitter]);
        numParticles++;
      }
    }
  }

  let particles = new ArrayBuffer(particleStructSize * numParticles);
  const oldView = new Uint8Array(particlesBuf);
  const newView = new Uint8Array(particles);
  newView.set(oldView.subarray(0, newView.length));


  return particles;
}


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
  const projection = mat4.perspective(fov, aspect, 0.1, 500)
  const view = mat4.identity()
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

  const clearGridModule = device.createShaderModule({ code: clearGrid });
  const p2g1Module = device.createShaderModule({ code: p2g_1 });
  const p2g2Module = device.createShaderModule({ code: p2g_2 });
  const updateGridModule = device.createShaderModule({ code: updateGrid });
  const g2pModule = device.createShaderModule({ code: g2p });

  const constants = {
    stiffness: 3., 
    restDensity: 4., 
    dynamic_viscosity: 0.03, 
    dt: 0.20, 
    fixed_point_multiplier: 1e7, 
  }
  const maxGridCount = max_x_grids * max_y_grids * max_z_grids;
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

  const fov = 45 * Math.PI / 180;
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
  const clearGridPipeline = device.createComputePipeline({
    label: "clear grid pipeline", 
    layout: 'auto', 
    compute: {
      module: clearGridModule, 
    }
  })
  const p2g1Pipeline = device.createComputePipeline({
    label: "p2g 1 pipeline", 
    layout: 'auto', 
    compute: {
      module: p2g1Module, 
      constants: {
        'fixed_point_multiplier': constants.fixed_point_multiplier
      }, 
    }
  })
  const p2g2Pipeline = device.createComputePipeline({
    label: "p2g 2 pipeline", 
    layout: 'auto', 
    compute: {
      module: p2g2Module, 
      constants: {
        'fixed_point_multiplier': constants.fixed_point_multiplier, 
        'stiffness': constants.stiffness, 
        'rest_density': constants.restDensity, 
        'dynamic_viscosity': constants.dynamic_viscosity, 
        'dt': constants.dt, 
      }, 
    }
  })
  const updateGridPipeline = device.createComputePipeline({
    label: "update grid pipeline", 
    layout: 'auto', 
    compute: {
      module: updateGridModule, 
      constants: {
        'fixed_point_multiplier': constants.fixed_point_multiplier, 
        'dt': constants.dt, 
      }, 
    }
  });
  const g2pPipeline = device.createComputePipeline({
    label: "g2p pipeline", 
    layout: 'auto', 
    compute: {
      module: g2pModule, 
      constants: {
        'fixed_point_multiplier': constants.fixed_point_multiplier, 
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
  const realBoxSizeViews = new Float32Array(realBoxSizeValues);
  const initBoxSizeValues = new ArrayBuffer(12);
  const initBoxSizeViews = new Float32Array(initBoxSizeValues);

  // storage buffer を作る
  const particlesBuffer = device.createBuffer({
    label: 'particles buffer', 
    size: particleStructSize * numParticlesMax, 
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })
  const cellsBuffer = device.createBuffer({ 
    label: 'cells buffer', 
    size: cellStructSize * maxGridCount,  
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  })


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
  const initBoxSizeBuffer = device.createBuffer({
    label: 'init box size buffer', 
    size: initBoxSizeValues.byteLength, 
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  })
  device.queue.writeBuffer(filterXUniformBuffer, 0, filterXUniformsValues);
  device.queue.writeBuffer(filterYUniformBuffer, 0, filterYUniformsValues);
  device.queue.writeBuffer(fluidUniformBuffer, 0, fluidUniformsValues);


  // 計算の bindGroup
  const clearGridBindGroup = device.createBindGroup({
      layout: clearGridPipeline.getBindGroupLayout(0), 
      entries: [
        { binding: 0, resource: { buffer: cellsBuffer }}, 
      ],  
  })
  const p2g1BindGroup = device.createBindGroup({
    layout: p2g1Pipeline.getBindGroupLayout(0), 
    entries: [
      { binding: 0, resource: { buffer: particlesBuffer }}, 
      { binding: 1, resource: { buffer: cellsBuffer }}, 
      { binding: 2, resource: { buffer: initBoxSizeBuffer }}, 
    ],  
  })
  const p2g2BindGroup = device.createBindGroup({
    layout: p2g2Pipeline.getBindGroupLayout(0), 
    entries: [
      { binding: 0, resource: { buffer: particlesBuffer }}, 
      { binding: 1, resource: { buffer: cellsBuffer }}, 
      { binding: 2, resource: { buffer: initBoxSizeBuffer }}, 
    ]
  })
  const updateGridBindGroup = device.createBindGroup({
    layout: updateGridPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: cellsBuffer }},
      { binding: 1, resource: { buffer: realBoxSizeBuffer }},
      { binding: 2, resource: { buffer: initBoxSizeBuffer }},
    ],
  })
  const g2pBindGroup = device.createBindGroup({
    layout: g2pPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: particlesBuffer }},
      { binding: 1, resource: { buffer: cellsBuffer }},
      { binding: 2, resource: { buffer: realBoxSizeBuffer }},
      { binding: 3, resource: { buffer: initBoxSizeBuffer }},
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
  let currentXtheta = -Math.PI / 2;
  let currentYtheta = -Math.PI / 12;
  const SENSITIVITY = 0.005;
  const MIN_YTHETA = -0.99 * Math.PI / 2.;
  const MAX_YTHETA = 0;
  let boxWidthRatio = 1.0;

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

  canvasElement.addEventListener("mousedown", (event: MouseEvent) => {
    isDragging = true;
    prevX = event.clientX;
    prevY = event.clientY;
  });
  canvasElement.addEventListener("wheel", (event: WheelEvent) => {
    event.preventDefault();
    var scrollDelta = event.deltaY;
    currentDistance += ((scrollDelta > 0) ? 1 : -1) * 0.5;
    const distanceParam = distanceParams[distanceParamsIndex];
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
    }
  });
  document.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;
    }
  });

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

  let init_box_size = [50, 50, 50];
  let real_box_size = [...init_box_size];
  const particlesData = init_dambreak(init_box_size);

  device.queue.writeBuffer(particlesBuffer, 0, particlesData)
  
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

  console.log(numParticles);

  let ballFl = false;
  let t = 0;
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
          clearValue: { r: 0.8, g: 0.8, b: 0.8, a: 1.0 },
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
    const view = recalculateView(currentDistance, currentYtheta, currentXtheta,
         [init_box_size[0] / 2., init_box_size[1] / 4, init_box_size[2] / 2.]
      );
    uniformsViews.view_matrix.set(view);
    fluidUniformsViews.view_matrix.set(view);
    mat4.inverse(view, inv_view);
    fluidUniformsViews.inv_view_matrix.set(inv_view); // Don't forget!!!!
    real_box_size[2] = init_box_size[2] * (0.25 * (Math.cos(0 * t) + 1.) + 0.5);
    realBoxSizeViews.set(real_box_size);
    initBoxSizeViews.set(init_box_size);
    device.queue.writeBuffer(uniformBuffer, 0, uniformsValues);
    device.queue.writeBuffer(fluidUniformBuffer, 0, fluidUniformsValues);
    device.queue.writeBuffer(realBoxSizeBuffer, 0, realBoxSizeValues);
    device.queue.writeBuffer(initBoxSizeBuffer, 0, initBoxSizeValues);
    const gridCount = Math.ceil(init_box_size[0]) * Math.ceil(init_box_size[1]) * Math.ceil(init_box_size[2]);
    if (gridCount > maxGridCount) {
      throw new Error("gridCount is bigger than maxGridCount");
    }
    // ボックスサイズの変更
    // const slider = document.getElementById("slider") as HTMLInputElement;
    // const sliderValue = document.getElementById("slider-value") as HTMLSpanElement;
    // const particle = document.getElementById("particle") as HTMLInputElement;
    // let curBoxWidthRatio = parseInt(slider.value) / 200 + 0.5;
    // const minClosingSpeed = -0.01;
    // const dVal = Math.max(curBoxWidthRatio - boxWidthRatio, minClosingSpeed);
    // boxWidthRatio += dVal;
    // sliderValue.textContent = curBoxWidthRatio.toFixed(2);
    // realBoxSizeViews.xHalf.set([environment.boxSize.xHalf]);
    // realBoxSizeViews.yHalf.set([environment.boxSize.yHalf]);
    // realBoxSizeViews.zHalf.set([environment.boxSize.zHalf * boxWidthRatio]);
    // device.queue.writeBuffer(realBoxSizeBuffer, 0, realBoxSizeValues);

    const commandEncoder = device.createCommandEncoder()

    // 計算のためのパス
    const computePass = commandEncoder.beginComputePass();
    for (let i = 0; i < 2; i++) { 
      computePass.setBindGroup(0, clearGridBindGroup);
      computePass.setPipeline(clearGridPipeline);
      computePass.dispatchWorkgroups(Math.ceil(gridCount / 64)) // これは gridCount だよな？
      computePass.setBindGroup(0, p2g1BindGroup)
      computePass.setPipeline(p2g1Pipeline)
      computePass.dispatchWorkgroups(Math.ceil(numParticles / 64))
      computePass.setBindGroup(0, p2g2BindGroup)
      computePass.setPipeline(p2g2Pipeline)
      computePass.dispatchWorkgroups(Math.ceil(numParticles / 64)) 
      computePass.setBindGroup(0, updateGridBindGroup)
      computePass.setPipeline(updateGridPipeline)
      computePass.dispatchWorkgroups(Math.ceil(gridCount / 64)) 
      computePass.setBindGroup(0, g2pBindGroup)
      computePass.setPipeline(g2pPipeline)
      computePass.dispatchWorkgroups(Math.ceil(numParticles / 64)) 
    }
    computePass.end()

    // レンダリングのためのパス
    if (!ballFl) {
      const circlePassEncoder = commandEncoder.beginRenderPass(circlePassDescriptor);
      circlePassEncoder.setBindGroup(0, circleBindGroup);
      circlePassEncoder.setPipeline(circlePipeline);
      circlePassEncoder.draw(6, numParticles);
      circlePassEncoder.end();
      for (var iter = 0; iter < 5; iter++) {
        const filterPassEncoderX = commandEncoder.beginRenderPass(filterPassDescriptors[0]);
        filterPassEncoderX.setBindGroup(0, filterBindGroups[0]);
        filterPassEncoderX.setPipeline(filterPipeline);
        filterPassEncoderX.draw(6);
        filterPassEncoderX.end();  
        const filterPassEncoderY = commandEncoder.beginRenderPass(filterPassDescriptors[1]);
        filterPassEncoderY.setBindGroup(0, filterBindGroups[1]);
        filterPassEncoderY.setPipeline(filterPipeline);
        filterPassEncoderY.draw(6);
        filterPassEncoderY.end();  
      }
  
      const thicknessPassEncoder = commandEncoder.beginRenderPass(thicknessPassDescriptor);
      thicknessPassEncoder.setBindGroup(0, thicknessBindGroup);
      thicknessPassEncoder.setPipeline(thicknessPipeline);
      thicknessPassEncoder.draw(6, numParticles);
      thicknessPassEncoder.end();
  
      for (var iter = 0; iter < 1; iter++) { // 多いか？
        const thicknessFilterPassEncoderX = commandEncoder.beginRenderPass(thicknessFilterPassDescriptors[0]);
        thicknessFilterPassEncoderX.setBindGroup(0, thicknessFilterBindGroups[0]);
        thicknessFilterPassEncoderX.setPipeline(thicknessFilterPipeline);
        thicknessFilterPassEncoderX.draw(6);
        thicknessFilterPassEncoderX.end(); 
        const thicknessFilterPassEncoderY = commandEncoder.beginRenderPass(thicknessFilterPassDescriptors[1]);
        thicknessFilterPassEncoderY.setBindGroup(0, thicknessFilterBindGroups[1]);
        thicknessFilterPassEncoderY.setPipeline(thicknessFilterPipeline);
        thicknessFilterPassEncoderY.draw(6);
        thicknessFilterPassEncoderY.end(); 
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