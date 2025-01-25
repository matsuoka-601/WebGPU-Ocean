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

	// const { devicePixelRatio } = window
	// let devicePixelRatio  = 5.0;
	let devicePixelRatio  = 0.5;
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

	console.log("initialization done")

	context.configure({
		device,
		format: presentationFormat,
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
	console.log("cubemap initialization done")

	// uniform buffer を作る
	renderUniformsViews.texel_size.set([1.0 / canvas.width, 1.0 / canvas.height]);

	// storage buffer を作る
	const maxParticleStructSize = Math.max(mlsmpmParticleStructSize, sphParticleStructSize)
	const particleBuffer = device.createBuffer({
		label: 'particles buffer', 
		size: maxParticleStructSize * numParticlesMax, 
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

	console.log("buffer allocating done")

	let mlsmpmNumParticleParams = [40000, 60000, 120000, 200000]
	let mlsmpmInitBoxSizes = [[35, 40, 55], [60, 60, 60], [45, 40, 80], [50, 50, 80]]
	let mlsmpmInitDistances = [60, 70, 90, 100]
	let sphNumParticleParams = [10000, 20000, 30000, 40000]
	let sphInitBoxSizes = [[0.7, 2.0, 0.7], [1.0, 2.0, 1.0], [1.2, 2.0, 1.2], [1.4, 2.0, 1.4]]
	let sphInitDistances = [2.6, 3.0, 3.4, 3.8]

	const canvasElement = document.getElementById("fluidCanvas") as HTMLCanvasElement;
	// シミュレーション，カメラの初期化
	const mlsmpmFov = 45 * Math.PI / 180
	const mlsmpmRadius = 0.7
	const mlsmpmDiameter = 2 * mlsmpmRadius
	const mlsmpmZoomRate = 1.5
	const mlsmpmSimulator = new MLSMPMSimulator(particleBuffer, posvelBuffer, mlsmpmDiameter, device)
	const sphFov = 45 * Math.PI / 180
	const sphRadius = 0.04
	const sphDiameter = 2 * sphRadius
	const sphZoomRate = 0.05
	const sphSimulator = new SPHSimulator(particleBuffer, posvelBuffer, sphDiameter, device)

	const mlsmpmRenderer = new FluidRenderer(device, canvas, presentationFormat, mlsmpmRadius, mlsmpmFov, posvelBuffer, renderUniformBuffer, cubemapTextureView)
	const sphRenderer = new FluidRenderer(device, canvas, presentationFormat, sphRadius, sphFov, posvelBuffer, renderUniformBuffer, cubemapTextureView)

	console.log("simulator initialization done")

	const camera = new Camera(canvasElement);

	// ボタン押下の監視
	let numberButtonForm = document.getElementById('number-button') as HTMLFormElement;
	let numberButtonPressed = false;
	let numberButtonPressedButton = "1"
	numberButtonForm.addEventListener('change', function(event) {
		const target = event.target as HTMLInputElement
		if (target?.name === 'options') {
			numberButtonPressed = true
			numberButtonPressedButton = target.value
		}
	}); 
	let simulationModeForm = document.getElementById('simulation-mode') as HTMLFormElement;
	let simulationModePressed = false;
	let simulationModePressedButton = "mls-mpm"
	simulationModeForm.addEventListener('change', function(event) {
		const target = event.target as HTMLInputElement
		if (target?.name === 'options') {
			simulationModePressed = true
			simulationModePressedButton = target.value
		}
	}); 
	// スペースキーの監視
	let stop = false;
	document.addEventListener("keydown", (event) => {
		if (event.code === "Space") {
			event.preventDefault(); 
			stop = !stop;
		}
	});

	const smallValue = document.getElementById("small-value") as HTMLSpanElement;
	const mediumValue = document.getElementById("medium-value") as HTMLSpanElement;
	const largeValue = document.getElementById("large-value") as HTMLSpanElement;
	const veryLargeValue = document.getElementById("very-large-value") as HTMLSpanElement;

	// デバイスロストの監視
	let errorLog = document.getElementById('error-reason') as HTMLSpanElement;
	errorLog.textContent = "";
	device.lost.then(info => {
		const reason = info.reason ? `reason: ${info.reason}` : 'unknown reason';
		errorLog.textContent = reason;
	});

	// はじめは mls-mpm
	const initDistance = mlsmpmInitDistances[1]
	let initBoxSize = mlsmpmInitBoxSizes[1]
	let realBoxSize = [...initBoxSize];
	mlsmpmSimulator.reset(mlsmpmNumParticleParams[1], mlsmpmInitBoxSizes[1])
	camera.reset(canvasElement, initDistance, [initBoxSize[0] / 2, initBoxSize[1] / 2, initBoxSize[2] / 2], 
		mlsmpmFov, mlsmpmZoomRate)

	smallValue.textContent = "40,000"
	mediumValue.textContent = "70,000"
	largeValue.textContent = "120,000"
	veryLargeValue.textContent = "200,000"

	let sphereRenderFl = false
	let sphFl = false
	let boxWidthRatio = 1.

	console.log("simulation start")
	async function frame() {
		const start = performance.now();

		if (simulationModePressed) {
			if (simulationModePressedButton == "mlsmpm") {
				sphFl = false
				smallValue.textContent = "40,000"
				mediumValue.textContent = "70,000"
				largeValue.textContent = "120,000"
				veryLargeValue.textContent = "200,000"
			} else {
				sphFl = true
				smallValue.textContent = "10,000"
				mediumValue.textContent = "20,000"
				largeValue.textContent = "30,000"
				veryLargeValue.textContent = "40,000"
			}
			simulationModePressed = false
			numberButtonPressed = true 
		}

		if (numberButtonPressed) { 
			const paramsIdx = parseInt(numberButtonPressedButton)
			if (sphFl) {
				initBoxSize = sphInitBoxSizes[paramsIdx]
				sphSimulator.reset(sphNumParticleParams[paramsIdx], initBoxSize)
				camera.reset(canvasElement, sphInitDistances[paramsIdx], [0, -initBoxSize[1] + 0.1, 0], 
					sphFov, sphZoomRate)
			} else {
				initBoxSize = mlsmpmInitBoxSizes[paramsIdx]
				mlsmpmSimulator.reset(mlsmpmNumParticleParams[paramsIdx], initBoxSize)
				camera.reset(canvasElement, mlsmpmInitDistances[paramsIdx], [initBoxSize[0] / 2, initBoxSize[1] / 4, initBoxSize[2] / 2], 
					mlsmpmFov, mlsmpmZoomRate)
			}
			realBoxSize = [...initBoxSize]
			let slider = document.getElementById("slider") as HTMLInputElement
			slider.value = "100"
			numberButtonPressed = false
		}

		// ボックスサイズの変更
		const slider = document.getElementById("slider") as HTMLInputElement
		const particle = document.getElementById("particle") as HTMLInputElement
		sphereRenderFl = particle.checked
		let curBoxWidthRatio = parseInt(slider.value) / 200 + 0.5
		const minClosingSpeed = sphFl ? -0.015 : -0.007
		const dVal = Math.max(curBoxWidthRatio - boxWidthRatio, minClosingSpeed)
		boxWidthRatio += dVal

		// 行列の更新
		realBoxSize[2] = initBoxSize[2] * boxWidthRatio
		if (sphFl) {
			sphSimulator.changeBoxSize(realBoxSize)
		} else {
			mlsmpmSimulator.changeBoxSize(realBoxSize)
		}
		device.queue.writeBuffer(renderUniformBuffer, 0, renderUniformsValues) 

		const commandEncoder = device.createCommandEncoder()

		// 計算のためのパス
		if (sphFl) {
			if (!stop) sphSimulator.execute(commandEncoder)
			sphRenderer.execute(context, commandEncoder, sphSimulator.numParticles, sphereRenderFl)
		} else {
			if (!stop) mlsmpmSimulator.execute(commandEncoder)
			mlsmpmRenderer.execute(context, commandEncoder, mlsmpmSimulator.numParticles, sphereRenderFl)
		}


		device.queue.submit([commandEncoder.finish()])
		const end = performance.now();
		// console.log(`js: ${(end - start).toFixed(1)}ms`);

		requestAnimationFrame(frame)
	} 
	requestAnimationFrame(frame)
}

main()