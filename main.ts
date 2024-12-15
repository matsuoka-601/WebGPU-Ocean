import triangleVertWGSL from './shader/triangle.vert.wgsl';
import redFragWGSL from './shader/red.frag.wgsl';

main();

async function main() {
	const canvas = document.querySelector('canvas') as HTMLCanvasElement;
	const adapter = await navigator.gpu.requestAdapter();
	if (!adapter) {
		throw new Error("WebGPU not supported")
	}
	const device = await adapter.requestDevice();

	const context = canvas.getContext('webgpu') as GPUCanvasContext;

	const devicePixelRatio = window.devicePixelRatio;
	canvas.width = canvas.clientWidth * devicePixelRatio;
	canvas.height = canvas.clientHeight * devicePixelRatio;
	const presentationFormat = navigator.gpu.getPreferredCanvasFormat();

	context.configure({
		device,
		format: presentationFormat,
	});

	const pipeline = device.createRenderPipeline({
		layout: 'auto',
		vertex: {
		module: device.createShaderModule({
			code: triangleVertWGSL,
		}),
		},
		fragment: {
		module: device.createShaderModule({
			code: redFragWGSL,
		}),
		targets: [
			{
			format: presentationFormat,
			},
		],
		},
		primitive: {
		topology: 'triangle-list',
		},
	});

	function frame() {
		const commandEncoder = device.createCommandEncoder();
		const textureView = context.getCurrentTexture().createView();

		const renderPassDescriptor: GPURenderPassDescriptor = {
		colorAttachments: [
			{
			view: textureView,
			clearValue: [0, 0, 0, 0], // Clear to transparent
			loadOp: 'clear',
			storeOp: 'store',
			},
		],
		};

		const passEncoder = commandEncoder.beginRenderPass(renderPassDescriptor);
		passEncoder.setPipeline(pipeline);
		passEncoder.draw(3);
		passEncoder.end();

		device.queue.submit([commandEncoder.finish()]);
		requestAnimationFrame(frame);
	}

	requestAnimationFrame(frame);
}

