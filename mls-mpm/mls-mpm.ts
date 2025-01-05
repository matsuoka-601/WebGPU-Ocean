import clearGrid from './clearGrid.wgsl';
import p2g_1 from './p2g_1.wgsl';
import p2g_2 from './p2g_2.wgsl';
import updateGrid from './updateGrid.wgsl';
import g2p from './g2p.wgsl';

import { particleStructSize, numParticlesMax, renderUniformsViews } from '../common';

export class MLSMPMSimulator {
    max_x_grids = 64;
    max_y_grids = 64;
    max_z_grids = 64;
    cellStructSize = 16;
    realBoxSizeBuffer: GPUBuffer
    initBoxSizeBuffer: GPUBuffer
    numParticlesMax = 400000
    numParticles = 0
    gridCount = 0

    clearGridPipeline: GPUComputePipeline
    p2g1Pipeline: GPUComputePipeline
    p2g2Pipeline: GPUComputePipeline
    updateGridPipeline: GPUComputePipeline
    g2pPipeline: GPUComputePipeline

    clearGridBindGroup: GPUBindGroup
    p2g1BindGroup: GPUBindGroup
    p2g2BindGroup: GPUBindGroup
    updateGridBindGroup: GPUBindGroup
    g2pBindGroup: GPUBindGroup

    device: GPUDevice

    constructor (particleBuffer: GPUBuffer, initBoxSize: number[], diameter: number, device: GPUDevice) {
        this.device = device
        renderUniformsViews.sphere_size.set([diameter])
        const clearGridModule = device.createShaderModule({ code: clearGrid });
        const p2g1Module = device.createShaderModule({ code: p2g_1 });
        const p2g2Module = device.createShaderModule({ code: p2g_2 });
        const updateGridModule = device.createShaderModule({ code: updateGrid });
        const g2pModule = device.createShaderModule({ code: g2p });

        const constants = {
            stiffness: 3., 
            restDensity: 4., 
            dynamic_viscosity: 0.1, 
            dt: 0.20, 
            fixed_point_multiplier: 1e7, 
        }

        this.clearGridPipeline = device.createComputePipeline({
            label: "clear grid pipeline", 
            layout: 'auto', 
            compute: {
                module: clearGridModule, 
            }
        })
        this.p2g1Pipeline = device.createComputePipeline({
            label: "p2g 1 pipeline", 
            layout: 'auto', 
            compute: {
                module: p2g1Module, 
                constants: {
                    'fixed_point_multiplier': constants.fixed_point_multiplier
                }, 
            }
        })
        this.p2g2Pipeline = device.createComputePipeline({
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
        this.updateGridPipeline = device.createComputePipeline({
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
        this.g2pPipeline = device.createComputePipeline({
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

        const maxGridCount = this.max_x_grids * this.max_y_grids * this.max_z_grids;
        this.gridCount = Math.ceil(initBoxSize[0]) * Math.ceil(initBoxSize[1]) * Math.ceil(initBoxSize[2]);
        if (this.gridCount > maxGridCount) {
            throw new Error("gridCount should be equal to or less than maxGridCount")
        }

        const realBoxSizeValues = new ArrayBuffer(12);
        const realBoxSizeViews = new Float32Array(realBoxSizeValues);
        const initBoxSizeValues = new ArrayBuffer(12);
        const initBoxSizeViews = new Float32Array(initBoxSizeValues);
        initBoxSizeViews.set(initBoxSize);    
        realBoxSizeViews.set(initBoxSize); // 最初は同じ

        const cellBuffer = device.createBuffer({ 
            label: 'cells buffer', 
            size: this.cellStructSize * maxGridCount,  
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
        this.realBoxSizeBuffer = device.createBuffer({
            label: 'real box size buffer', 
            size: realBoxSizeValues.byteLength, 
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        this.initBoxSizeBuffer = device.createBuffer({
            label: 'init box size buffer', 
            size: initBoxSizeValues.byteLength, 
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        device.queue.writeBuffer(this.initBoxSizeBuffer, 0, initBoxSizeValues);
        device.queue.writeBuffer(this.realBoxSizeBuffer, 0, realBoxSizeValues);

        // BindGroup
        this.clearGridBindGroup = device.createBindGroup({
            layout: this.clearGridPipeline.getBindGroupLayout(0), 
            entries: [
              { binding: 0, resource: { buffer: cellBuffer }}, 
            ],  
        })
        this.p2g1BindGroup = device.createBindGroup({
            layout: this.p2g1Pipeline.getBindGroupLayout(0), 
            entries: [
                { binding: 0, resource: { buffer: particleBuffer }}, 
                { binding: 1, resource: { buffer: cellBuffer }}, 
                { binding: 2, resource: { buffer: this.initBoxSizeBuffer }}, 
            ],  
        })
        this.p2g2BindGroup = device.createBindGroup({
            layout: this.p2g2Pipeline.getBindGroupLayout(0), 
            entries: [
                { binding: 0, resource: { buffer: particleBuffer }}, 
                { binding: 1, resource: { buffer: cellBuffer }}, 
                { binding: 2, resource: { buffer: this.initBoxSizeBuffer }}, 
            ]
        })
        this.updateGridBindGroup = device.createBindGroup({
            layout: this.updateGridPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: cellBuffer }},
                { binding: 1, resource: { buffer: this.realBoxSizeBuffer }},
                { binding: 2, resource: { buffer: this.initBoxSizeBuffer }},
            ],
        })
        this.g2pBindGroup = device.createBindGroup({
            layout: this.g2pPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer }},
                { binding: 1, resource: { buffer: cellBuffer }},
                { binding: 2, resource: { buffer: this.realBoxSizeBuffer }},
                { binding: 3, resource: { buffer: this.initBoxSizeBuffer }},
            ],
        })

        const particlesData = this.initDambreak(initBoxSize);
        device.queue.writeBuffer(particleBuffer, 0, particlesData)
        console.log(this.numParticles)
    }

    initDambreak(initBoxSize: number[]) {
        let particlesBuf = new ArrayBuffer(particleStructSize * numParticlesMax);
        const spacing = 0.65;

        this.numParticles = 0;
        
        for (let j = 0; j < initBoxSize[1] * 0.40; j += spacing) {
            for (let i = 3; i < initBoxSize[0] - 4; i += spacing) {
                for (let k = 3; k < initBoxSize[2] / 2; k += spacing) {
                    const offset = particleStructSize * this.numParticles;
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
                    this.numParticles++;
                }
            }
        }
        
        let particles = new ArrayBuffer(particleStructSize * this.numParticles);
        const oldView = new Uint8Array(particlesBuf);
        const newView = new Uint8Array(particles);
        newView.set(oldView.subarray(0, newView.length));
        
        return particles;
    }

    execute(commandEncoder: GPUCommandEncoder) {
        const computePass = commandEncoder.beginComputePass();
        for (let i = 0; i < 2; i++) { 
            computePass.setBindGroup(0, this.clearGridBindGroup);
            computePass.setPipeline(this.clearGridPipeline);
            computePass.dispatchWorkgroups(Math.ceil(this.gridCount / 64)) // これは gridCount だよな？
            computePass.setBindGroup(0, this.p2g1BindGroup)
            computePass.setPipeline(this.p2g1Pipeline)
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64))
            computePass.setBindGroup(0, this.p2g2BindGroup)
            computePass.setPipeline(this.p2g2Pipeline)
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64)) 
            computePass.setBindGroup(0, this.updateGridBindGroup)
            computePass.setPipeline(this.updateGridPipeline)
            computePass.dispatchWorkgroups(Math.ceil(this.gridCount / 64)) 
            computePass.setBindGroup(0, this.g2pBindGroup)
            computePass.setPipeline(this.g2pPipeline)
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64)) 
        }
        computePass.end()
    }

    changeBoxSize(realBoxSize: number[]) {
        const realBoxSizeValues = new ArrayBuffer(12);
        const realBoxSizeViews = new Float32Array(realBoxSizeValues);
        realBoxSizeViews.set(realBoxSize)
        this.device.queue.writeBuffer(this.realBoxSizeBuffer, 0, realBoxSizeViews)
    }
}