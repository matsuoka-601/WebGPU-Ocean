import gridClear from './grid/gridClear.wgsl'
import gridBuild from './grid/gridBuild.wgsl'
import reorderParticles from './grid/reorderParticles.wgsl'
import density from './density.wgsl'
import force from './force.wgsl'
import integrate from './integrate.wgsl'
import copyPosition from './copyPosition.wgsl'

import { PrefixSumKernel } from 'webgpu-radix-sort';

import { renderUniformsViews, numParticlesMax } from '../common';

export const sphParticleStructSize = 64

export class SPHSimulator {
    device: GPUDevice

    gridClearPipeline: GPUComputePipeline
    gridBuildPipeline: GPUComputePipeline
    reorderPipeline: GPUComputePipeline
    densityPipeline: GPUComputePipeline
    forcePipeline: GPUComputePipeline
    integratePipeline: GPUComputePipeline
    copyPositionPipeline: GPUComputePipeline

    gridClearBindGroup: GPUBindGroup
    gridBuildBindGroup: GPUBindGroup
    reorderBindGroup: GPUBindGroup
    densityBindGroup: GPUBindGroup
    forceBindGroup: GPUBindGroup
    integrateBindGroup: GPUBindGroup
    copyPositionBindGroup: GPUBindGroup

    cellParticleCountBuffer: GPUBuffer
    particleBuffer: GPUBuffer
    realBoxSizeBuffer: GPUBuffer
    sphParamsBuffer: GPUBuffer

    prefixSumKernel: any

    kernelRadius = 0.07
    numParticles = 0
    gridCount = 0

    renderDiameter: number

    constructor (particleBuffer: GPUBuffer, posvelBuffer: GPUBuffer, renderDiameter: number, device: GPUDevice) {
        this.device = device
        this.renderDiameter = renderDiameter
        const densityModule = device.createShaderModule({ code: density })
        const forceModule = device.createShaderModule({ code: force })
        const integrateModule = device.createShaderModule({ code: integrate })
        const gridBuildModule = device.createShaderModule({ code: gridBuild })
        const gridClearModule = device.createShaderModule({ code: gridClear })
        const reorderParticlesModule = device.createShaderModule({ code: reorderParticles })
        const copyPositionModule = device.createShaderModule({ code: copyPosition })

        const cellSize = 1.0 * this.kernelRadius
        const xHalfMax = 2.0
        const yHalfMax = 2.0
        const zHalfMax = 2.0
        const xLen = 2.0 * xHalfMax
        const yLen = 2.0 * yHalfMax
        const zLen = 2.0 * zHalfMax
        const sentinel = 4 * cellSize
        const xGrids = Math.ceil((xLen + sentinel) / cellSize)
        const yGrids = Math.ceil((yLen + sentinel) / cellSize)
        const zGrids = Math.ceil((zLen + sentinel) / cellSize)
        this.gridCount = xGrids * yGrids * zGrids;
        const offset = sentinel / 2;

        const stiffness = 20;
        const nearStiffness = 2.0;
        const mass = 1.0;
        const restDensity = 15000;
        const viscosity = 150;
        const dt = 0.006;

        this.gridClearPipeline = device.createComputePipeline({
            label: "grid clear pipeline", 
            layout: 'auto', 
            compute: {
                module: gridClearModule, 
            }
        })
        this.gridBuildPipeline = device.createComputePipeline({
            label: "grid build pipeline", 
            layout: 'auto', 
            compute: {
              module: gridBuildModule, 
            }
        })
        this.reorderPipeline = device.createComputePipeline({
            label: "reorder pipeline", 
            layout: 'auto', 
            compute: {
                module: reorderParticlesModule, 
            }
        })
        this.densityPipeline = device.createComputePipeline({
            label: "density pipeline", 
            layout: 'auto', 
            compute: {
              module: densityModule, 
            }
        });
        this.forcePipeline = device.createComputePipeline({
            label: "force pipeline", 
            layout: 'auto', 
            compute: {
              module: forceModule, 
            }
        });
        this.integratePipeline = device.createComputePipeline({
            label: "integrate pipeline", 
            layout: 'auto', 
            compute: {
              module: integrateModule, 
            }
        });
        this.copyPositionPipeline = device.createComputePipeline({
            label: "copy position pipeline", 
            layout: 'auto', 
            compute: {
              module: copyPositionModule, 
            }
        });

        const environmentValues = new ArrayBuffer(32);
        const environmentViews = {
            xGrids: new Int32Array(environmentValues, 0, 1),
            yGrids: new Int32Array(environmentValues, 4, 1),
            zGrids: new Int32Array(environmentValues, 8, 1),
            cellSize: new Float32Array(environmentValues, 12, 1),
            xHalf: new Float32Array(environmentValues, 16, 1),
            yHalf: new Float32Array(environmentValues, 20, 1),
            zHalf: new Float32Array(environmentValues, 24, 1),
            offset: new Float32Array(environmentValues, 28, 1),
        }
        environmentViews.xGrids.set([xGrids]); 
        environmentViews.yGrids.set([yGrids]); 
        environmentViews.zGrids.set([zGrids]); 
        environmentViews.cellSize.set([cellSize]); 
        environmentViews.xHalf.set([xHalfMax]); 
        environmentViews.yHalf.set([yHalfMax]); 
        environmentViews.zHalf.set([zHalfMax]); 
        environmentViews.offset.set([offset]); 

        const sphParamsValues = new ArrayBuffer(48);
        const sphParamsViews = {
            mass: new Float32Array(sphParamsValues, 0, 1),
            kernelRadius: new Float32Array(sphParamsValues, 4, 1),
            kernelRadiusPow2: new Float32Array(sphParamsValues, 8, 1),
            kernelRadiusPow5: new Float32Array(sphParamsValues, 12, 1),
            kernelRadiusPow6: new Float32Array(sphParamsValues, 16, 1),
            kernelRadiusPow9: new Float32Array(sphParamsValues, 20, 1),
            dt: new Float32Array(sphParamsValues, 24, 1),
            stiffness: new Float32Array(sphParamsValues, 28, 1),
            nearStiffness: new Float32Array(sphParamsValues, 32, 1),
            restDensity: new Float32Array(sphParamsValues, 36, 1),
            viscosity: new Float32Array(sphParamsValues, 40, 1),
            n: new Uint32Array(sphParamsValues, 44, 1),
        };
        sphParamsViews.mass.set([mass])
        sphParamsViews.kernelRadius.set([this.kernelRadius])
        sphParamsViews.kernelRadiusPow2.set([Math.pow(this.kernelRadius, 2)])
        sphParamsViews.kernelRadiusPow5.set([Math.pow(this.kernelRadius, 5)])
        sphParamsViews.kernelRadiusPow6.set([Math.pow(this.kernelRadius, 6)])
        sphParamsViews.kernelRadiusPow9.set([Math.pow(this.kernelRadius, 9)])
        sphParamsViews.dt.set([dt])
        sphParamsViews.stiffness.set([stiffness])
        sphParamsViews.nearStiffness.set([nearStiffness])
        sphParamsViews.restDensity.set([restDensity])
        sphParamsViews.viscosity.set([viscosity])
        // n はあとで


        const realBoxSizeValues = new ArrayBuffer(12);
        this.cellParticleCountBuffer = device.createBuffer({ // 累積和はここに保存
            label: 'cell particle count buffer', 
            size: 4 * (this.gridCount + 1),  // 1 要素余分にとっておく
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
        const targetParticlesBuffer = device.createBuffer({
            label: 'target particles buffer', 
            size: sphParticleStructSize * numParticlesMax, 
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
        const particleCellOffsetBuffer = device.createBuffer({
            label: 'particle cell offset buffer', 
            size: 4 * numParticlesMax,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
        })
        this.realBoxSizeBuffer = device.createBuffer({
            label: 'real box size buffer', 
            size: realBoxSizeValues.byteLength, 
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        const environmentBuffer = device.createBuffer({
            label: 'environment buffer', 
            size: environmentValues.byteLength, 
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        this.sphParamsBuffer = device.createBuffer({
            label: 'sph params buffer', 
            size: sphParamsValues.byteLength, 
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
        })
        device.queue.writeBuffer(environmentBuffer, 0, environmentValues)
        device.queue.writeBuffer(this.sphParamsBuffer, 0, sphParamsValues)

        // BindGroup
        this.gridClearBindGroup = device.createBindGroup({
            layout: this.gridClearPipeline.getBindGroupLayout(0), 
            entries: [
                { binding: 0, resource: { buffer: this.cellParticleCountBuffer }}, 
            ],  
        })
        this.gridBuildBindGroup = device.createBindGroup({
            layout: this.gridBuildPipeline.getBindGroupLayout(0), 
            entries: [
              { binding: 0, resource: { buffer: this.cellParticleCountBuffer }}, 
              { binding: 1, resource: { buffer: particleCellOffsetBuffer }}, 
              { binding: 2, resource: { buffer: particleBuffer }}, 
              { binding: 3, resource: { buffer: environmentBuffer }}, 
              { binding: 4, resource: { buffer: this.sphParamsBuffer }}, 
            ],  
        })
        this.reorderBindGroup = device.createBindGroup({
            layout: this.reorderPipeline.getBindGroupLayout(0), 
            entries: [
                { binding: 0, resource: { buffer: particleBuffer }}, 
                { binding: 1, resource: { buffer: targetParticlesBuffer }}, 
                { binding: 2, resource: { buffer: this.cellParticleCountBuffer }}, 
                { binding: 3, resource: { buffer: particleCellOffsetBuffer }}, 
                { binding: 4, resource: { buffer: environmentBuffer }}, 
                { binding: 5, resource: { buffer: this.sphParamsBuffer }}, 
            ]
        })
        
        this.densityBindGroup = device.createBindGroup({
            layout: this.densityPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer }},
                { binding: 1, resource: { buffer: targetParticlesBuffer }},
                { binding: 2, resource: { buffer: this.cellParticleCountBuffer }},
                { binding: 3, resource: { buffer: environmentBuffer }}, 
                { binding: 4, resource: { buffer: this.sphParamsBuffer }}, 
            ],
        })
        this.forceBindGroup = device.createBindGroup({
            layout: this.forcePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer }},
                { binding: 1, resource: { buffer: targetParticlesBuffer }},
                { binding: 2, resource: { buffer: this.cellParticleCountBuffer }},
                { binding: 3, resource: { buffer: environmentBuffer }}, 
                { binding: 4, resource: { buffer: this.sphParamsBuffer }}, 
            ],
        })
        this.integrateBindGroup = device.createBindGroup({
            layout: this.integratePipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer }},
                { binding: 1, resource: { buffer: this.realBoxSizeBuffer }},
                { binding: 2, resource: { buffer: this.sphParamsBuffer }},
            ],
        })
        this.copyPositionBindGroup = device.createBindGroup({
            layout: this.copyPositionPipeline.getBindGroupLayout(0),
            entries: [
                { binding: 0, resource: { buffer: particleBuffer }},
                { binding: 1, resource: { buffer: posvelBuffer }},
                { binding: 2, resource: { buffer: this.sphParamsBuffer }},
            ],
        })

        this.particleBuffer = particleBuffer
    }

    reset(numParticles: number, initHalfBoxSize: number[]) {
        renderUniformsViews.sphere_size.set([this.renderDiameter])
        const particleData = this.initDambreak(initHalfBoxSize, numParticles)
        const realBoxSizeValues = new ArrayBuffer(12);
        const realBoxSizeViews = {
            xHalf: new Float32Array(realBoxSizeValues, 0, 1),
            yHalf: new Float32Array(realBoxSizeValues, 4, 1),
            zHalf: new Float32Array(realBoxSizeValues, 8, 1),
        };
        realBoxSizeViews.xHalf.set([initHalfBoxSize[0]]); 
        realBoxSizeViews.yHalf.set([initHalfBoxSize[1]]); 
        realBoxSizeViews.zHalf.set([initHalfBoxSize[2]]);
        const numParticleValue = new Float32Array(1);
        numParticleValue[0] = this.numParticles
        console.log(this.numParticles)
        this.device.queue.writeBuffer(this.sphParamsBuffer, 44, numParticleValue) // TODO : avoid hardcoding
        this.device.queue.writeBuffer(this.particleBuffer, 0, particleData)
        this.device.queue.writeBuffer(this.realBoxSizeBuffer, 0, realBoxSizeValues)
    }

    execute(commandEncoder: GPUCommandEncoder) {
        const computePass = commandEncoder.beginComputePass();
        for (let i = 0; i < 2; i++) {
            computePass.setBindGroup(0, this.gridClearBindGroup);
            computePass.setPipeline(this.gridClearPipeline);
            computePass.dispatchWorkgroups(Math.ceil((this.gridCount + 1) / 64)) 
            computePass.setBindGroup(0, this.gridBuildBindGroup);
            computePass.setPipeline(this.gridBuildPipeline);
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64)) 
            this.prefixSumKernel = new PrefixSumKernel({
                device: this.device, data: this.cellParticleCountBuffer, count: this.gridCount + 1
            })
            this.prefixSumKernel.dispatch(computePass);
            computePass.setBindGroup(0, this.reorderBindGroup);
            computePass.setPipeline(this.reorderPipeline)
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64))

            computePass.setBindGroup(0, this.densityBindGroup)
            computePass.setPipeline(this.densityPipeline)
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64))
            computePass.setBindGroup(0, this.reorderBindGroup);
            computePass.setPipeline(this.reorderPipeline)
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64))
            computePass.setBindGroup(0, this.forceBindGroup)
            computePass.setPipeline(this.forcePipeline)
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64)) 
            computePass.setBindGroup(0, this.integrateBindGroup)
            computePass.setPipeline(this.integratePipeline)
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64)) 
            computePass.setBindGroup(0, this.copyPositionBindGroup)
            computePass.setPipeline(this.copyPositionPipeline)
            computePass.dispatchWorkgroups(Math.ceil(this.numParticles / 64))
        }

        computePass.end()
    }

    initDambreak(initHalfBoxSize: number[], numParticles: number) {
        let particlesBuf = new ArrayBuffer(sphParticleStructSize * numParticles);
        this.numParticles = 0;
        const DIST_FACTOR = 0.5
      
        for (var y = -initHalfBoxSize[1] * 0.95; this.numParticles < numParticles; y += DIST_FACTOR * this.kernelRadius) {
            for (var x = -0.95 * initHalfBoxSize[0]; x < 0.95 * initHalfBoxSize[0] && this.numParticles < numParticles; x += DIST_FACTOR * this.kernelRadius) {
                for (var z = -0.95 * initHalfBoxSize[2]; z < 0 * initHalfBoxSize[2] && this.numParticles < numParticles; z += DIST_FACTOR * this.kernelRadius) {
                    let jitter = 0.001 * Math.random();
                    const offset = sphParticleStructSize * this.numParticles;
                    const particleViews = {
                        position: new Float32Array(particlesBuf, offset + 0, 3),
                        v: new Float32Array(particlesBuf, offset + 16, 3),
                        force: new Float32Array(particlesBuf, offset + 32, 3),
                        density: new Float32Array(particlesBuf, offset + 44, 1),
                        nearDensity: new Float32Array(particlesBuf, offset + 48, 1),
                    };
                    particleViews.position.set([x + jitter, y + jitter, z + jitter]);
                    this.numParticles++;
                }
            }
        }

        console.log(this.numParticles)
        return particlesBuf;
    }

    changeBoxSize(realBoxSize: number[]) {
        const realBoxSizeValues = new ArrayBuffer(12);
        const realBoxSizeViews = new Float32Array(realBoxSizeValues);
        realBoxSizeViews.set(realBoxSize)
        this.device.queue.writeBuffer(this.realBoxSizeBuffer, 0, realBoxSizeViews)
    }
}