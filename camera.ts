import { mat4 } from 'wgpu-matrix'
import { renderUniformsValues, renderUniformsViews } from './common'

export class Camera {
    isDragging: boolean
    prevX: number
    prevY: number
    prevHoverX: number
    prevHoverY: number
    currentHoverX: number
    currentHoverY: number
    currentXtheta: number
    currentYtheta: number
    maxYTheta: number
    minYTheta: number
    sensitivity: number
    currentDistance: number
    maxDistance: number
    minDistance: number
    target: number[]
    fov: number
    zoomRate: number

    canvas: HTMLCanvasElement

    constructor (canvas: HTMLCanvasElement) {
        this.canvas = canvas;

        this.canvas.addEventListener("mousedown", (event: MouseEvent) => {
            this.isDragging = true;
            this.prevX = event.clientX;
            this.prevY = event.clientY;
        });

        this.canvas.addEventListener("wheel", (event: WheelEvent) => {
            event.preventDefault();
            var scrollDelta = event.deltaY;
            this.currentDistance += ((scrollDelta > 0) ? 1 : -1) * this.zoomRate;
            if (this.currentDistance < this.minDistance) this.currentDistance = this.minDistance;
            if (this.currentDistance > this.maxDistance) this.currentDistance = this.maxDistance;  
            this.recalculateView()
        })

        this.canvas.addEventListener("mousemove", (event: MouseEvent) => {
            this.currentHoverX = event.clientX;
            this.currentHoverY = event.clientY;
            if (this.isDragging) {
                const deltaX = this.prevX - event.clientX;
                // const deltaY = this.prevY - event.clientY;
                this.currentXtheta += this.sensitivity * deltaX;
                // this.currentYtheta += this.sensitivity * deltaY;
                if (this.currentYtheta > this.maxYTheta) this.currentYtheta = this.maxYTheta
                if (this.currentYtheta < this.minYTheta) this.currentYtheta = this.minYTheta
                this.prevX = event.clientX;
                this.prevY = event.clientY;
                this.recalculateView()
            }
        });
        
        this.canvas.addEventListener("mouseup", () => {
            if (this.isDragging) this.isDragging = false;
        });
    }

    reset(initDistance: number, target: number[], fov: number, zoomRate: number) {
        this.isDragging = false
        this.prevX = 0
        this.prevY = 0
        this.currentXtheta = -Math.PI / 2 * 1
        // this.currentYtheta = -Math.PI / 12
        this.currentYtheta = 0
        this.maxYTheta = 0
        this.minYTheta = -0.99 * Math.PI / 2.
        this.sensitivity = 0.005
        this.currentDistance = initDistance
        this.maxDistance = 2. * this.currentDistance
        this.minDistance = 0.7 * this.currentDistance
        this.target = target
        this.fov = fov
        this.zoomRate = zoomRate

        const aspect = this.canvas.clientWidth / this.canvas.clientHeight
        const projection = mat4.perspective(fov, aspect, 0.1, 300) 
        renderUniformsViews.projection_matrix.set(projection)
        renderUniformsViews.inv_projection_matrix.set(mat4.inverse(projection))
        this.recalculateView()
    }

    recalculateView() {
        var mat = mat4.identity();
        mat4.translate(mat, this.target, mat)
        mat4.rotateY(mat, this.currentXtheta, mat)
        mat4.rotateX(mat, this.currentYtheta, mat)
        mat4.translate(mat, [0, 0, this.currentDistance], mat)
        var position = mat4.multiply(mat, [0, 0, 0, 1])

        const view = mat4.lookAt(
          [position[0], position[1], position[2]], // position
          this.target, // target
          [0, 1, 0], // up
        )

        renderUniformsViews.view_matrix.set(view)
        renderUniformsViews.inv_view_matrix.set(mat4.inverse(view))
    }

    calcMouseVelocity() {
        let velX = (this.currentHoverX - this.prevHoverX) / this.canvas.width * (this.canvas.width / this.canvas.height);
        let velY = -(this.currentHoverY - this.prevHoverY) / this.canvas.height;
        return [velX, velY]
    }

    setNewPrevMouseCoord() {
        this.prevHoverX = this.currentHoverX;
        this.prevHoverY = this.currentHoverY;
    }
}