import { mat4 } from 'wgpu-matrix'
import { renderUniformsValues, renderUniformsViews } from './common'

export class Camera {
    isDragging: boolean
    prevX: number
    prevY: number
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

    constructor (canvasElement: HTMLCanvasElement) {
        canvasElement.addEventListener("mousedown", (event: MouseEvent) => {
            this.isDragging = true;
            this.prevX = event.clientX;
            this.prevY = event.clientY;
        });

        canvasElement.addEventListener("wheel", (event: WheelEvent) => {
            event.preventDefault();
            var scrollDelta = event.deltaY;
            this.currentDistance += ((scrollDelta > 0) ? 1 : -1) * 1.5;
            if (this.currentDistance < this.minDistance) this.currentDistance = this.minDistance;
            if (this.currentDistance > this.maxDistance) this.currentDistance = this.maxDistance;  
            this.recalculateView()
        })

        canvasElement.addEventListener("mousemove", (event: MouseEvent) => {
            if (this.isDragging) {
                const currentX = event.clientX;
                const currentY = event.clientY;
                const deltaX = this.prevX - currentX;
                const deltaY = this.prevY - currentY;
                this.currentXtheta += this.sensitivity * deltaX;
                this.currentYtheta += this.sensitivity * deltaY;
                if (this.currentYtheta > this.maxYTheta) this.currentYtheta = this.maxYTheta
                if (this.currentYtheta < this.minYTheta) this.currentYtheta = this.minYTheta
                this.prevX = currentX;
                this.prevY = currentY;
                this.recalculateView()
            }
        });
        
        canvasElement.addEventListener("mouseup", () => {
            if (this.isDragging) this.isDragging = false;
        });
    }

    reset(canvasElement: HTMLCanvasElement, initDistance: number, target: number[], fov: number) {
        this.isDragging = false
        this.prevX = 0
        this.prevY = 0
        this.currentXtheta = Math.PI / 4 * 1
        this.currentYtheta = -Math.PI / 12
        this.maxYTheta = 0
        this.minYTheta = -0.99 * Math.PI / 2.
        this.sensitivity = 0.005
        this.currentDistance = initDistance
        this.maxDistance = 2. * this.currentDistance
        this.minDistance = 0.3 * this.currentDistance
        this.target = target
        this.fov = fov

        const aspect = canvasElement.clientWidth / canvasElement.clientHeight
        const projection = mat4.perspective(fov, aspect, 0.1, 500) // TODO : ここの max を変える
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
}