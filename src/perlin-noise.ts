"use strict";
import tgpu from "typegpu";
import { arrayOf, u32, vec2u } from "typegpu/data";

type Position = [number, number];

class PerlinNoise {
  gradients: Array<Position>;
  memory: Array<number>;
  gridSize: number;
  resolution: number;

  constructor(gridSize: number, resolution: number) {
    this.gridSize = gridSize;
    this.resolution = resolution;
    this.gradients = [];
    this.memory = [];
  }
  randonAngledVector(): Position {
    const theta = Math.random() * 2 * Math.PI;
    return [Math.cos(theta), Math.sin(theta)];
  }
  dotProdGrid(position: Position, gridPosition: Position) {
    let gridIndexes: Position;
    let positionInAGridCell = [
      position[0] - gridPosition[0],
      position[1] - gridPosition[1],
    ];
    const gradientsIndex = Math.round(
      gridPosition[0] + gridPosition[1] * this.gridSize
    );
    if (this.gradients[gradientsIndex]) {
      gridIndexes = this.gradients[gradientsIndex];
    } else {
      gridIndexes = this.randonAngledVector();
      this.gradients[gradientsIndex] = gridIndexes;
    }
    return (
      positionInAGridCell[0] * gridIndexes[0] +
      positionInAGridCell[1] * gridIndexes[1]
    );
  }
  smootherstep(x: number) {
    return 6 * x ** 5 - 15 * x ** 4 + 10 * x ** 3;
  }
  interpolate(x: number, a: number, b: number) {
    return a + this.smootherstep(x) * (b - a);
  }
  get(positionOnGrid: Position) {
    const multiplier = this.resolution / this.gridSize;
    const memoryIndex = Math.round(
      (positionOnGrid[0] + positionOnGrid[1] * this.resolution) * multiplier
    );
    if (this.memory[memoryIndex]) return this.memory[memoryIndex];
    const gridCellIndexes: Position = [
      Math.floor(positionOnGrid[0]),
      Math.floor(positionOnGrid[1]),
    ];
    //interpolate
    const interpolatedValues = {
      topLeft: this.dotProdGrid(positionOnGrid, gridCellIndexes),
      topRight: this.dotProdGrid(positionOnGrid, [
        gridCellIndexes[0] + 1,
        gridCellIndexes[1],
      ]), // positionOnGrid, gridCellIndexes
      bottomLeft: this.dotProdGrid(positionOnGrid, [
        gridCellIndexes[0],
        gridCellIndexes[1] + 1,
      ]), // positionOnGrid, gridCellIndexes
      bottomRight: this.dotProdGrid(positionOnGrid, [
        gridCellIndexes[0] + 1,
        gridCellIndexes[1] + 1,
      ]),
    };
    const topValueToInterpolate = this.interpolate(
      positionOnGrid[0] - gridCellIndexes[0],
      interpolatedValues.topLeft,
      interpolatedValues.topRight
    );
    const bottomValueToInterpolate = this.interpolate(
      positionOnGrid[0] - gridCellIndexes[0],
      interpolatedValues.bottomLeft,
      interpolatedValues.bottomRight
    );
    const interpolatedValue = this.interpolate(
      positionOnGrid[1] - gridCellIndexes[1],
      topValueToInterpolate,
      bottomValueToInterpolate
    );
    this.memory[memoryIndex] = interpolatedValue;
    return interpolatedValue;
  }
}

const startup = async () => {
  const root = await tgpu.init();
  const device = root.device;

  const canvas = document.querySelector("canvas") as HTMLCanvasElement;
  // const context = canvas.getContext("webgpu") as GPUCanvasContext;
  const devicePixelRatio = window.devicePixelRatio;
  canvas.width = canvas.clientWidth * devicePixelRatio;
  canvas.height = canvas.clientHeight * devicePixelRatio;
  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  // context.configure({
  //   device,
  //   format: presentationFormat,
  // });

  let ctx = canvas.getContext("2d");

  const GRID_SIZE = 4;
  const RESOLUTION = 512;
  const COLOR_SCALE = 250;
  const perlin = new PerlinNoise(GRID_SIZE, RESOLUTION);

  const pixelSize = canvas.width / RESOLUTION;

  for (let y = 0; y < RESOLUTION; y += 1) {
    for (let x = 0; x < RESOLUTION; x += 1) {
      ctx.fillStyle = `hsl(${
        perlin.get([
          (x * GRID_SIZE) / RESOLUTION,
          (y * GRID_SIZE) / RESOLUTION,
        ]) * COLOR_SCALE
      },50%,50%)`;
      ctx.fillRect(x * pixelSize, y * pixelSize, pixelSize, pixelSize);
    }
  }
};

window.onload = startup;
