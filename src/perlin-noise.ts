"use strict";
import tgpu from "typegpu";
import * as d from "typegpu/data";

let genSizes = [2048, 2048];
const GRID_SIZE = 16;

const computeShaderString = `override blockSize = 16;
  
fn getCell(x: u32, y: u32) -> f32 {
  let h = size.y;
  let w = size.x;
  return memory[(y % h) * w + (x % w)];
}

fn getGradientsGridIndexes(position: vec3u, xShift: u32, yShift: u32) -> vec2u {
  return vec2u((position.x / gradientsCellSize.x) + xShift, (position.y / gradientsCellSize.y) + yShift);
}

fn smootherstep(x: f32) -> f32 {
  return  6 * pow(x, 5) - 15 * pow(x, 4) + 10 * pow(x, 3);
}
fn interpolate(x: f32, a: f32, b: f32) -> f32 {
  return a + smootherstep(x) * (b - a);
}

fn dotProdGrid(position: vec3u, gradientsGridPosition: vec2u) -> f32 {
  let positionInAGridCell: vec2f = vec2f(
    f32(position.x) / f32(gradientsCellSize.x) - f32(gradientsGridPosition.x),
    f32(position.y) / f32(gradientsCellSize.y) - f32(gradientsGridPosition.y),
  );
  let gridVector: vec2f = gradients[gradientsGridPosition.x + gradientsGridPosition.y * (gradientsGridSize.x + 1)];
  return (
    positionInAGridCell.x * gridVector.x +
    positionInAGridCell.y * gridVector.y
  );
}

fn calculateValuePerPosition(position: vec3u, index: u32) -> f32 {
  let gradientsGridPosition: vec2u = getGradientsGridIndexes(position, 0, 0);
  let positionInAGridCell: vec2f = vec2f(
    f32(position.x) / f32(gradientsCellSize.x) - f32(gradientsGridPosition.x), 
    f32(position.y) / f32(gradientsCellSize.y) - f32(gradientsGridPosition.y)
  );
  let topLeft: f32 = dotProdGrid(position, gradientsGridPosition);
  let topRight: f32 = dotProdGrid(position, getGradientsGridIndexes(position, 1, 0));
  let bottomLeft: f32 = dotProdGrid(position, getGradientsGridIndexes(position, 0, 1));
  let bottomRight: f32 = dotProdGrid(position, getGradientsGridIndexes(position, 1, 1));
  let topValueToInterpolate: f32 = interpolate(positionInAGridCell.x, topLeft, topRight);
  let bottomValueToInterpolate: f32 = interpolate(positionInAGridCell.x, bottomLeft, bottomRight);
  let interpolatedValue: f32 = interpolate(positionInAGridCell.y, topValueToInterpolate, bottomValueToInterpolate);
  return interpolatedValue;
}

@compute @workgroup_size(blockSize, blockSize)
fn main(@builtin(global_invocation_id) grid: vec3u) {
  let index: u32 = grid.x + grid.y * size.x;
  let val: f32 = calculateValuePerPosition(grid, index);
  memory[index] = val;
}
`;

const renderShaderString = `struct Out {
    @builtin(position) pos: vec4f,
    @location(0) cell: f32,
    @location(1) uv: vec2f,
}
  
@vertex
fn vert(@builtin(instance_index) i: u32, @location(0) cell: f32, @location(1) pos: vec2u) -> Out {
    let w = size.x;
    let h = size.y;
    let x = (f32(i % w + pos.x) / f32(w) - 0.5) * 2. * f32(w) / f32(max(w, h));
    let y = (f32((i - (i % w)) / w + pos.y) / f32(h) - 0.5) * 2. * f32(h) / f32(max(w, h));
  
    return Out(vec4f(x, y, 0., 1.), cell, vec2f(x,y));
}
  
@fragment
fn frag(@location(0) cell: f32, @builtin(position) pos: vec4f) -> @location(0) vec4f {  
    return vec4f((cell + 1.)/2., 0, 1. - (cell + 1.)/2., 1);
}`;

type Vector2 = { x: number; y: number };

class PerlinNoise {
  gradients: Array<Vector2>;
  memory: Array<number>;
  resolution: Vector2;
  gradientsGridSize: Vector2;
  gradientsCellSize: Vector2;

  constructor(
    gridSize: number,
    resolution: number,
    gradients: Array<Vector2> = []
  ) {
    this.resolution = { x: resolution, y: resolution };
    this.gradientsGridSize = { x: gridSize, y: gridSize };
    this.gradientsCellSize = {
      x: resolution / gridSize,
      y: resolution / gridSize,
    };
    this.gradients = gradients;
    this.memory = [];
  }
  randonAngledVector(): Vector2 {
    const theta = Math.random() * 2 * Math.PI;
    return { x: Math.cos(theta), y: Math.sin(theta) };
  }
  dotProdGrid(position: Vector2, gradientsGridPosition: Vector2) {
    let positionInAGridCell = {
      x: position.x / this.gradientsCellSize.x - gradientsGridPosition.x,
      y: position.y / this.gradientsCellSize.y - gradientsGridPosition.y,
    };
    let gridVector: Vector2;
    const gradientsIndex = Math.round(
      gradientsGridPosition.x +
        gradientsGridPosition.y * (this.gradientsGridSize.x + 1)
    );
    if (this.gradients[gradientsIndex]) {
      gridVector = this.gradients[gradientsIndex];
    } else {
      gridVector = this.randonAngledVector();
      this.gradients[gradientsIndex] = gridVector;
    }
    return (
      positionInAGridCell.x * gridVector.x +
      positionInAGridCell.y * gridVector.y
    );
  }
  smootherstep(x: number) {
    return 6 * x ** 5 - 15 * x ** 4 + 10 * x ** 3;
  }
  interpolate(x: number, a: number, b: number) {
    return a + this.smootherstep(x) * (b - a);
  }
  getGradientsGridIndexes(
    position: Vector2,
    xShift: number,
    yShift: number
  ): Vector2 {
    return {
      x: Math.floor(position.x / this.gradientsCellSize.x) + xShift,
      y: Math.floor(position.y / this.gradientsCellSize.y) + yShift,
    };
  }
  calculateValuePerPosition(position: Vector2) {
    const memoryIndex = Math.round(position.x + position.y * this.resolution.x);
    if (this.memory[memoryIndex]) return this.memory[memoryIndex];
    let gradientsGridPosition = this.getGradientsGridIndexes(position, 0, 0);
    let positionInAGridCell = {
      x: position.x / this.gradientsCellSize.x - gradientsGridPosition.x,
      y: position.y / this.gradientsCellSize.y - gradientsGridPosition.y,
    };
    //interpolate
    const topLeft = this.dotProdGrid(position, gradientsGridPosition);
    const topRight = this.dotProdGrid(
      position,
      this.getGradientsGridIndexes(position, 1, 0)
    ); // positionOnGrid, gridCellIndexes
    const bottomLeft = this.dotProdGrid(
      position,
      this.getGradientsGridIndexes(position, 0, 1)
    ); // positionOnGrid, gridCellIndexes
    const bottomRight = this.dotProdGrid(
      position,
      this.getGradientsGridIndexes(position, 1, 1)
    );
    const topValueToInterpolate = this.interpolate(
      positionInAGridCell.x,
      topLeft,
      topRight
    );
    const bottomValueToInterpolate = this.interpolate(
      positionInAGridCell.x,
      bottomLeft,
      bottomRight
    );
    const interpolatedValue = this.interpolate(
      positionInAGridCell.y,
      topValueToInterpolate,
      bottomValueToInterpolate
    );
    this.memory[memoryIndex] = interpolatedValue;
    return interpolatedValue;
  }
  get(position: Vector2) {
    return this.calculateValuePerPosition(position);
  }
}

const startup = async () => {
  const canvas = document.querySelector("#canvas2") as HTMLCanvasElement;
  let ctx = canvas.getContext("2d");
  const RESOLUTION = genSizes[0];
  const COLOR_SCALE = 255;

  const perlin = new PerlinNoise(GRID_SIZE, RESOLUTION);
  const pixelSize = canvas.width / RESOLUTION;

  for (let y = 0; y < RESOLUTION; y += 1) {
    for (let x = 0; x < RESOLUTION; x += 1) {
      const perlin2 = perlin.get({ x, y });
      ctx.fillStyle = `rgb(${Math.round(((perlin2 + 1) / 2) * COLOR_SCALE)},0,${
        COLOR_SCALE - Math.round(((perlin2 + 1) / 2) * COLOR_SCALE)
      })`;
      ctx.fillRect(x * pixelSize, y * pixelSize, pixelSize, pixelSize);
    }
  }
};

const startupTGPU = async () => {
  const root = await tgpu.init();
  const device = root.device;

  const canvas = document.querySelector("canvas") as HTMLCanvasElement;
  const context = canvas.getContext("webgpu") as GPUCanvasContext;

  const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format: presentationFormat,
    alphaMode: "premultiplied",
  });

  let workgroupSize = 16;

  const bindGroupLayoutCompute = tgpu.bindGroupLayout({
    memory: {
      storage: (arrayLength: number) => d.arrayOf(d.f32, arrayLength),
      access: "mutable",
    },
    gradients: {
      storage: (arrayLength: number) => d.arrayOf(d.vec2f, arrayLength),
      access: "readonly",
    },
    size: {
      storage: d.vec2u,
      access: "readonly",
    },
    gradientsGridSize: {
      storage: d.vec2u,
      access: "readonly",
    },
    gradientsCellSize: {
      storage: d.vec2u,
      access: "readonly",
    },
  });
  const bindGroupLayoutRender = tgpu.bindGroupLayout({
    size: {
      uniform: d.vec2u,
    },
  });

  const computeShader = device.createShaderModule({
    code: tgpu.resolve({
      template: computeShaderString,
      externals: {
        ...bindGroupLayoutCompute.bound,
      },
    }),
  });

  const renderShader = device.createShaderModule({
    code: tgpu.resolve({
      template: renderShaderString,
      externals: {
        ...bindGroupLayoutRender.bound,
      },
    }),
  });

  const computePipeline = device.createComputePipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [root.unwrap(bindGroupLayoutCompute)],
    }),
    compute: {
      module: computeShader,
      constants: {
        blockSize: workgroupSize,
      },
    },
  });

  const squareVertexLayout = tgpu.vertexLayout(
    (n: number) => d.arrayOf(d.location(1, d.vec2u), n),
    "vertex"
  );

  const memoryVertexLayout = tgpu.vertexLayout(
    (n: number) => d.arrayOf(d.location(0, d.f32), n),
    "instance"
  );

  const memoryLength = genSizes[0] * genSizes[1];
  const memory = Array.from({ length: memoryLength }).fill(0) as Array<number>;
  const gradientsGridSizes = [GRID_SIZE + 1, GRID_SIZE + 1];

  const gradientsLength = gradientsGridSizes[0] * gradientsGridSizes[1];
  const gradients = Array.from({ length: gradientsLength })
    .fill(0)
    .map(() => {
      const theta = Math.random() * 2 * Math.PI;
      return d.vec2f(Math.cos(theta), Math.sin(theta));
    });

  const memoryBuffer = root
    .createBuffer(d.arrayOf(d.f32, memoryLength), memory)
    .$usage("storage", "vertex");

  const gradientsBuffer = root
    .createBuffer(d.arrayOf(d.vec2f, gradientsLength), gradients)
    .$usage("uniform", "storage");

  const sizeBuffer = root
    .createBuffer(d.vec2u, d.vec2u(genSizes[0], genSizes[1]))
    .$usage("uniform", "storage");

  const squareBuffer = root
    .createBuffer(d.arrayOf(d.u32, 8), [0, 0, 1, 0, 0, 1, 1, 1])
    .$usage("vertex");

  const gradientsGridSizeBuffer = root
    .createBuffer(
      d.vec2u,
      d.vec2u(gradientsGridSizes[0] - 1, gradientsGridSizes[1] - 1)
    )
    .$usage("uniform", "storage");

  const gradientsCellSizeBuffer = root
    .createBuffer(
      d.vec2u,
      d.vec2u(
        Math.round(genSizes[0] / (gradientsGridSizes[0] - 1)),
        Math.round(genSizes[1] / (gradientsGridSizes[1] - 1))
      )
    )
    .$usage("uniform", "storage");

  const bindGroup = root.createBindGroup(bindGroupLayoutCompute, {
    size: sizeBuffer,
    memory: memoryBuffer,
    gradients: gradientsBuffer,
    gradientsGridSize: gradientsGridSizeBuffer,
    gradientsCellSize: gradientsCellSizeBuffer,
  });

  const uniformBindGroup = root.createBindGroup(bindGroupLayoutRender, {
    size: sizeBuffer,
  });

  const view = context.getCurrentTexture().createView();
  const renderPass: GPURenderPassDescriptor = {
    colorAttachments: [
      {
        view,
        loadOp: "clear",
        storeOp: "store",
      },
    ],
  };

  const commandEncoder = device.createCommandEncoder();
  const passEncoderCompute = commandEncoder.beginComputePass();

  passEncoderCompute.setPipeline(computePipeline);
  passEncoderCompute.setBindGroup(0, root.unwrap(bindGroup));

  passEncoderCompute.dispatchWorkgroups(
    genSizes[0] / workgroupSize,
    genSizes[1] / workgroupSize
  );
  passEncoderCompute.end();

  const renderPipeline = device.createRenderPipeline({
    layout: device.createPipelineLayout({
      bindGroupLayouts: [root.unwrap(bindGroupLayoutRender)],
    }),
    primitive: {
      topology: "triangle-strip",
    },
    vertex: {
      module: renderShader,
      buffers: [
        root.unwrap(memoryVertexLayout),
        root.unwrap(squareVertexLayout),
      ],
    },
    fragment: {
      module: renderShader,
      targets: [
        {
          format: presentationFormat,
        },
      ],
    },
  });

  const passEncoderRender = commandEncoder.beginRenderPass(renderPass);
  passEncoderRender.setPipeline(renderPipeline);

  passEncoderRender.setVertexBuffer(0, root.unwrap(memoryBuffer));
  passEncoderRender.setVertexBuffer(1, root.unwrap(squareBuffer));
  passEncoderRender.setBindGroup(0, root.unwrap(uniformBindGroup));

  passEncoderRender.draw(4, memoryLength);
  passEncoderRender.end();
  device.queue.submit([commandEncoder.finish()]);
};

window.onload = async () => {
  const time1 = Date.now();
  await startupTGPU();
  const time2 = Date.now();
  startup();
  const time3 = Date.now();
  console.log("GPU time:", time2 - time1, "JS time:", time3 - time2);
};
