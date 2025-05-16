"use strict";
import tgpu from "typegpu";
import * as d from "typegpu/data";
import computeShaderString from "./shaders/compute.wgsl";
import renderShaderString from "./shaders/render.wgsl";

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
  const canvas = document.querySelector("canvas") as HTMLCanvasElement;
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
  let genSizes = [8192, 8192];
  const gradientsGridSizes = [64, 64];

  const computeLayout = {
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
    gridSize: {
      storage: d.vec2u,
      access: "readonly",
    },
  } as const;

  const groupLayout = {
    size: {
      uniform: d.vec2u,
    },
  } as const;

  const bindGroupLayoutCompute = tgpu.bindGroupLayout(computeLayout);
  const bindGroupLayoutRender = tgpu.bindGroupLayout(groupLayout);

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
  console.log(memoryLength);
  const memory = Array.from({ length: memoryLength }).fill(0) as Array<number>;

  const gradientsLength = gradientsGridSizes[0] * gradientsGridSizes[1];
  const gradients = Array.from({ length: gradientsLength })
    .fill(0)
    .map(() => d.vec2f(Math.random(), Math.random()));

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
      d.vec2u(gradientsGridSizes[0], gradientsGridSizes[1])
    )
    .$usage("uniform", "storage");

  const gradientsCellSizeBuffer = root
    .createBuffer(
      d.vec2u,
      d.vec2u(
        Math.round(genSizes[0] / gradientsGridSizes[0]),
        Math.round(genSizes[1] / gradientsGridSizes[1])
      )
    )
    .$usage("uniform", "storage");

  const bindGroup = root.createBindGroup(bindGroupLayoutCompute, {
    size: sizeBuffer,
    memory: memoryBuffer,
    gradients: gradientsBuffer,
    gradientsGridSize: gradientsGridSizeBuffer,
    gridSize: gradientsCellSizeBuffer,
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

  passEncoderRender.draw(4, length);
  passEncoderRender.end();
  device.queue.submit([commandEncoder.finish()]);
};

window.onload = startupTGPU;
