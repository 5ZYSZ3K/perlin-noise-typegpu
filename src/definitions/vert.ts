/* generated via tgpu-gen by TypeGPU */
import tgpu from 'typegpu';
import * as d from 'typegpu/data';

/* structs */
export const Out = d.struct({
  pos: d.vec4f,
  cell: d.f32,
});

/* bindGroupLayouts */
export const layout0 = tgpu.bindGroupLayout({
  size: {
    uniform: d.vec2u,
  },
});
