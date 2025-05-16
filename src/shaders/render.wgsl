struct Out {
    @builtin(position) pos: vec4f,
    @location(0) cell: f32,
}
  
@vertex
fn vert(@builtin(instance_index) i: u32, @location(0) cell: f32, @location(1) pos: vec2u) -> Out {
    let w = size.x;
    let h = size.y;
    let x = (f32(i % w + pos.x) / f32(w) - 0.5) * 2. * f32(w) / f32(max(w, h));
    let y = (f32((i - (i % w)) / w + pos.y) / f32(h) - 0.5) * 2. * f32(h) / f32(max(w, h));
  
    return Out(
      vec4f(x, y, 0., 1.),
      f32(cell)
    );
}
  
@fragment
fn frag(@location(0) cell: f32, @builtin(position) pos: vec4f) -> @location(0) vec4f {  
    return vec4f(
      25 - floor(25 * cell),
      floor(255 * cell),
      0,
      1
    );
}