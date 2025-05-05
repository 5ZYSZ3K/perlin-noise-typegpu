@fragment
fn main(@location(0) cell: f32, @builtin(position) pos: vec4f) -> @location(0) vec4f {
  return vec4f(
    max(f32(cell) * pos.x / 1024, 0), 
    max(f32(cell) * pos.y / 1024, 0), 
    max(f32(cell) * (1 - pos.x / 1024), 0),
    1.
  );
}
