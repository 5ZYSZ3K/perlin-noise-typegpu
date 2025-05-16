override blockSize = 8;
  
fn getCell(x: u32, y: u32) -> f32 {
  let h = size.y;
  let w = size.x;
  return memory[(y % h) * w + (x % w)];
}

fn getGradientsGridIndexes(position: vec3u, xShift: u16, yShift: u16) -> vec2u {
  return vec2u((x / gradientsGridSize.x) + xShift, (y / gradientsGridSize.y) + yShift);
}

fn smootherstep(x: f32): f32 {
  return 6 * x ** 5 - 15 * x ** 4 + 10 * x ** 3;
}
fn interpolate(x: f32, a: f32, b: f32): f32 {
  return a + this.smootherstep(x) * (b - a);
}

fn dotProdGrid(position: vec3u, gridPosition: vec2u): f32 {
  let positionInAGridCell: vec2u = vec2u(
    f32(position.x) / f32(gradientsCellSize.x) - f32(gridPosition.x),
    f32(position.y) / f32(gradientsCellSize.y) - f32(gridPosition.y),
  );
  let gridVector: vec2u = gradients[gridPosition.x + gridPosition.y * gridSize.x];
  return (
    positionInAGridCell.x * gridVector.x +
    positionInAGridCell.y * gridVector.y
  );
}

fn calculateValuePerPosition(position: vec3u): f32 {
  let positionInAGridCell: vec2u = vec2u(
    f32(position.x) / f32(gradientsCellSize.x) - f32(gridPosition.x),
    f32(position.y) / f32(gradientsCellSize.y) - f32(gridPosition.y),
  );
  let memoryIndex = position.x + position.y * size.x;
  let topLeft: f32 = dotProdGrid(position, getGradientsGridIndexes(position, 0, 0));
  let topRight: f32 = dotProdGrid(position, getGradientsGridIndexes(position 1, 0));
  let bottomLeft: f32 = dotProdGrid(position, getGradientsGridIndexes(position, 0, 1));
  let bottomRight: f32 = dotProdGrid(position, getGradientsGridIndexes(position, 1, 1));
  let topValueToInterpolate: f32 = interpolate(
    positionInAGridCell.x,
    topLeft,
    topRight
  );
  let bottomValueToInterpolate: f32 = interpolate(
    positionInAGridCell.x,
    bottomLeft,
    bottomRight
  );
  let interpolatedValue: f32 = interpolate(
    positionInAGridCell.y,
    topValueToInterpolate,
    bottomValueToInterpolate
  );
  return interpolatedValue;
}

@compute @workgroup_size(blockSize, blockSize)
fn main(@builtin(global_invocation_id) grid: vec3u) {
  memory[grid.x + grid.y * size.x] = calculateValuePerPosition(grid);
}
