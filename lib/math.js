export const lerp = (a, b, t) => a + (b - a) * t;
export const dist2D = (ax, ay, bx, by) => Math.sqrt((bx - ax) ** 2 + (by - ay) ** 2);
