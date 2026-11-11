// Turns a user-picked File into a reference entry ({ name, url }) for face swap.
// The image stays in the browser — the url is a local object URL.
const MAX_BYTES = 10 * 1024 * 1024;

export function referenceFromFile(file) {
  if (!file.type.startsWith("image/")) throw new Error("Please choose an image file");
  if (file.size > MAX_BYTES) throw new Error("Image is too large (max 10 MB)");
  return {
    name: file.name.replace(/\.[^.]+$/, ""),
    url: URL.createObjectURL(file),
  };
}
