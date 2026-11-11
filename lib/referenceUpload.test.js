import { describe, it, expect } from "vitest";
import { referenceFromFile } from "./referenceUpload.js";

const jpeg = (name, bytes = 10) => new File([new Uint8Array(bytes)], name, { type: "image/jpeg" });

describe("referenceFromFile", () => {
  it("turns an uploaded image into a reference named after the file", () => {
    const ref = referenceFromFile(jpeg("my friend.jpg"));

    expect(ref.name).toBe("my friend");
    expect(ref.url).toMatch(/^blob:/);
  });

  it("rejects files that are not images", () => {
    const pdf = new File(["%PDF"], "cv.pdf", { type: "application/pdf" });

    expect(() => referenceFromFile(pdf)).toThrow("Please choose an image file");
  });

  it("rejects images larger than 10 MB", () => {
    const huge = jpeg("huge.jpg", 10 * 1024 * 1024 + 1);

    expect(() => referenceFromFile(huge)).toThrow("Image is too large (max 10 MB)");
  });

  it("accepts an image of exactly 10 MB", () => {
    expect(() => referenceFromFile(jpeg("edge.jpg", 10 * 1024 * 1024))).not.toThrow();
  });
});
