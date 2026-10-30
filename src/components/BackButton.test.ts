import { describe, expect, it } from "vitest";
import { parentPath } from "./BackButton";

describe("parentPath", () => {
  it("has no back on home", () => {
    expect(parentPath("th", "/th/")).toBeNull();
    expect(parentPath("th", "/th")).toBeNull();
  });
  it("capture screens and results go to the capture hub", () => {
    expect(parentPath("th", "/th/capture/skin/")).toBe("/th/capture/");
    expect(parentPath("en", "/en/results/")).toBe("/en/capture/");
  });
  it("other pages go home", () => {
    expect(parentPath("en", "/en/capture/")).toBe("/en/");
    expect(parentPath("th", "/th/history/")).toBe("/th/");
  });
});
