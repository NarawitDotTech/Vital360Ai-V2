import { describe, expect, it } from "vitest";
import { dictionaries, format, getDict, isLocale } from "./index";

function flatten(obj: unknown, prefix = ""): Record<string, string> {
  const out: Record<string, string> = {};
  if (Array.isArray(obj)) {
    obj.forEach((v, i) => Object.assign(out, flatten(v, `${prefix}[${i}]`)));
  } else if (obj && typeof obj === "object") {
    for (const [k, v] of Object.entries(obj))
      Object.assign(out, flatten(v, prefix ? `${prefix}.${k}` : k));
  } else {
    out[prefix] = String(obj);
  }
  return out;
}

describe("i18n", () => {
  const th = flatten(dictionaries.th);
  const en = flatten(dictionaries.en);

  it("th and en have identical keys", () => {
    expect(Object.keys(en).sort()).toEqual(Object.keys(th).sort());
  });

  it("has no empty strings", () => {
    for (const [k, v] of [...Object.entries(th), ...Object.entries(en)])
      expect(v.trim(), k).not.toBe("");
  });

  it("uses the same placeholders in both languages", () => {
    const ph = (s: string) => (s.match(/\{\w+\}/g) ?? []).sort().join(",");
    for (const k of Object.keys(th)) expect(ph(en[k]), k).toBe(ph(th[k]));
  });

  it("Thai strings contain Thai text (except brand and technical terms)", () => {
    const thai = /[฀-๿]/;
    const allowed = new Set([
      "meta.appName",
      "header.switchLabel",
      "header.switchText",
      "root.linkEn",
    ]);
    for (const [k, v] of Object.entries(th)) {
      if (allowed.has(k) || k.startsWith("dashboard.")) continue;
      expect(thai.test(v), k).toBe(true);
    }
  });

  it("format fills placeholders and getDict falls back to Thai", () => {
    expect(format("เหลือ {n} วินาที", { n: 5 })).toBe("เหลือ 5 วินาที");
    expect(format("{x}", {})).toBe("{x}");
    expect(getDict("zz")).toBe(dictionaries.th);
    expect(isLocale("en")).toBe(true);
    expect(isLocale("fr")).toBe(false);
  });
});
