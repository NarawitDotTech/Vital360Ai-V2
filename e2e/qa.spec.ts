import { expect, test, type Page } from "@playwright/test";
import { join } from "node:path";

const fixture = join(process.cwd(), "e2e", "fixtures", "skin.jpg");

async function consent(page: Page) {
  await page.goto("/th/consent/");
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "ยินยอมและเริ่มใช้งาน" }).click();
  await expect(page).toHaveURL(/\/th\/capture\/$/);
}

test("history lists entries before the safety boxes, delete keeps focus on the confirmation", async ({ page }) => {
  await consent(page);
  await page.goto("/th/capture/skin/");
  await page.getByTestId("skin-file").setInputFiles(fixture);
  await expect(page).toHaveURL(/\/th\/results\/$/);
  await page.getByRole("button", { name: "บันทึกลงประวัติ" }).click();
  await expect(page.getByRole("button", { name: /บันทึกแล้ว/ })).toBeVisible();

  await page.goto("/th/history/");
  const del = page.getByRole("button", { name: /ลบรายการ/ });
  await expect(del).toHaveCount(1);
  const listY = (await del.boundingBox())!.y;
  const guideY = (await page.getByTestId("urgent-guide").boundingBox())!.y;
  expect(listY).toBeLessThan(guideY);

  await del.click();
  const notice = page.getByRole("status").filter({ hasText: "ลบแล้ว" });
  await expect(notice).toBeFocused();
});

test("denied camera shows a recovery message instead of a dead end", async ({ page }) => {
  await page.addInitScript(() => {
    navigator.mediaDevices.getUserMedia = () => Promise.reject(new DOMException("denied", "NotAllowedError"));
  });
  await consent(page);
  await page.goto("/th/capture/skin/");
  await page.getByRole("button", { name: "เปิดกล้อง" }).click();
  await expect(page.getByRole("alert").filter({ hasText: "เปิดกล้องไม่ได้" })).toBeVisible();
  await expect(page.getByRole("button", { name: /เลือกรูป/ })).toBeEnabled();
});

test("dashboard scrollable tables are keyboard reachable", async ({ page }) => {
  await page.goto("/en/dashboard/");
  const regions = page.locator('[data-slot="table-container"]');
  expect(await regions.count()).toBeGreaterThan(0);
  await expect(regions.first()).toHaveAttribute("tabindex", "0");
});
