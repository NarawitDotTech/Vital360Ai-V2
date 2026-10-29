import { expect, test } from "@playwright/test";
import { join } from "node:path";

const fixture = join(process.cwd(), "e2e", "fixtures", "skin.jpg");

test("root redirects to /th/ and capture pages require consent", async ({ page }) => {
  await page.goto("/");
  await expect(page).toHaveURL(/\/th\/$/);
  await expect(page.locator("html")).toHaveAttribute("lang", "th");

  await page.goto("/th/capture/skin/");
  await expect(page).toHaveURL(/\/th\/consent\/$/);
  await page.goto("/th/results/");
  await expect(page).toHaveURL(/\/th\/consent\/$/);
});

test("consent -> skin photo upload -> result with disclaimer", async ({ page }) => {
  await page.goto("/th/");
  await expect(page.getByRole("heading", { level: 1 })).toContainText("คัดกรองสุขภาพ");

  await page.getByRole("link", { name: "เริ่มคัดกรอง" }).click();
  await expect(page).toHaveURL(/\/th\/consent\/$/);

  // must tick the box first
  await page.getByRole("button", { name: "ยินยอมและเริ่มใช้งาน" }).click();
  await expect(page.getByText("กรุณาติ๊กช่องยินยอมก่อน")).toBeVisible();
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "ยินยอมและเริ่มใช้งาน" }).click();

  await expect(page).toHaveURL(/\/th\/capture\/$/);
  await page.getByRole("link", { name: /ถ่ายรูปผิวหนัง/ }).click();
  await expect(page).toHaveURL(/\/th\/capture\/skin\/$/);

  await page.getByTestId("skin-file").setInputFiles(fixture);

  await expect(page).toHaveURL(/\/th\/results\/$/);
  await expect(page.getByTestId("disclaimer")).toContainText("ไม่ใช่การวินิจฉัยโรค");
  await expect(page.getByTestId("urgent-guide")).toBeVisible();
  await expect(page.getByTestId("result-skin")).toBeVisible();

  // save to history, then it appears there
  await page.getByRole("button", { name: "บันทึกลงประวัติ" }).click();
  await expect(page.getByRole("button", { name: /บันทึกแล้ว/ })).toBeVisible();
  await page.goto("/th/history/");
  await expect(page.getByRole("button", { name: /ลบรายการ/ })).toHaveCount(1);

  // delete my data wipes it and requires consent again
  await page.goto("/th/delete/");
  await page.getByRole("button", { name: "ลบข้อมูลทั้งหมดในเครื่องนี้" }).click();
  await page.getByRole("button", { name: "ยืนยัน ลบเลย" }).click();
  await expect(page.getByTestId("delete-done")).toBeVisible();
  await page.goto("/th/capture/");
  await expect(page).toHaveURL(/\/th\/consent\/$/);
});

test("English locale works and dashboard shows the sample banner", async ({ page }) => {
  await page.goto("/en/");
  await expect(page.locator("html")).toHaveAttribute("lang", "en");
  await expect(page.getByRole("link", { name: "Start screening" })).toBeVisible();
  await page.goto("/en/dashboard/");
  await expect(page.getByTestId("sample-banner").first()).toContainText("SAMPLE DATA, not measured");
});

test("app shell works offline after first load", async ({ page, context }) => {
  await page.goto("/th/consent/");
  await page.evaluate(async () => {
    await navigator.serviceWorker.ready;
  });
  // wait until the install step has cached the shell (pages + their JS)
  await expect
    .poll(async () => page.evaluate(async () => (await caches.keys()).some((k) => k.startsWith("vitals360-shell-"))))
    .toBe(true);
  await page.reload(); // now controlled by the service worker
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "ยินยอมและเริ่มใช้งาน" }).click();
  await expect(page).toHaveURL(/\/th\/capture\/$/);

  await context.setOffline(true);
  // full page loads of pages never visited online, including the lazy-loaded capture code
  await page.goto("/th/capture/skin/");
  await expect(page.getByRole("button", { name: "เปิดกล้อง" })).toBeVisible();
  await page.goto("/th/capture/resp/");
  await expect(page.getByRole("button", { name: "เปิดกล้อง" })).toBeVisible();
  await page.goto("/en/dashboard/");
  await expect(page.getByTestId("sample-banner").first()).toBeVisible();
  await context.setOffline(false);
});
