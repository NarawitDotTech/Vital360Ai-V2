import { expect, test } from "@playwright/test";

test("AI summary on results and hub after skin + resp", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto("/th/consent/");
  await page.getByRole("checkbox").check();
  await page.getByRole("button", { name: "ยินยอมและเริ่มใช้งาน" }).click();
  await expect(page).toHaveURL(/\/th\/capture\/$/);
  const mk = (
    modality: string,
    tier: string,
    label: string,
    extras?: object,
  ) => ({
    modality,
    tier,
    label,
    confidence: 0.8,
    probs: { [label]: 0.8 },
    modelVersion: "real-1",
    latencyMs: 1,
    ...(extras ? { extras } : {}),
  });
  await page.evaluate(
    (r) => sessionStorage.setItem("v360.session", JSON.stringify(r)),
    [
      mk("skin", "amber", "melanoma"),
      mk("resp", "green", "normal", { breathsPerMin: 16 }),
    ],
  );
  await page.goto("/th/capture/");
  await expect(page.getByTestId("see-summary")).toBeVisible();
  await page.getByTestId("see-summary").click();
  await expect(page).toHaveURL(/\/th\/results\/$/);
  const card = page.getByTestId("ai-summary");
  await expect(card).toContainText("สรุปผลโดย AI");
  await expect(page.getByTestId("ai-summary-headline")).toContainText(
    "ควรให้แพทย์ช่วยดู",
  );
  await expect(card).toContainText("ยังไม่ได้คัดกรอง");
  await expect(page.getByTestId("disclaimer")).toBeVisible();
  await page.waitForTimeout(800);
  await page.screenshot({
    path: "/tmp/claude-0/-home-user-Vital360Ai-V2/8c64c4d0-7cfd-5965-89a0-1522b0ac8b05/scratchpad/summary.png",
  });
});
