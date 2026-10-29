# eval/metrics

Every `*.json` file here is rendered by the dashboard (`/th/dashboard/`, `/en/dashboard/`) at build time.
Add a file per evaluation run. The dashboard shows a big "SAMPLE DATA, not measured" banner for any file with `"isSample": true`.
`sample.json` is **invented** and exists only so the page can be built and tested. Never cite it.

## Schema (version 1)

```jsonc
{
  "schemaVersion": 1,
  "isSample": false,                    // true => red "SAMPLE DATA" banner
  "generatedAt": "2026-08-01T00:00:00Z",// ISO 8601
  "description": "free text, shown under the title",
  "models": {                           // any subset of: skin, audio, resp, fusion
    "skin": {
      "label": "Skin image",
      "classes": ["green", "amber", "red"],        // order = rows/cols of confusionMatrix
      "accuracy": 0.0,                              // 0..1
      "sensitivity": 0.0,                           // macro average, 0..1
      "specificity": 0.0,                           // macro average, 0..1
      "f1": 0.0,                                    // macro average, 0..1
      "perClass": { "green": { "sensitivity": 0.0, "specificity": 0.0 } },
      "confusionMatrix": [[0]],                     // rows = actual, cols = predicted, counts
      "latencyMs": 0,                               // median on-device latency
      "sizeBytes": 0,                               // model file size
      "testSetDescription": "what it was measured on, same test set for all models"
    }
  }
}
```

`fusion` must be measured on the same test set as the unimodal models for the comparison to be valid.
