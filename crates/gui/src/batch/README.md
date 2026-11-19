# Batch background removal

This folder contains the batch feature's queue, secondary-window UI, and per-file processing logic. It is deliberately
kept separate so it can be branched, reviewed, or updated with minimal conflicts when the rest of `rembg-gui` changes.

## Files

- `window.rs`: batch window, file/folder selection and drag/drop, selectable queue, asynchronous before/after preview, options, and progress display.
- `processor.rs`: sequential processing, shared output-path planning, skip-existing behavior, and cancellation on the existing worker thread.
- `mod.rs`: job/event types shared with the GUI worker.

## Small integration surface outside this folder

1. `crates/gui/src/lib.rs` declares `mod batch`, stores `BatchWindow`, adds the **Batch…** button, routes batch/download events, and opens the secondary viewport with drag-and-drop enabled.
2. `crates/gui/src/worker.rs` adds one batch command/event and invokes `batch::processor::process` on the existing inference thread.

No new crate dependencies are required. The processor intentionally uses the existing `Engine` instance rather than
creating another one, so batch processing does not spin up a second ONNX Runtime engine or GPU context. The worker
runs one batch sequentially; the rest of the UI remains responsive, while single-image inference commands wait in
the worker queue until the batch completes or is cancelled.

## Behavior

- Supported extensions match the GUI's currently enabled image formats: PNG, JPEG, WebP, BMP, TIFF, and GIF.
- Folder recursion is optional and off by default.
- Image files and folders can be dropped into the batch window or added using the file dialogs.
- Selecting a queue entry loads its source and any existing/generated output in a separate preview-loading thread.
- Output paths are planned by one shared helper, so previews and processing use the same suffix and duplicate-name rules.
- Outputs are transparent PNG files named `<stem>_nobg.png` by default.
- **Skip files that already exist** is checked by default; unchecking it enables overwriting.
- Duplicate output names within a batch receive a numeric disambiguator.
- Cancellation stops before the next image; inference already in progress is allowed to finish and save.
- A model that is not installed is downloaded before the batch starts.
