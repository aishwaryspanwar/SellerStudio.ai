# SellerStudio.ai

Upload a product image and let our AI detect and extract the item. Choose from AI-generated mannequin or live-model templates, then get high-resolution mockups of your product worn by a model from multiple angles—ready for your online catalog.

## Development Checklist

- [x] **Repo & environment**

  - Create GitHub repo with README, `.gitignore` (Python) & license
  - Set up Python virtual environment and install core dependencies

- [ ] **Object detection & extraction**

  - Integrate a detection model (e.g. YOLO, Detectron2)
  - Test segmentation masks on sample images

- [ ] **Template generation**

  - Connect to Gemini 2.5 Pro API for mannequin/live-model renders
  - Build a small gallery of AI-generated model poses

- [ ] **User interface**

  - Design CLI or web UI for uploading, previewing & selecting templates
  - Implement file handling and preview thumbnails

- [ ] **Compositing engine**

  - Overlay extracted product onto chosen template
  - Render at least two distinct camera angles

- [ ] **Optimization & testing**

  - Enhance image quality (color match, lighting)
  - Write unit tests for each module
  - End-to-end integration tests

- [ ] **Deployment & docs**
  - Containerize with Docker & define CI/CD pipeline
  - Finalize README with usage examples and API keys guide
  - Publish initial release on GitHub
