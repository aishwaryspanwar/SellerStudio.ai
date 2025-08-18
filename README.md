# SellerStudio.ai

**SellerStudio.ai** lets you instantly visualize your product worn by AI-generated fashion models — perfect for building clean, modern, high-conversion catalogs. Upload a garment image, and our AI will handle everything: automatic tagging, category selection, model generation, and realistic try-on rendering.

## 🧠 AI Models Used

| Step                     | Model Used                | Purpose                                       |
| ------------------------ | ------------------------- | --------------------------------------------- |
| Product Tagging          | `gemini-1.5-flash`        | Extract comma-separated tags from image       |
| Garment Categorization   | `gemini-1.5-flash`        | Predict single category slug                  |
| Model Preview Generation | `imagen-3.0-generate-002` | Generate AI model poses with plain clothing   |
| Try-On Compositing       | `gemini-2.0-flash-exp`    | Realistically drape garment onto chosen model |

---

## 🚀 How It Works

1. **Upload a Product Image**

   - We extract the garment and describe it using Gemini Flash.

2. **Classify the Product**

   - Automatically assigned a category like `dresses` or `footwear`.

3. **Generate Model Previews**

   - Imagen 3 creates a model gallery with neutral poses and consistent lighting.

4. **Select a Model Template**

   - You pick the best preview — front or angled.

5. **Composite the Final Output**
   - Gemini Flash (2.0) composites the garment onto the selected model, preserving design fidelity and background realism.

---

## ✅ Development Checklist

- [x] **Repo & Environment**
  - GitHub repo, Python venv, dependencies
- [x] **AI Classification Pipeline**
  - Tagging + category classification with Gemini 1.5
- [x] **Model Gallery Generator**
  - Imagen 3 integration with pose prompts
- [x] **User Interface**
  - Streamlit web UI for upload, preview & try-on
- [x] **Try-On Compositing**
  - Gemini 2.0-based photorealistic render with full garment fidelity
- [ ] **Optimization & Testing**
  - Image cleanup, lighting consistency
  - Unit & integration tests
- [ ] **Deployment & Docs**
  - Docker container, CI/CD
  - Finalized documentation & API guide
  - GitHub release
