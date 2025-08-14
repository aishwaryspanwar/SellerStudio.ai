# app.py
import os
import streamlit as st
from dotenv import load_dotenv
from modules.api_handler import (
    TEMP_DIR,
    classify_product,
    infer_category,
    is_tryon_supported,
    generate_studio_models,
    run_tryon_with_selected_model,
)

load_dotenv()
st.set_page_config(page_title="SellerStudio.ai")
st.title("SellerStudio.ai")

if "base_models" not in st.session_state:
    st.session_state.base_models = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None
if "final_image" not in st.session_state:
    st.session_state.final_image = None
if "product_tags" not in st.session_state:
    st.session_state.product_tags = []
if "detected_category" not in st.session_state:
    st.session_state.detected_category = "upper_body"

up = st.file_uploader("Upload a product image (PNG/JPG)", type=["png", "jpg", "jpeg"])

if up:
    os.makedirs(TEMP_DIR, exist_ok=True)
    product_img = os.path.abspath(os.path.join(TEMP_DIR, up.name))
    with open(product_img, "wb") as f:
        f.write(up.getbuffer())
    st.image(product_img, caption="Your Uploaded Product", width=260)

    with st.spinner("Analyzing product..."):
        tags = classify_product(product_img)
        st.session_state.product_tags = tags
        st.write("Detected Product Type:", ", ".join(tags[:8]) if tags else "No tags detected")

    detected = infer_category(tags) if tags else "upper_body"
    st.session_state.detected_category = detected

    st.info("Category auto-selected from classification. You may override if it looks off.")
    cats = ["upper_body", "lower_body", "dresses", "footwear", "headwear"]
    if detected not in cats:
        detected = "upper_body"
    chosen = st.selectbox("Garment category:", options=cats, index=cats.index(detected))

    if chosen in {"footwear", "headwear"}:
        st.warning("Virtual try-on is unavailable for footwear/headwear. You can still generate previews.")

    if st.button("Generate Model Previews", type="primary"):
        with st.spinner("Generating previews..."):
            st.session_state.base_models = generate_studio_models(
                st.session_state.product_tags,
                num_options=3,
                forced_category=chosen
            )
            st.session_state.selected_model = None
            st.session_state.final_image = None

    if st.session_state.base_models:
        st.subheader("Choose a Base Model")
        cols = st.columns(3)
        for i, p in enumerate(st.session_state.base_models):
            with cols[i]:
                st.image(p, use_container_width=True)
                if st.button(f"Select Model {i+1}", key=f"select_{i}"):
                    st.session_state.selected_model = os.path.abspath(p)
                    st.success(f"Selected Model {i+1}")

    if st.session_state.selected_model:
        st.write("---")
        st.subheader("Selected Base Model")
        st.image(st.session_state.selected_model, width=320)

        if is_tryon_supported(chosen):
            if st.button("Generate Final Try-On"):
                with st.spinner("Applying garment..."):
                    out = run_tryon_with_selected_model(
                        selected_model_img_path=st.session_state.selected_model,
                        product_image_path=product_img,
                        product_tags=st.session_state.product_tags,
                        forced_category=chosen,
                        steps=40
                    )
                    if out:
                        st.session_state.final_image = out
                    else:
                        st.error("Failed to generate the final try-on. Please try again.")
        else:
            st.warning(f"Try-on isn’t supported for {chosen}. Pick upper_body, lower_body, or dresses.")

    if st.session_state.final_image:
        st.write("---")
        st.header("Final Image")
        st.image(st.session_state.final_image, width=420)
        st.success("Done!")
else:
    st.caption("Upload a clear, front-facing product photo against a clean background for best results.")
