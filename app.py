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
    classify_category_with_flash,
)

load_dotenv()

# Ensure temp dir exists
os.makedirs(TEMP_DIR, exist_ok=True)

# Streamlit page configuration
st.set_page_config(
    page_title="SellerStudio.ai",
    page_icon="👗",
    layout="wide"
)

st.title("👗 SellerStudio.ai")
st.markdown("**AI-Powered Virtual Try-On for Fashion Products**")

# Initialize session state
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

# File uploader
uploaded_file = st.file_uploader(
    "Upload a product image (PNG/JPG)",
    type=["png", "jpg", "jpeg"],
    help="Upload a clear, front-facing product photo against a clean background for best results."
)

if uploaded_file:
    # Save uploaded file
    product_img_path = os.path.abspath(os.path.join(TEMP_DIR, uploaded_file.name))
    with open(product_img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Display uploaded image
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image(product_img_path, caption="Your Uploaded Product", width=260)

    with col2:
        # 1) Analyze product (tags)
        with st.spinner("🔍 Analyzing product (tags + category)... This may take up to 60 seconds."):
            try:
                tags = classify_product(product_img_path, max_retries=2)
                st.session_state.product_tags = tags or []
                if tags:
                    st.success("✅ Product analysis complete!")
                    with st.expander("🏷️ Detected Product Tags"):
                        st.write(", ".join(tags[:10]))
                else:
                    st.info("ℹ️ Proceeding without tags. Using generic prompts.")
            except Exception as e:
                st.error(f"❌ Error analyzing product: {e}")
                tags = []
                st.session_state.product_tags = []

        # 2) Flash-based category classification (authoritative)
        with st.spinner("🧠 Detecting garment category..."):
            detected_category = classify_category_with_flash(product_img_path) or (infer_category(tags) if tags else "upper_body")
            categories = ["upper_body", "lower_body", "dresses", "footwear", "headwear"]
            if detected_category not in categories:
                detected_category = "upper_body"
            st.session_state.detected_category = detected_category
            st.info(f"🎯 Auto-detected category: **{detected_category.replace('_',' ').title()}**")

        # 3) Auto-generate model previews (no button)
        with st.spinner("🤖 Generating AI model previews..."):
            try:
                tags_to_use = st.session_state.product_tags or ["fashion", "clothing"]
                generated_models = generate_studio_models(
                    tags_to_use,
                    num_options=3,
                    forced_category=detected_category,
                    aspect_ratio="1:1",
                    sample_image_size="1K",  # ignored by the function
                )
                if generated_models:
                    st.session_state.base_models = generated_models
                    st.success(f"✅ Generated {len(generated_models)} model preview(s)!")
                else:
                    st.error("❌ Failed to generate model previews. Please check your Google GenAI API key and try again.")
                    st.info("💡 Make sure your GEMINI_API_KEY is set correctly in your .env file.")
            except Exception as e:
                st.error(f"❌ Error generating previews: {e}")

        # 4) Auto-select first model and run try-on if supported
        if st.session_state.base_models:
            st.session_state.selected_model = os.path.abspath(st.session_state.base_models[0])
            if is_tryon_supported(detected_category):
                with st.spinner("👗 Applying garment to model... This may take a moment."):
                    try:
                        tags_to_use = st.session_state.product_tags or ["clothing"]
                        result = run_tryon_with_selected_model(
                            selected_model_img_path=st.session_state.selected_model,
                            product_image_path=product_img_path,
                            product_tags=tags_to_use,
                            forced_category=detected_category,
                            steps=40
                        )
                        if result and os.path.exists(result):
                            st.session_state.final_image = result
                            st.success("✅ Try-on complete!")
                            st.rerun()
                        else:
                            st.error("❌ Failed to generate the final try-on. Please try again.")
                    except Exception as e:
                        st.error(f"❌ Try-on failed: {e}")
            else:
                st.info("ℹ️ Try-on is unavailable for footwear/headwear. Showing previews only.")

    # Display generated models
    if st.session_state.base_models:
        st.markdown("---")
        st.subheader("🧑‍🦰 Generated Base Models")

        cols = st.columns(min(len(st.session_state.base_models), 3))
        for i, model_path in enumerate(st.session_state.base_models):
            with cols[i % len(cols)]:
                if os.path.exists(model_path):
                    st.image(model_path, use_container_width=True)
                    st.caption(f"Model {i+1}")
                else:
                    st.error(f"Model {i+1} file not found")

    # Display final result
    if st.session_state.final_image and os.path.exists(st.session_state.final_image):
        st.markdown("---")
        st.header("🎉 Final Result")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(st.session_state.final_image, width=420)

        st.success("🎊 **Try-on complete!** Your AI-generated fashion preview is ready.")

        try:
            with open(st.session_state.final_image, "rb") as file:
                st.download_button(
                    label="📥 Download Final Image",
                    data=file.read(),
                    file_name="sellerstudio_tryon_result.png",
                    mime="image/png",
                    use_container_width=True
                )
        except Exception as e:
            st.error(f"Error creating download button: {e}")

