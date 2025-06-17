import streamlit as st
import numpy as np
from PIL import Image
from inference import perform_inference
import gc
import subprocess
import ollama



# Set full-width page layout
st.set_page_config(layout="wide")

# Inject CSS to make mobile view clean and compact
st.markdown(
    """
    <style>
    /* Reduce block padding */
    .block-container {
        padding-top: 0.5rem;
        padding-bottom: 0.5rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    /* Tighter heading margins */
    h1, h2, h3, h4, h5, h6 {
        margin-bottom: 0.3rem;
        margin-top: 0.8rem;
    }

    /* Reduce text area spacing */
    textarea {
        min-height: 50px;
        font-size: 16px !important;
    }

    /* Image scaling for mobile */
    img {
        max-width: 100%;
        height: auto;
    }

    /* Reduce file uploader spacing */
    .stFileUploader {
        margin-bottom: 0.5rem;
    }

    /* Metric and class stats font smaller */
    .stMarkdown p {
        font-size: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# Utility Functions
# =========================

def compute_class_distribution(mask):
    """
    Compute the percentage distribution of each class (soil, crop, weed) in the segmentation mask.
    Args:
        mask (np.ndarray): 2D array with class labels.
    Returns:
        dict: Class distribution as percentages.
    """
    class_labels = {
        "soil": 0,
        "crop": 1,
        "weed": 2
    }
    distribution = {}
    total_pixels = 320 * 320  # Assumes mask is 320x320

    for cls, id in class_labels.items():
        pixels = np.sum(mask == id)
        distribution[cls] = round((pixels / total_pixels) * 100, 2)
    
    return distribution

def preprocess_img(img_org):
    """
    Preprocess input image: resize and split into RGB and CIR channels, then concatenate.
    Args:
        img_org (np.ndarray): Input image array with at least 4 channels.
    Returns:
        np.ndarray: Preprocessed image ready for inference.
    """
    test_images = []
    img_rgb = img_org[:, :, :2]
    img_cir = img_org[:, :, 2:]
    img_rgb_pil = Image.fromarray(img_rgb.astype('uint8'))
    img_cir_pil = Image.fromarray(img_cir.astype('uint8'))
    img_rgb_resized = img_rgb_pil.resize((320, 320))
    img_cir_resized = img_cir_pil.resize((320, 320))
    img_rgb_resized_np = np.array(img_rgb_resized)
    img_cir_resized_np = np.array(img_cir_resized)
    img_resized = np.concatenate((img_rgb_resized_np, img_cir_resized_np), axis=2)
    test_images.append(img_resized)
    return np.array(test_images)

def colorize_mask(img):
    """
    Convert a grayscale mask to an RGB image using a fixed color map.
    Args:
        img (np.ndarray): 2D array with class labels.
    Returns:
        PIL.Image: Colorized mask image.
    """
    color_map = {
        0: [0, 0, 0],       # Soil: Black
        1: [0, 255, 0],     # Crop: Green
        2: [255, 255, 0]    # Weed: Yellow
    }
    h, w = img.shape
    rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
    for gray_value, rgb_value in color_map.items():
        rgb_image[img == gray_value] = rgb_value
    return Image.fromarray(rgb_image)

# =========================
# Model Path
# =========================

MODEL_PATH = "MODEL_PATH"

# =========================
# Streamlit App UI
# =========================



st.markdown("""
<div style='background-color: #e0ffe0; padding: 2px 2px; border-radius: 2px; text-align: center;'>
    <span style='font-size: 2rem; font-weight: bold;'>🌾 UAV Field Intelligence Dashboard</span>
</div>
""", unsafe_allow_html=True)


# Initialize class statistics
class_stats = {'soil': 0, 'crop': 0, 'weed': 0}

# Upload section with styling
st.markdown(
    """
    <div style=" padding: 1px; border-radius: 8px;">
    """,
    unsafe_allow_html=True
)
# uploaded = st.file_uploader("📤 Upload UAV image")

# st.markdown("</div>", unsafe_allow_html=True)

holder = st.empty()
bottom_image = holder.file_uploader('📤 Upload UAV image (if you upload an unspported image, a sample image will be used to demonstrate the applications workflow)')

# if uploaded:
if bottom_image is not None:
    # Preprocess and display success message
    # img_org = preprocess_img(np.load(uploaded))
    try:
        img_org = preprocess_img(np.load(bottom_image))
        holder.empty()
    except:
        img_org = preprocess_img(np.load('sample_input.npy'))
    st.success("✅ Image uploaded and preprocessed")

    # Run inference and post-processing
    with st.spinner("Generating and analyzing the segmentation mask..."):
        segmented_mask = perform_inference(model_path=MODEL_PATH, img=img_org)
        
        # if no model is loaded or no image is loaded, use sample output
        sample_output = True

        if len(segmented_mask) == 3:
            segmented_mask = np.argmax(segmented_mask, axis=-1)
            sample_output = False


        if sample_output:
            st.success("✅ Segmentation not performed, using sample output.")
        else:
            st.success("✅ Segmentation completed")


        # Colorize mask and compute stats
        color_mask = colorize_mask(segmented_mask)
        class_stats = compute_class_distribution(segmented_mask)
        gc.collect()

        if sample_output:
            st.markdown("### 🖼️ Segmentation Mask (sample)")
        else:
            st.markdown("### 🖼️ Segmentation Mask")


        st.image(color_mask, caption="Segmentation Mask", width=350)

        st.markdown("### 📊 Class Distribution")
        st.markdown(f"""
        **🟫 Soil:** {class_stats['soil']}%  
        **🌱 Crop:** {class_stats['crop']}%  
        **🌿 Weed:** {class_stats['weed']}%
        """)
        # # Visual divider
        # st.markdown("""<hr style="margin: 2rem 0;">""", unsafe_allow_html=True)

    # First inject CSS to control spacing
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
        }
        h3, h4 {
            margin-bottom: 0.2rem;
            margin-top: 1rem;
        }
        label {
            font-size: 0.9rem;
            margin-bottom: 0.1rem;
        }
        textarea {
            margin-top: 0rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # Now the actual fields (compact layout)
    st.markdown("### 📝 Editable Field Summary")

    # Section 1
    st.markdown("#### 📄 1. Farmer-Sourced Agronomic & Phenotyping Data")
    management_info_default = f"""- Location: Rheinbach, Germany
    - Crop: Sugar beet (Beta vulgaris), variety ‘Samuela’
    - Growth Stage (Duration): One Month
    - Row Spacing: 50 cm; Intra-row Spacing: 20 cm
    - Prior Weed Control Treatment: One post-emergence mechanical control
    - Fertilization Regimen: 103 kg N/ha applied
    """
    management_info = st.text_area("Edit management information", value=management_info_default, height=170)

    # Section 2
    st.markdown("#### 🌱 2. Field-Surveyed Phenotypic Observations")
    field_observation_default = f"""- Crop Developmental Stage: 6–8 leaf stage
    - Soil Condition: Adequate nitrogen levels, well-drained soil
    """
    field_observation = st.text_area("Edit field observations", value=field_observation_default, height=70)

    # Section 3
    st.markdown("#### 🛰️ 3. UAV-Sourced Segmentation-Derived Phenomics Data")
    image_metrics_default = f"""- Ground Coverage Estimates:
        - Soil Coverage: {class_stats['soil']}%
        - Crop Coverage: {class_stats['crop']}%
        - Weed Coverage: {class_stats['weed']}%
    - Weed Spatial Distribution: Patchy and moderately dense, with most weed presence located in the lower diagonal region beneath crop rows, and scattered between-row occurrences.
    - Crop Density Status: Slightly below target
    """
    image_metrics = st.text_area("Edit image-derived metrics", value=image_metrics_default, height=230)


    # 🔎 One Button for All
    if st.button("🔎 Get Recommendations from LLM"):
        with st.spinner("Analyzing with LLaMA 3..."):
            # Merge all sections into one complete prompt
            edited_summary = "\n\n".join([management_info, field_observation, image_metrics])

            response = ollama.chat(
                model="llama3",
                messages=[
                    {"role": "user", "content": edited_summary},
                    {"role": "user", "content": """Based on the current field summary, provide 3 to 10 concise, weed control recommendations as bullet points.
                        - Each bullet point must be 4 to 7 words.
                        - Focus only on weed management (ignore other topics).
                        - Format your output as a Markdown bullet list.
                        - Do not include any explanations, headings, or non-bullet content."""}
                ]
            )

            # Output section with style
            st.markdown(
                """
                <div style='background-color: #e3f2fd; padding: 12px 16px; border-radius: 8px;'>
                    <span style='color: #1565c0; font-size: 1.6rem; font-weight: 600;'>🤖 LLM Recommendations</span>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.markdown(response['message']['content'])            


            subprocess.run(["ollama", "stop", "llama3"])
