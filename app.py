"""
Streamlit Dashboard for Spatial ViT Analysis (Epic 4, Story 4.1).

Interactive web interface to:
- Upload satellite images for classification
- Compare model outputs side-by-side
- Display top-k predictions with confidence scores
- Show attention and occlusion maps
- Display model information (params, runtime, architecture)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from PIL import Image

from eurosat_vit_analysis.inference import (
    EUROSAT_CLASSES,
    InferenceResult,
    load_model_for_inference,
    predict_single_image,
)

# Note: Overlay visualization functionality will be added in future updates
# from eurosat_vit_analysis.spatial_robustness import occlusion_sensitivity
# from eurosat_vit_analysis.vis.attention import (
#     compute_attention_rollout,
#     capture_attention_maps,
#     overlay_heatmap,
# )

# Page configuration
st.set_page_config(
    page_title="Spatial ViT Analysis",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Model configurations
AVAILABLE_MODELS = {
    "vit_base": {
        "display_name": "ViT-Base",
        "checkpoint": "checkpoints/vit_base_best.pt",
        "supports_attention": True,
    },
    "resnet50": {
        "display_name": "ResNet-50",
        "checkpoint": "checkpoints/resnet50_best.pt",
        "supports_attention": False,
    },
    "swin_t": {
        "display_name": "Swin-Tiny",
        "checkpoint": "checkpoints/swin_t_best.pt",
        "supports_attention": True,
    },
}


@st.cache_resource
def load_model(model_name: str, use_checkpoint: bool = True):
    """
    Load and cache a model for inference.

    Args:
        model_name: Model architecture name
        use_checkpoint: Whether to load from checkpoint

    Returns:
        Tuple of (model, model_info)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint_path = None
    if use_checkpoint:
        checkpoint_file = Path(AVAILABLE_MODELS[model_name]["checkpoint"])
        if checkpoint_file.exists():
            checkpoint_path = str(checkpoint_file)

    model, model_info = load_model_for_inference(
        model_name, checkpoint_path=checkpoint_path, device=device
    )

    return model, model_info, device


def format_number(num: int) -> str:
    """Format large numbers with M/K suffixes."""
    if num >= 1_000_000:
        return f"{num / 1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.1f}K"
    return str(num)


def plot_confidence_bar(result: InferenceResult) -> plt.Figure:
    """
    Create a horizontal bar chart of top-k predictions.

    Args:
        result: Inference result with predictions and confidences

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))

    y_pos = np.arange(len(result.predictions))
    ax.barh(y_pos, result.confidences, align="center", color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(result.predictions)
    ax.invert_yaxis()  # Highest confidence at top
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_title("Top Predictions", fontsize=14, fontweight="bold")
    ax.set_xlim([0, 1])

    # Add value labels
    for i, (pred, conf) in enumerate(zip(result.predictions, result.confidences)):
        ax.text(
            conf + 0.02,
            i,
            f"{conf:.2%}",
            va="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    return fig


def plot_all_probabilities(result: InferenceResult) -> plt.Figure:
    """
    Create a bar chart of all class probabilities.

    Args:
        result: Inference result with all probabilities

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x_pos = np.arange(len(EUROSAT_CLASSES))
    colors = [
        "steelblue" if cls in result.predictions else "lightgray"
        for cls in EUROSAT_CLASSES
    ]

    ax.bar(x_pos, result.all_probabilities, color=colors)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(EUROSAT_CLASSES, rotation=45, ha="right")
    ax.set_ylabel("Probability", fontsize=12)
    ax.set_title("All Class Probabilities", fontsize=14, fontweight="bold")
    ax.set_ylim([0, 1])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    return fig


def display_inference_result(
    result: InferenceResult,
    image: Image.Image,
    show_overlay: bool = False,
    overlay_type: str = "none",
):
    """
    Display inference results with model info and visualizations.

    Args:
        result: Inference result to display
        image: Original input image
        show_overlay: Whether to show attention/occlusion overlay
        overlay_type: Type of overlay ("attention", "occlusion", or "none")
    """
    # Model information
    st.markdown(f"**Model:** {result.model_info.display_name}")
    st.markdown(f"**Architecture:** {result.model_info.architecture}")
    st.markdown(f"**Parameters:** {format_number(result.model_info.num_params)}")
    st.markdown(f"**Inference Time:** {result.inference_time_ms:.1f} ms")

    st.markdown("---")

    # Top-k predictions
    st.markdown("### Top Predictions")
    for i, (pred, conf) in enumerate(zip(result.predictions, result.confidences), 1):
        st.markdown(f"**{i}.** {pred}: `{conf:.2%}`")

    st.markdown("---")

    # Confidence bar chart
    st.markdown("### Confidence Chart")
    fig = plot_confidence_bar(result)
    st.pyplot(fig)
    plt.close(fig)

    # All probabilities chart
    with st.expander("View All Class Probabilities"):
        fig = plot_all_probabilities(result)
        st.pyplot(fig)
        plt.close(fig)

    # Overlay visualization
    if show_overlay and overlay_type != "none":
        st.markdown("---")
        st.markdown(f"### {overlay_type.title()} Overlay")
        st.info(f"{overlay_type.title()} visualization coming soon!")


def main():
    """Main Streamlit application."""
    st.title("🛰️ Spatial ViT Analysis Dashboard")
    st.markdown(
        """
        Compare Vision Transformer and CNN models for satellite image
        classification. Upload an image to see model predictions, confidence
        scores, and spatial analysis.
        """
    )

    # Show device info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        st.success(f"⚡ GPU Acceleration Enabled: {gpu_name}")
    else:
        st.info("💻 Running on CPU (inference may take 10-20 seconds)")

    # Sidebar configuration
    st.sidebar.header("Configuration")

    # Mode selection
    mode = st.sidebar.radio(
        "Mode",
        ["Single Model", "Compare Models"],
        help="Single model mode or side-by-side comparison",
    )

    # Model selection
    if mode == "Single Model":
        model_1_name = st.sidebar.selectbox(
            "Select Model",
            list(AVAILABLE_MODELS.keys()),
            format_func=lambda x: AVAILABLE_MODELS[x]["display_name"],
        )
        model_2_name = None
    else:  # Compare Models
        col1, col2 = st.sidebar.columns(2)
        with col1:
            model_1_name = st.selectbox(
                "Model 1",
                list(AVAILABLE_MODELS.keys()),
                format_func=lambda x: AVAILABLE_MODELS[x]["display_name"],
                key="model_1",
            )
        with col2:
            model_2_name = st.selectbox(
                "Model 2",
                list(AVAILABLE_MODELS.keys()),
                index=1 if len(AVAILABLE_MODELS) > 1 else 0,
                format_func=lambda x: AVAILABLE_MODELS[x]["display_name"],
                key="model_2",
            )

    # Overlay toggle
    show_overlay = st.sidebar.checkbox("Show Spatial Analysis", value=False)
    if show_overlay:
        overlay_type = st.sidebar.radio(
            "Overlay Type",
            ["attention", "occlusion"],
            help="Attention: highlight important regions. Occlusion: sensitivity map.",
        )
    else:
        overlay_type = "none"

    # Top-k selection
    top_k = st.sidebar.slider("Top-K Predictions", min_value=1, max_value=10, value=3)

    # Use checkpoint toggle
    use_checkpoint = st.sidebar.checkbox(
        "Use Trained Checkpoints",
        value=True,
        help="Load trained weights. Uncheck to use ImageNet pretrained models.",
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("**About**")
    st.sidebar.markdown(
        "This dashboard demonstrates spatial representation analysis "
        "of Vision Transformers vs CNNs on satellite imagery."
    )

    # Main content area
    st.markdown("---")

    # Image upload
    uploaded_file = st.file_uploader(
        "Upload a satellite image",
        type=["png", "jpg", "jpeg"],
        help="Upload an EuroSAT image or similar satellite imagery",
    )

    # Sample images option
    if uploaded_file is None:
        st.info("👆 Upload an image or use sample images below")

        # Check if sample images exist
        sample_dir = Path("data/samples")
        if sample_dir.exists():
            sample_images = list(sample_dir.glob("*.jpg")) + list(
                sample_dir.glob("*.png")
            )
            if sample_images:
                selected_sample = st.selectbox(
                    "Or select a sample image",
                    ["None"] + [img.name for img in sample_images],
                )
                if selected_sample != "None":
                    uploaded_file = str(sample_dir / selected_sample)

    # Process uploaded image
    if uploaded_file is not None:
        # Load image
        if isinstance(uploaded_file, str):
            image = Image.open(uploaded_file).convert("RGB")
        else:
            image = Image.open(uploaded_file).convert("RGB")

        st.markdown("---")
        st.markdown("### Uploaded Image")

        # Show image in center
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image(image, caption="Input Image", use_container_width=True)

        st.markdown("---")

        # Run inference
        if mode == "Single Model":
            st.markdown("### Model Predictions")

            model_display = AVAILABLE_MODELS[model_1_name]["display_name"]
            with st.spinner(f"Running inference with {model_display}..."):
                try:
                    model, model_info, device = load_model(model_1_name, use_checkpoint)
                    result = predict_single_image(
                        model, image, model_info, device, top_k=top_k
                    )

                    display_inference_result(result, image, show_overlay, overlay_type)

                except Exception as e:
                    st.error(f"Error during inference: {str(e)}")
                    st.exception(e)

        else:  # Compare Models
            st.markdown("### Model Comparison")

            col1, col2 = st.columns(2)

            # Model 1
            with col1:
                st.markdown(f"#### {AVAILABLE_MODELS[model_1_name]['display_name']}")
                with st.spinner("Running inference..."):
                    try:
                        model_1, model_info_1, device = load_model(
                            model_1_name, use_checkpoint
                        )
                        result_1 = predict_single_image(
                            model_1, image, model_info_1, device, top_k=top_k
                        )

                        display_inference_result(
                            result_1, image, show_overlay, overlay_type
                        )

                    except Exception as e:
                        st.error(f"Error with Model 1: {str(e)}")

            # Model 2
            with col2:
                st.markdown(f"#### {AVAILABLE_MODELS[model_2_name]['display_name']}")
                with st.spinner("Running inference..."):
                    try:
                        model_2, model_info_2, device = load_model(
                            model_2_name, use_checkpoint
                        )
                        result_2 = predict_single_image(
                            model_2, image, model_info_2, device, top_k=top_k
                        )

                        display_inference_result(
                            result_2, image, show_overlay, overlay_type
                        )

                    except Exception as e:
                        st.error(f"Error with Model 2: {str(e)}")

            # Comparison summary
            st.markdown("---")
            st.markdown("### Comparison Summary")

            try:
                comparison_col1, comparison_col2, comparison_col3 = st.columns(3)

                with comparison_col1:
                    faster_model = (
                        AVAILABLE_MODELS[model_1_name]["display_name"]
                        if result_1.inference_time_ms < result_2.inference_time_ms
                        else AVAILABLE_MODELS[model_2_name]["display_name"]
                    )
                    faster_time = min(
                        result_1.inference_time_ms, result_2.inference_time_ms
                    )
                    st.metric("Speed Winner", faster_model, f"{faster_time:.1f} ms")

                with comparison_col2:
                    st.metric(
                        "More Confident",
                        AVAILABLE_MODELS[model_1_name]["display_name"]
                        if result_1.confidences[0] > result_2.confidences[0]
                        else AVAILABLE_MODELS[model_2_name]["display_name"],
                        f"{max(result_1.confidences[0], result_2.confidences[0]):.2%}",
                    )

                with comparison_col3:
                    agreement = (
                        "✅ Agree"
                        if result_1.predictions[0] == result_2.predictions[0]
                        else "❌ Disagree"
                    )
                    st.metric(
                        "Top Prediction",
                        agreement,
                        result_1.predictions[0]
                        if result_1.predictions[0] == result_2.predictions[0]
                        else f"{result_1.predictions[0]} vs {result_2.predictions[0]}",
                    )

            except Exception:
                st.warning("Could not generate comparison summary")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Built with ❤️ for Vision Transformer spatial analysis research</p>
            <p><a href="https://github.com/tgrytnes/Spatial-Representation-Analysis-of-Vision-Transformers-for-Satellite-Image-Classification">GitHub</a></p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
