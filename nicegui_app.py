"""
NiceGUI Dashboard for Spatial ViT Analysis (Epic 4, Story 4.1).

Interactive web interface using NiceGUI that connects to RunPod API for inference.
"""

import asyncio
import io

import httpx
import matplotlib.pyplot as plt
import numpy as np
from nicegui import ui

# Configuration
API_URL = "http://localhost:8000"  # Will be replaced with RunPod URL
EUROSAT_CLASSES = [
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
]

# Global state
state = {
    "api_url": API_URL,
    "mode": "single",
    "model1": "vit_base",
    "model2": "resnet50",
    "top_k": 3,
    "current_image": None,
}


def create_confidence_chart(predictions: list[str], confidences: list[float]) -> str:
    """Create matplotlib confidence bar chart and return as base64."""
    fig, ax = plt.subplots(figsize=(8, 4))

    y_pos = np.arange(len(predictions))
    ax.barh(y_pos, confidences, align="center", color="steelblue")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(predictions)
    ax.invert_yaxis()
    ax.set_xlabel("Confidence", fontsize=12)
    ax.set_title("Top Predictions", fontsize=14, fontweight="bold")
    ax.set_xlim([0, 1])

    # Add value labels
    for i, conf in enumerate(confidences):
        ax.text(
            conf + 0.02, i, f"{conf:.2%}", va="center", fontsize=10, fontweight="bold"
        )

    plt.tight_layout()

    # Save to bytes
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)

    # Convert to base64
    import base64

    img_base64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{img_base64}"


async def call_inference_api(
    image_bytes: bytes, model_name: str, top_k: int = 3
) -> dict:
    """Call the RunPod inference API."""
    async with httpx.AsyncClient(timeout=60.0) as client:
        files = {"file": ("image.jpg", image_bytes, "image/jpeg")}
        data = {"model_name": model_name, "top_k": top_k}

        response = await client.post(
            f"{state['api_url']}/predict",
            files=files,
            data=data,
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"API Error: {response.status_code} - {response.text}")


def create_result_card(result: dict, model_name: str) -> ui.card:
    """Create a result display card."""
    with ui.card().classes("w-full"):
        ui.label(result["model_info"]["display_name"]).classes("text-2xl font-bold")

        # Model info
        with ui.row().classes("gap-4"):
            ui.label(f"Architecture: {result['model_info']['architecture']}")
            ui.label(f"Parameters: {result['model_info']['num_params']:,}")
            ui.label(f"Inference: {result['inference_time_ms']:.1f} ms").classes(
                "text-green-600 font-bold"
            )

        ui.separator()

        # Top predictions
        ui.label("Top Predictions").classes("text-xl font-bold mt-4")
        for i, (pred, conf) in enumerate(
            zip(result["predictions"], result["confidences"]), 1
        ):
            with ui.row().classes("items-center gap-2"):
                ui.label(f"{i}.").classes("font-bold")
                ui.label(pred).classes("flex-grow")
                ui.badge(f"{conf:.2%}").classes("bg-blue-500")

        # Confidence chart
        ui.label("Confidence Chart").classes("text-xl font-bold mt-4")
        chart_img = create_confidence_chart(
            result["predictions"], result["confidences"]
        )
        ui.html(f'<img src="{chart_img}" style="max-width: 100%; height: auto;"/>')

    return ui.card


async def process_image(image_bytes: bytes, container: ui.column):
    """Process uploaded image with selected model(s)."""
    container.clear()

    with container:
        if state["mode"] == "single":
            # Single model inference
            ui.label("Processing...").classes("text-xl")
            with ui.spinner(size="lg"):
                try:
                    result = await call_inference_api(
                        image_bytes, state["model1"], state["top_k"]
                    )
                    container.clear()
                    with container:
                        ui.label("Results").classes("text-2xl font-bold mb-4")
                        create_result_card(result, state["model1"])
                except Exception as e:
                    container.clear()
                    with container:
                        ui.label(f"Error: {str(e)}").classes("text-red-500 text-xl")

        else:
            # Compare mode - two models
            ui.label("Processing both models...").classes("text-xl")

            try:
                # Run both inferences in parallel
                results = await asyncio.gather(
                    call_inference_api(image_bytes, state["model1"], state["top_k"]),
                    call_inference_api(image_bytes, state["model2"], state["top_k"]),
                )

                container.clear()
                with container:
                    ui.label("Model Comparison").classes("text-2xl font-bold mb-4")

                    with ui.row().classes("w-full gap-4"):
                        with ui.column().classes("flex-1"):
                            create_result_card(results[0], state["model1"])

                        with ui.column().classes("flex-1"):
                            create_result_card(results[1], state["model2"])

                    # Comparison summary
                    ui.separator().classes("my-4")
                    ui.label("Comparison Summary").classes("text-2xl font-bold")

                    with ui.row().classes("gap-8"):
                        # Speed winner
                        faster_idx = (
                            0
                            if results[0]["inference_time_ms"]
                            < results[1]["inference_time_ms"]
                            else 1
                        )
                        with ui.card().classes("p-4"):
                            ui.label("⚡ Speed Winner").classes("text-lg font-bold")
                            ui.label(
                                results[faster_idx]["model_info"]["display_name"]
                            ).classes("text-xl text-green-600")
                            ui.label(
                                f"{results[faster_idx]['inference_time_ms']:.1f} ms"
                            )

                        # More confident
                        confident_idx = (
                            0
                            if results[0]["confidences"][0]
                            > results[1]["confidences"][0]
                            else 1
                        )
                        with ui.card().classes("p-4"):
                            ui.label("🎯 More Confident").classes("text-lg font-bold")
                            ui.label(
                                results[confident_idx]["model_info"]["display_name"]
                            ).classes("text-xl text-blue-600")
                            ui.label(f"{results[confident_idx]['confidences'][0]:.2%}")

                        # Agreement
                        agree = (
                            results[0]["predictions"][0] == results[1]["predictions"][0]
                        )
                        with ui.card().classes("p-4"):
                            ui.label("📊 Top Prediction").classes("text-lg font-bold")
                            if agree:
                                ui.label("✅ Agree").classes("text-xl text-green-600")
                                ui.label(results[0]["predictions"][0])
                            else:
                                ui.label("❌ Disagree").classes("text-xl text-red-600")
                                pred1 = results[0]["predictions"][0]
                                pred2 = results[1]["predictions"][0]
                                ui.label(f"{pred1} vs {pred2}")

            except Exception as e:
                container.clear()
                with container:
                    ui.label(f"Error: {str(e)}").classes("text-red-500 text-xl")


async def handle_upload(e, result_container: ui.column):
    """Handle image upload."""
    if e.content:
        image_bytes = e.content.read()
        state["current_image"] = image_bytes

        # Show preview
        await process_image(image_bytes, result_container)


@ui.page("/")
def main_page():
    """Main dashboard page."""

    # Header
    with ui.header().classes("items-center justify-between"):
        ui.label("🛰️ Spatial ViT Analysis").classes("text-2xl font-bold")

        # API status indicator
        with ui.row().classes("items-center gap-2"):
            ui.icon("cloud_queue").classes("text-2xl")
            ui.label("API: Checking...").classes("text-sm")

    # Main content
    with ui.row().classes("w-full h-full"):
        # Sidebar
        with ui.left_drawer(bordered=True).classes("bg-gray-100 p-4").style(
            "width: 300px"
        ):
            ui.label("Configuration").classes("text-xl font-bold mb-4")

            # API URL
            ui.label("RunPod API URL").classes("font-bold mt-4")
            ui.input(
                placeholder="http://runpod-api-url:8000", value=state["api_url"]
            ).classes("w-full").bind_value(state, "api_url")

            ui.separator()

            # Mode selection
            ui.label("Mode").classes("font-bold mt-4")
            ui.radio(
                ["single", "compare"],
                value="single",
                on_change=lambda e: state.update({"mode": e.value}),
            ).props("inline")

            ui.separator()

            # Model selection
            ui.label("Model 1").classes("font-bold mt-4")
            ui.select(
                ["vit_base", "resnet50", "swin_t"],
                value="vit_base",
                label="Select Model 1",
            ).classes("w-full").bind_value(state, "model1")

            # Show model 2 selector if in compare mode
            model2_selector = (
                ui.select(
                    ["vit_base", "resnet50", "swin_t"],
                    value="resnet50",
                    label="Select Model 2",
                )
                .classes("w-full")
                .bind_value(state, "model2")
            )
            model2_selector.set_visibility(state["mode"] == "compare")

            ui.separator()

            # Top-k slider
            ui.label("Top-K Predictions").classes("font-bold mt-4")
            ui.slider(min=1, max=10, value=3).classes("w-full").bind_value(
                state, "top_k"
            )
            ui.label().bind_text_from(state, "top_k", lambda x: f"Top {x} predictions")

        # Main content area
        with ui.column().classes("flex-1 p-8"):
            ui.label("Satellite Image Classification").classes(
                "text-3xl font-bold mb-2"
            )
            ui.label(
                "Upload a satellite image to compare Vision Transformer and "
                "CNN predictions"
            ).classes("text-gray-600 mb-6")

            # Upload section
            with ui.card().classes("w-full p-6 mb-6"):
                ui.label("Upload Image").classes("text-xl font-bold mb-4")

                result_container = ui.column().classes("w-full")

                upload = (
                    ui.upload(
                        auto_upload=True,
                        on_upload=lambda e: handle_upload(e, result_container),
                    )
                    .classes("w-full")
                    .props("accept=image/*")
                )

                with ui.row().classes("gap-2 mt-2"):
                    ui.button("Choose File", on_click=upload.open).props(
                        "color=primary"
                    )
                    ui.label("Supports: PNG, JPG, JPEG").classes(
                        "text-sm text-gray-500"
                    )

            # Results section
            with ui.card().classes("w-full p-6"):
                with result_container:
                    ui.label("👆 Upload an image to get started").classes(
                        "text-xl text-gray-400 text-center p-8"
                    )

    # Footer
    with ui.footer().classes("bg-gray-100 p-4"):
        with ui.row().classes("w-full items-center justify-center"):
            ui.label(
                "Built with ❤️ for Vision Transformer spatial analysis research"
            ).classes("text-sm text-gray-600")


# Check API health on startup
async def check_api_health():
    """Check if the RunPod API is accessible."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{state['api_url']}/health")
            return response.status_code == 200
    except Exception:
        return False


if __name__ in {"__main__", "__mp_main__"}:
    # Run the app
    ui.run(
        title="Spatial ViT Analysis",
        port=8080,
        reload=True,
        show=False,
    )
