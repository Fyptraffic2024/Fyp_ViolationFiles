import os
import glob
import json
import gradio as gr

def get_original_videos():
    original_dir = "videos/original"
    if not os.path.exists(original_dir):
        return []
    return sorted([f for f in os.listdir(original_dir) if f.endswith((".mp4", ".avi", ".mov"))])

def analyze_video(selected_video):
    if not selected_video:
        return [], [], None
    base_name = os.path.splitext(selected_video)[0]
    thumbnails = sorted(glob.glob(f"videos/thumbnails/{base_name}_*.jpg"))
    # Return thumbnails, thumbnails (state), and the path of the selected video
    return thumbnails, thumbnails, os.path.join("videos/original", selected_video)

def display_clip(evt: gr.SelectData, thumbnails):
    if not thumbnails or evt.index >= len(thumbnails):
        return None, {}
    
    selected_thumbnail = thumbnails[evt.index]
    thumbnail_name = os.path.basename(selected_thumbnail)
    clip_id = os.path.splitext(thumbnail_name)[0]
    
    clip_path = f"videos/extracted_clips/{clip_id}.mp4"
    json_path = f"videos/violations/{clip_id}.json"
    
    violation_data = {}
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            violation_data = json.load(f)
    
    return clip_path, violation_data

with gr.Blocks(title="Video Violation Analyzer") as demo:
    gr.Markdown("## Video Violation Analysis System")
    
    thumbnail_state = gr.State([])
    
    with gr.Row():
        with gr.Column(scale=1):
            video_selector = gr.Dropdown(
                label="Select Original Video",
                choices=get_original_videos(),
                interactive=True
            )
            video_select = gr.Video(label="Selected Video", interactive=False, width="60%", elem_id="autoplay_original_video")
            analyze_btn = gr.Button("Analyze", variant="primary")
        
        with gr.Column(scale=1):
            clip_gallery = gr.Gallery(
                label="Extracted Clips",
                columns=2,
                object_fit="contain",
                height="300px",
                min_width="200px",
            )
    
    with gr.Row():
        video_player = gr.Video(label="Selected Clip", interactive=False, 
                               width="60%", elem_id="autoplay_video")
        violation_display = gr.JSON(label="Violation Record", height=300)

    # Update analyze_btn.click to include video_select as an output
    analyze_btn.click(
        lambda: ([], [], None),
        inputs=None,
        outputs=[clip_gallery, thumbnail_state, video_select]
    ).then(
        analyze_video,
        inputs=video_selector,
        outputs=[clip_gallery, thumbnail_state, video_select]
    ).then(
        None,
        js="""
        () => {
            // Wait for video element to load then play
            setTimeout(() => {
                const videoElement = document.querySelector('#autoplay_video video');
                if (videoElement) {
                    videoElement.muted = true;  // Auto-play requires muted audio
                    videoElement.play();
                }
            }, 100);
        }
        """
    )
    
    clip_gallery.select(
        display_clip,
        inputs=[thumbnail_state],
        outputs=[video_player, violation_display]
    ).then(
        None,
        js="""
        () => {
            // Wait for video element to load then play
            setTimeout(() => {
                const videoElement = document.querySelector('#autoplay_video video');
                if (videoElement) {
                    videoElement.muted = true;  // Auto-play requires muted audio
                    videoElement.play();
                }
            }, 100);
        }
        """
    )

if __name__ == "__main__":
    demo.launch(
        share = True,
    )