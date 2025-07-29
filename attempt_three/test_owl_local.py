import os
from PIL import Image
from local_services.owlv2_local import OWLViT

def test_owl_local():
    # === Config ===
    image_path = "test_data/pour_test/rgb/00000.jpg"   # change if needed
    output_path = "test_data/pour_test/owlvit_test_output"
    os.makedirs(output_path, exist_ok=True)

    text_prompts = ["bowl"]  # try changing this to test other objects

    # === Load image ===
    image = Image.open(image_path).convert("RGB")

    # === Initialize detector ===
    detector = OWLViT()

    # === Run detection ===
    detections = detector.detect_objects(
        image=image,
        text_prompts=text_prompts,
        bbox_score_top_k=10,
        bbox_conf_threshold=0.13
    )

    print(f"Detected {len(detections)} objects:")
    for det in detections:
        print(f" - {det['box_name']} | score: {det['score']:.3f} | bbox: {det['bbox']}")

    # === Draw and save result ===
    result_img = detector.draw_detections(image, detections)
    save_path = os.path.join(output_path, "detection_result.png")
    Image.fromarray(result_img).save(save_path)
    print(f"Saved visualized detection to {save_path}")

if __name__ == "__main__":
    test_owl_local()