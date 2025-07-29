from owlvit_sam_pipeline import OwlVitSamPipeline

test_image = "test_samples/00000.jpg"
prompts = ["bowl", "teapot"]
output_dir = "output/test_run"

pipeline = OwlVitSamPipeline()
results = pipeline.run(test_image, prompts, output_dir=output_dir)

for r in results:
    print(f"Detected: {r['box_name']} | Score: {r['score']:.3f}")