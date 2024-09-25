# from retinaface import  RetinaFace
# from arcface import ArcFace
from deepface import DeepFace

# help(RetinaFace)
# print(type(ArcFace))   #Module

# help(DeepFace)
keys = ['verified', 'distance', "msx_threshold_to_verify", "model",
        "detector_backend", "similarity_metric", "facial_area", "time"]
# You get some funny verification test mj1 with other mj 😅
result = DeepFace.verify(
    "asset/Face_Recon_Dataset/MichaelJackson/MJ5.jpg", "asset/Face_Recon_Dataset/MichaelJackson/MJ1.jpg", model_name="ArcFace",detector_backend='retinaface')
# print(result)
# Display the results in a more human-friendly format
print("Face Verification Result:")
print(f"Faces Match: {result['verified']}")  # True or False
print(f"Distance: {result['distance']:.4f}")
print(f"Threshold: {result['threshold']:.2f}")
print(f"Model: {result['model']}")
print(f"Detector Backend: {result['detector_backend']}")
print(f"Similarity Metric: {result['similarity_metric']}")

print(f"Facial Areas:")
for img, areas in result['facial_areas'].items():
    print(f"  {img}:")
    print(f"    Coordinates: (x={areas['x']}, y={areas['y']}, w={areas['w']}, h={areas['h']})")
    if areas['left_eye'] and areas['right_eye']:
        print(f"    Left Eye: {areas['left_eye']}")
        print(f"    Right Eye: {areas['right_eye']}")
    else:
        print("    Eyes: Not detected")
print(f"Time taken: {result['time']:.2f} seconds")


# Embeddings:
img = 'asset/Face_Recon_Dataset/Eminem/Eminem1.jpg'
# Get the embedding
embeddings = DeepFace.represent(img)
print(embeddings)
