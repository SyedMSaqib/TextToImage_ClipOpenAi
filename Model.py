import os
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from sklearn.metrics.pairwise import cosine_similarity
import pickle


captions_file = r'C:\Users\SAQIB\PycharmProjects\Resnet50\GenerativeAi\Flickr_Data\Flickr_Data\Flickr_TextData\Flickr8k.token.txt'


images_path = r'C:\Users\SAQIB\PycharmProjects\Resnet50\GenerativeAi\Flickr_Data\Flickr_Data\Images'


embeddings_file = 'image_embeddings.pkl'


def load_captions(captions_file):
    with open(captions_file, 'r') as f:
        captions_data = f.readlines()
    captions_dict = {}
    for line in captions_data:
        parts = line.strip().split('\t')
        image_file, caption = parts[0].split('#')[0], parts[1]
        if image_file not in captions_dict:
            captions_dict[image_file] = []
        captions_dict[image_file].append(caption)
    return captions_dict

captions_dict = load_captions(captions_file)


clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def get_clip_text_embedding(text):
    inputs = clip_processor(text=text, return_tensors="pt", padding=True)
    outputs = clip_model.get_text_features(**inputs)
    return outputs.squeeze().detach().numpy()


def get_clip_image_embedding(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = clip_processor(images=image, return_tensors="pt")
    outputs = clip_model.get_image_features(**inputs)
    return outputs.squeeze().detach().numpy()


def save_embeddings(embeddings, filename):
    with open(filename, 'wb') as f:
        pickle.dump(embeddings, f)


def load_embeddings(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


image_files = os.listdir(images_path)


if os.path.exists(embeddings_file):
    print("Loading image embeddings from file...")
    image_embeddings = load_embeddings(embeddings_file)
else:
    print("Processing images and saving embeddings to file...")

    image_embeddings = {}
    total_images = len(image_files)
    print(f"Total number of images to process: {total_images}")

    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(images_path, image_file)
        image_embeddings[image_file] = get_clip_image_embedding(image_path)
        print(f"Processed {idx+1}/{total_images}: {image_file}")


    save_embeddings(image_embeddings, embeddings_file)


def calculate_similarity(text_embedding, image_embedding):
    similarity = cosine_similarity([text_embedding], [image_embedding])[0][0]
    print(f"Similarity: {similarity}")  # Debugging statement
    return similarity


def find_best_match(input_text, image_embeddings):
    text_embedding = get_clip_text_embedding(input_text)
    print(f"Text Embedding: {text_embedding.shape}")
    best_score = -1
    best_image_file = None
    for image_file, image_embedding in image_embeddings.items():
        print(f"Comparing with image: {image_file}")
        score = calculate_similarity(text_embedding, image_embedding)
        print(f"Score for {image_file}: {score}")
        if score > best_score:
            best_score = score
            best_image_file = image_file
    print(f"Best score: {best_score}")
    return best_image_file

# CLI Interface
if __name__ == "__main__":
    while True:
        input_text = input("Enter a description (or 'quit' to exit): ")
        if input_text.lower() == 'quit':
            break
        best_image = find_best_match(input_text, image_embeddings)
        if best_image:
            print(f"Best match: {best_image}")
            best_image_path = os.path.join(images_path, best_image)
            best_image = Image.open(best_image_path)
            best_image.show()
        else:
            print("No match found.")
