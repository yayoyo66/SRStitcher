import open_clip
from PIL import Image
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity


# Mean Pooling - Take attention mask into account for correct averaging
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def get_CCSscores(img1path, img2path, input1path, input2path):
    device = 'cuda'

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens')
    model2 = AutoModel.from_pretrained('sentence-transformers/bert-base-nli-mean-tokens').to(device)

    model, _, transform = open_clip.create_model_and_transforms(
        model_name="coca_ViT-L-14",
        pretrained="mscoco_finetuned_laion2B-s13B-b90k"
    )
    model = model.to(device)

    def split_image(img):
        width, height = img.size

        part_width = width // 2
        part_height = height // 2

        box1 = (0, 0, part_width, part_height)
        box2 = (part_width, 0, width, part_height)
        box3 = (0, part_height, part_width, height)
        box4 = (part_width, part_height, width, height)

        image1 = img.crop(box1)
        image2 = img.crop(box2)
        image3 = img.crop(box3)
        image4 = img.crop(box4)

        image1 = transform(image1).unsqueeze(0).to(device)
        image2 = transform(image2).unsqueeze(0).to(device)
        image3 = transform(image3).unsqueeze(0).to(device)
        image4 = transform(image4).unsqueeze(0).to(device)

        return image1, image2, image3, image4

    with torch.no_grad(), torch.cuda.amp.autocast():
        im1 = Image.open(img1path).convert("RGB")
        im2 = Image.open(img2path).convert("RGB")

        image11, image21, image31, image41 = split_image(im1)
        im_text11 = model.generate(image11)
        im_text21 = model.generate(image21)
        im_text31 = model.generate(image31)
        im_text41 = model.generate(image41)

        image12, image22, image32, image42 = split_image(im2)
        im_text12 = model.generate(image12)
        im_text22 = model.generate(image22)
        im_text32 = model.generate(image32)
        im_text42 = model.generate(image42)

        im_text11 = open_clip.decode(im_text11[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
        im_text21 = open_clip.decode(im_text21[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
        im_text31 = open_clip.decode(im_text31[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
        im_text41 = open_clip.decode(im_text41[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")

        im_text12 = open_clip.decode(im_text12[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
        im_text22 = open_clip.decode(im_text22[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
        im_text32 = open_clip.decode(im_text32[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
        im_text42 = open_clip.decode(im_text42[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")

        # local CCS
        sentences1 = [im_text11 + im_text21 + im_text31 + im_text41, im_text12 + im_text22 + im_text32 + im_text42]
        print(sentences1)

        encoded_input1 = tokenizer(sentences1, padding=True, truncation=True, return_tensors='pt').to(device)
        model_output1 = model2(**encoded_input1)
        sentence_embeddings1 = mean_pooling(model_output1, encoded_input1['attention_mask']).cpu()
        scores1 = cosine_similarity([sentence_embeddings1[0]], sentence_embeddings1[1:])
        print(scores1)

        # global CCS

        input1 = Image.open(input1path).convert("RGB")
        input1 = transform(input1).unsqueeze(0).to(device)
        input2 = Image.open(input2path).convert("RGB")
        input2 = transform(input2).unsqueeze(0).to(device)

        input1_text = model.generate(input1)
        input2_text = model.generate(input2)
        input1_text = open_clip.decode(input1_text[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")
        input2_text = open_clip.decode(input2_text[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")

        im1 = transform(im1).unsqueeze(0).to(device)
        im1_text = model.generate(im1)
        im1_text = open_clip.decode(im1_text[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")

        sentences2 = [im1_text, input1_text + input2_text]
        print(sentences2)

        encoded_input2 = tokenizer(sentences2, padding=True, truncation=True, return_tensors='pt').to(device)

        model_output2 = model2(**encoded_input2)

        sentence_embeddings2 = mean_pooling(model_output2, encoded_input2['attention_mask']).cpu()

        scores2 = cosine_similarity([sentence_embeddings2[0]], sentence_embeddings2[1:])
        print(scores2)

        return (np.mean([scores1[0][0], scores2[0][0]]))