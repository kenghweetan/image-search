# Code referenced from https://huggingface.co/docs/transformers/tasks/image_classification
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    ViTForImageClassification,
    AutoModelForImageClassification,
    DefaultDataCollator,
    TrainingArguments,
    Trainer,
)
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from datasets import load_dataset, get_dataset_split_names
import evaluate
import numpy as np

accuracy = evaluate.load("accuracy")

# Load
print(get_dataset_split_names("ashraq/fashion-product-images-small"))
dataset = load_dataset("ashraq/fashion-product-images-small", split="train")
dataset = dataset.train_test_split(test_size=0.2)

labels = dataset["train"].features["label"].names
label2id, id2label = dict(), dict()
for i, label in enumerate(labels):
    label2id[label] = str(i)
    id2label[str(i)] = label

print(len(labels))

""" # Preprocess dataset
checkpoint = "timm/resnet50.a1_in1k"
image_processor = AutoImageProcessor.from_pretrained(checkpoint)


normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
size = (
    image_processor.size["shortest_edge"]
    if "shortest_edge" in image_processor.size
    else (image_processor.size["height"], image_processor.size["width"])
)
_transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])


def transforms(dataset):
    dataset["pixel_values"] = [
        _transforms(img.convert("RGB")) for img in dataset["image"]
    ]
    del dataset["image"]
    return dataset


dataset = dataset.with_transform(transforms)
data_collator = DefaultDataCollator()


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return accuracy.compute(predictions=predictions, references=labels)


# Train
model = AutoModelForImageClassification.from_pretrained(
    checkpoint,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)
 """
