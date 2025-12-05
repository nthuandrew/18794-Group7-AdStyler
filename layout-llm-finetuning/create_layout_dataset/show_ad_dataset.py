from datasets import load_from_disk

ad_data = load_from_disk("../ad-dataset/AdImageNet")


print(ad_data["train"][1])