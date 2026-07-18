import kagglehub

path = kagglehub.competition_download(
    "physionet-ecg-image-digitization"
)

print("Dataset downloaded to:", path)