disease_names = [
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites Two-spotted_spider_mite",
    "Tomato_Target_Spot",
    "Tomato_Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_Tomato_mosaic_virus",
    "Tomato__healthy"
]

# _ karakterini boşluk olarak değiştir
cleaned_names = [name.replace("_", " ") for name in disease_names]

# Sonuçları yazdır
print(cleaned_names)