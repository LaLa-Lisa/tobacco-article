"""Визуализация нормализованных результатов независимо от SDK."""

def plot_prediction(image, prediction):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    from PIL import Image
    if isinstance(image, np.ndarray):
        rgb = image[..., ::-1]
    elif isinstance(image, Image.Image):
        rgb = image.convert('RGB')
    else:
        with Image.open(image) as source:
            rgb = source.convert('RGB')
    _, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(rgb)
    for index, box in enumerate(prediction.boxes_xyxy):
        color = plt.get_cmap('tab10')(index % 10)
        x1, y1, x2, y2 = box
        ax.add_patch(Rectangle((x1, y1), x2-x1, y2-y1, fill=False, edgecolor=color))
        label = ''
        if prediction.class_ids is not None:
            cls = int(prediction.class_ids[index])
            label = (prediction.names or {}).get(cls, str(cls))
        ax.text(x1, y1, f'{label} {prediction.scores[index]:.2f}', color=color)
        if prediction.masks is not None:
            overlay = np.zeros((*prediction.image_shape, 4))
            overlay[prediction.masks[index]] = (*color[:3], 0.35)
            ax.imshow(overlay)
    ax.axis('off')
    return ax
