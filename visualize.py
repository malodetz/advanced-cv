# visualize.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def visualize_prediction(image, boxes, classes, scores):
    fig, ax = plt.subplots(1)
    ax.imshow(image)
    
    for box, cls, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        rect = patches.Rectangle(
            (x1*image.shape[1], y1*image.shape[0]),
            width*image.shape[1],
            height*image.shape[0],
            linewidth=2,
            edgecolor='r',
            facecolor='none'
        )
        ax.add_patch(rect)
        plt.text(
            x1*image.shape[1], 
            y1*image.shape[0], 
            f'{cls}: {score:.2f}',
            color='white',
            bbox=dict(facecolor='red', alpha=0.5)
        )
    
    plt.show()
