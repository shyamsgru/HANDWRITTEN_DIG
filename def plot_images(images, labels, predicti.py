def plot_images(images, labels, predictions, num=10):
    plt.figure(figsize=(10, 4))
    for i in range(num):
        plt.subplot(2, 10, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(f'True: {labels[i]}\nPred: {predictions[i]}')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plot a few test images and their predictions
plot_images(x_test, y_true, y_pred_classes)
