import numpy as np

def dilate(image, kernel_size=3, iterations=0):
    kernel_size = 3 if kernel_size < 3 else kernel_size

    kernel = np.full(shape=(kernel_size, kernel_size), fill_value=255)
    padding = (kernel_size - 1) // 2

    # Add padding to the image
    padded_image = np.pad(image, padding, 'constant', constant_values=0)
    height_reduce, width_reduce = (padded_image.shape[0] - image.shape[0], padded_image.shape[1] - image.shape[1])

    dilated_image = np.zeros(image.shape)
    for i in range(padded_image.shape[0] - height_reduce):
        for j in range(padded_image.shape[1] - width_reduce):
            if (padded_image[i:(i+kernel_size), j:(j+kernel_size)] == kernel).any():
                dilated_image[i,j] = 255
            else:
                dilated_image[i,j] = 0

    dilated_image = dilated_image.reshape(image.shape)

    if iterations > 1:
        dilated_image = dilate(dilated_image, kernel_size, iterations - 1)

    return np.uint8(dilated_image)

def erode(image, kernel_size=3, iterations=0):
    kernel_size = 3 if kernel_size < 3 else kernel_size

    kernel = np.full(shape=(kernel_size, kernel_size), fill_value=0)
    padding = (kernel_size - 1) // 2

    # Add padding to the image
    padded_image = np.pad(image, padding, 'constant', constant_values=0)
    height_reduce, width_reduce = (padded_image.shape[0] - image.shape[0], padded_image.shape[1] - image.shape[1])

    eroded_image = np.zeros(image.shape)
    for i in range(padded_image.shape[0] - height_reduce):
        for j in range(padded_image.shape[1] - width_reduce):
            if (padded_image[i:(i+kernel_size), j:(j+kernel_size)] == kernel).any():
                eroded_image[i,j] = 0
            else:
                eroded_image[i,j] = 255

    eroded_image = eroded_image.reshape(image.shape)

    if iterations > 1:
        eroded_image = dilate(eroded_image, kernel_size, iterations - 1)

    return np.uint8(eroded_image)
