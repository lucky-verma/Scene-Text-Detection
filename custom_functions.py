import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# clear the cell output
from IPython.display import clear_output


def draw_poly(json_file, save_path):
    """
    This functions plots the boxes on the relevant images and saves them to the output folder
    """

    with open(json_file) as f:
        data = json.load(f)

        # get the boxes for each image in img folder
        for i in data.keys():
            boxes = data[i]['FastResults'][0]["bboxes"]

            # image path is key of the dictionary
            image = cv2.imread(i)

            for pts in boxes:
                pts = np.array(pts).reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(image, [pts], True, (0, 255, 0), 2)

            # plot the image
            plt.figure(figsize=(20, 10))
            plt.imshow(image)

            # save the image
            plt.savefig(save_path + i.split('/')[-1][:-4] + '.png')

            # clear cell output
            clear_output()

    # # Driver code
    # draw_poly('final_results.json', save_path='final_result/')

