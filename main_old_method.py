import Kentner_PA2 as kpa
import Kentner_Project_Util as kpu
import sys, getopt
import os
import cv2
import numpy as np


# Deprecated function; used for following the edges found via FFT and applying RGB-based constraints
def DFS(root_y, root_x, visited, img_base, img_fft, thld, max_y, max_x):
    # Initialize a Node object with the root's information
    root_node = kpa.Node()
    root_node.location = [root_y, root_x]
    root_node.rgb = img_base[root_y, root_x]

    # Set up two stacks: One for exploration, and one for revisions
    stack_explore = []
    stack_revise = []

    # Get the start and end points for exploring neighbors (avoid out-of-bounds exception)
    neigh_min_y = root_y - 1 if root_y - 1 > -1 else root_y
    neigh_max_y = root_y + 1 if root_y + 1 < max_y else root_y
    neigh_min_x = root_x - 1 if root_x - 1 > -1 else root_x
    neigh_max_x = root_x + 1 if root_x + 1 < max_x else root_x

    # Add all bordering pixels to the root; explore edge pixels, and constrain on non-edge pixels
    i = neigh_min_y
    j = neigh_min_x
    while i < neigh_max_y:
        while j < neigh_max_x:
            # Skip the current root
            if i != root_y and j != root_x:
                # Separate the edge neighbors from the non-edge neighbors based on threshold thld
                if img_fft[i, j] > thld:
                    # Edge pixel
                    stack_explore.append([i, j])
                
                else:
                    # Non-edge pixel
                    stack_revise.append([i, j])

                # Edge or not, this pixel may be added to visited
                visited[i, j] = True

            # Next column
            j += 1

        # Next row
        i += 1

    k = 0
    skip = -1
    while k < len(stack_revise):
        cur_y, cur_x = stack_revise[k]
        if k != skip and compare_pixels(img_base[root_y, root_x], img_base[cur_y, cur_x], root_node.similarity_index):
            # Relax the similarity requirement (makes revisions less likely)
            root_node.similarity_index -= 0.05
            
        else:
            # No revisions; check next neighbor
            k += 1

    # After correcting this pixel, explore each of the edge-pixel neighbors using DFS
    # print(stack_explore)
    img_new = None
    for neighbor in stack_explore:
        DFS(neighbor[0], neighbor[1], visited, img_base, img_fft, thld, max_y, max_x)
        
        
def compare_pixels(px1, px2, sim):
    changes = False
    mag_1 = (int(px1[0]) + int(px1[1]) + int(px1[2])) / 3
    mag_2 = (int(px2[0]) + int(px2[1]) + int(px2[2])) / 3

    if mag_1 < mag_2:
        changes = True
        px1[0] = px2[0]
        px1[1] = px2[1]
        px1[2] = px2[2]

    return changes


# Main function
def main(argv, argc, preprocessing=False):
    print(argv)
    
    # Get the filename from the command line
    file_name = ''
    sharp_name = ''
    try:
        # If this is training data, a pair of filenames should be provided
        if argc > 1:
            # Get the command-line arguments as strings
            file_name = str(argv[1])
            
            # Check if a sharp "target" image was included
            if argc > 2:
                sharp_name = str(argv[2])
            
    except:
        print('ERROR: Make sure that the filename for the image was passed as a command-line argument')
        return 1
    
    # Check if file_name was initialized
    if file_name == '':
        print('ERROR: Filename not initialized')
        return 2
    else:
        print('Filename:', file_name)
    
    # Get the filtered image
    img_filtered, img_color = kpu.FFT_Test(file_name, 25, csp_demo=True)
    
    # Create a bitmap to indicate whether a pixel has been visited 
    # https://stackoverflow.com/questions/5891410/numpy-array-initialization-fill-with-identical-values
    visited = np.full(img_filtered.shape, False)
    
    # Send the filtered image to the DFS
    R, C = img_filtered.shape
    print(R, C)
    for y in range(R):
        for x in range(C):
            DFS(y, x, visited, img_color, img_filtered, 20, R, C)
    
    # Write the output image
    if img_color is not None:
        print('Hear')
        cv2.imwrite('IMG_CSP.jpg', img_color)


# Runs the main function
# https://www.tutorialspoint.com/python/python_command_line_arguments.htm
if __name__ == '__main__':
    # Pass all command-line arguments to main; add a True flag to blur an image
    if len(sys.argv) > 1:
        main(sys.argv, len(sys.argv))
    else:
        print('ERROR: Input filename expected as a command-line parameter')


