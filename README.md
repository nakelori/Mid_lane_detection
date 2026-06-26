# Mid_lane_detection
An algorithm solving a problem of finding the real mid lane lines on a road, using different image angels and distances

The algorithm receives a json file containing different frames from the car's front camera in different distances and angels.
each frame consists a list with an RT matrix, a list of points (x,y,z coordinates) which assemble road lines, error rate for each point and confidence level for each line.
The goal is to assemble a unite 1 frame that consists the real midlane lines, considering the error rate, confidence and position of each line from each frame.

solution:
1. rotating each line set and each error rate set to the desired final coordinates by multiplying the RT matrix.
2. clustering and labeling the lines by the distance from a reference point of one of the lines (here the reference was Z = 20m distance beacuse lines shorter than 20m are probably noise), in curved lines there's a need of multiple reference points. also i decided that lines that are 1.5m away or less in the horizontal axis X will be clustered together.
3. Creating the final line by first spliting each line into segments, and approximating the closest point of each line to it's segment (for expample Z value of 32.5m will be mapped to 30m).
4. next is averging the X values and Y values of each segment by weighted avergae of each line's confidence value.
