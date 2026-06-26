import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.axes import Axes
import json
from dataclasses import dataclass

@dataclass
class Line:
    """
    Represents a predicted 3D line with associated confidence and error estimation.

    Attributes:
        points (np.ndarray): An Nx3 numpy array containing 3D points defining the line.
        confidence (float): A scalar representing the probability of the line being real.
        errest (np.ndarray): An Nx3 numpy array representing the error estimation (uncertainty) for each point.
    """
    points: np.ndarray
    confidence: float
    errest: np.ndarray
    
    def plot(self, plot_err_est: bool=True, color: str=None):
        """
        Plots the line in the current figure.

        - The transparency of the line is based on the confidence squared.
        - If `plot_err_est` is True, the uncertainty region is also displayed.

        Args:
            plot_err_est (bool): Whether to plot error estimation around the line.
            color (str): Specify color string
        """
        line_obj = plt.plot(self.points[:, 0], self.points[:, 2], alpha=self.confidence**2, color=color)[0]
        if plot_err_est:
            poly = np.concatenate([self.points + self.errest, self.points[::-1] - self.errest[::-1]], axis=0)
            plt.fill(poly[:, 0], poly[:,2], color=line_obj.get_color(), alpha=0.3*self.confidence**2)


    def interpolate_in_image(self, line: np.ndarray, first_Z: float, errest: np.ndarray):
        """
        Interpolates out of image points to the image end.

        Args:
            line (np.ndarray): Points to interpolate with respect to
            first_Z (float): First Z in image
            errest (np.ndarray): error estimation to interpolate w.r.t the line interpolation
        Returns:
            np.ndarray: 2D NumPy array of the new interpolated line in image
            np.ndarray: 2D NumPy array of the new interpolated error estimation in image
        """
        if np.all(line[:,2] >= first_Z):
            return line, errest
        first_in_image = np.where(line[:,2] >= first_Z)[0]
        if len(first_in_image) == 0:
            return [], []
        if first_in_image[0] == 0:
            return line, errest
        last_ooi = first_in_image[0] - 1
        first_in_image = first_in_image[0]
        alpha = (line[first_in_image,2] - first_Z) / (line[first_in_image,2] - line[last_ooi,2])
        out_line = np.copy(line)
        out_errest = np.copy(errest)
        out_line[last_ooi] = out_line[last_ooi] * alpha + out_line[first_in_image] * (1. - alpha)
        out_errest[last_ooi] = out_errest[last_ooi] * alpha + out_errest[first_in_image] * (1. - alpha)
        out_line = out_line[last_ooi:]
        out_errest = out_errest[last_ooi:]
        return out_line, out_errest
        

    def plot_on_image(self, origin: np.ndarray, focal_len: float, bounds: np.ndarray, camH: float, plot_err_est: bool=True, color: str=None):
        """
        Projects and plots the 3D line onto a 2D image.

        Args:
            origin (np.ndarray): A 2-element array specifying the origin offset in image space.
            focal_len (float): The camera focal length used for pinhole projection.
            plot_err_est (bool): Whether to plot the projected error estimation.
            color (str): Specify color string
        """
        first_Z_buffer = 1.
        first_Z = focal_len * camH / origin[1] + first_Z_buffer
        points_cam, errest = self.interpolate_in_image(self.points, first_Z, self.errest)
        x = focal_len * points_cam[:,0] / points_cam[:,2] + origin[0]
        y = focal_len * points_cam[:,1] / points_cam[:,2] + origin[1]
        line_obj = plt.plot(x, y, alpha=self.confidence**2)[0]

        if plot_err_est:
            poly = np.concatenate([points_cam + errest, points_cam[::-1] - errest[::-1]], axis=0)
            x = focal_len * poly[:,0] / poly[:,2] + origin[0]
            y = focal_len * poly[:,1] / poly[:,2] + origin[1]
            plt.fill(x, y, color=line_obj.get_color(), alpha=0.5*self.confidence)

    
@dataclass
class FrameLines:
    """
    Represents a frame containing predicted lines and an RT matrix.

    Attributes:
        lines (list[Line]): A list of predicted `Line` objects in the frame.
        RT (np.ndarray): A 4x4 RT matrix that transforms the lines into the current frameâ€™s coordinate system.
    """
    lines: list[Line]
    RT: np.ndarray
    
    def plot(self, plot_err_est=True, color: str=None):
        """
        Plots all lines in the frame using the Line::plot method.

        Args:
            plot_err_est (bool): Whether to plot error estimation around the lines.
            color (str): Specify color string
        """
        for line in self.lines:
            line.plot(plot_err_est=plot_err_est, color=color)
    
    
class DataReader:
    def __init__(self, json_path: str, image_path: str):
        self.read_data(json_path)
        self.read_image(image_path)


    def read_image(self, image_path: str='') -> np.ndarray:
        """
        Reads an image from the given file path, extracts the first channel (e.g., red or grayscale),
        scales pixel values from [0, 1] to [0, 255], and converts the result to integers.

        Parameters:
            image_path (str): Path to the image file.

        Returns:
            np.ndarray: 2D NumPy array of integer pixel values in the range [0, 255].
        """
        if image_path == '':
            image_path = 'image.png'
        img = mpimg.imread(image_path)
        img = (img[:,:,0] * 255).astype(int)
        self.img = img
    

    def read_data(self, json_path: str) -> list[FrameLines]:
        """
        Reads the test case JSON file and parses it.

        Args:
            json_path (str): Path to the JSON file.

        Returns:
            list[FrameLines]: A list of `FrameLines` objects, each representing a frame with predicted lines.
        """
        with open(json_path, "r") as f:
            data_dict = json.load(f)
        frames = []
        frames_dict = {k: v for k,v in data_dict.items() if k.startswith("frame_")}
        for frame in frames_dict.values():
            current_frame = FrameLines([], np.array(frame["RT"]))
            for line in frame["lines"]:
                current_frame.lines.append(Line(np.array(line["points"]), line["confidence"], np.array(line["errest"])))
            frames.append(current_frame)
        self.frames = frames
        self.origin = np.array([data_dict["img_origin_x"], data_dict["img_origin_y"]])
        self.img_bounds = np.array([[data_dict["img_bound_left"], data_dict["img_bound_right"]],
                                    [data_dict["img_bound_bottom"], data_dict["img_bound_top"]]])
        self.focal_len = data_dict["img_focal_len"]
        self.camH = data_dict["camH"]


    def plot_frames(self, frames: list[FrameLines]=None, plot_err_est: bool=True, color: str=None, ax: Axes=None):
        """
        Plots all the FrameLines from a list.

        Args:
            frames (list[FrameLines]): A list of `FrameLines` objects to be plotted.
            plot_err_est (bool): Whether to visualize error estimation.
            color (str): Specify color string
            ax (matplotlib.axes.Axes): axes to plot on
        """
        if frames is None:
            frames = self.frames

        if ax is not None:
            plt.sca(ax)
        else:
            plt.figure()
        for frame in frames:
            frame.plot(plot_err_est=plot_err_est, color=color)


    def plot_lines(self, lines: list[Line]=None, plot_err_est: bool=True, color: str=None, ax: Axes=None):
        """
        Plots all the Lines from a list.

        Args:
            frames (list[Line]): A list of `Line` objects to be plotted.
            plot_err_est (bool): Whether to visualize error estimation.
            color (str): Specify color string
            ax (matplotlib.axes.Axes): axes to plot on
        """
        if ax is not None:
            plt.sca(ax)
        else:
            plt.figure()
        for line in lines:
            line.plot(plot_err_est=plot_err_est, color=color)


    def plot_lines_on_image(self, lines: list[Line], plot_err_est: bool=True, color: str=None, ax: Axes=None):
        """
        Projects and plots lines over the loaded image using camera projection.

        Args:
            lines (List[Line]): Lines to be projected and drawn.
            color (str): Specify color string
            ax (matplotlib.axes.Axes): axes to plot on
        """
        if ax is not None:
            plt.sca(ax)
        else:
            plt.figure()
        plt.imshow(self.img, origin="lower", cmap="gray")
        for line in lines:
            line.plot_on_image(self.origin, self.focal_len, self.img_bounds, self.camH, plot_err_est=plot_err_est, color=color)
        plt.axis([self.img_bounds[0,0] + self.origin[0], self.img_bounds[0,1] + self.origin[0], self.img_bounds[1,0] + self.origin[1], self.img_bounds[1,1] + self.origin[1]])


def convert_to_current_frame(orig_frame: FrameLines) -> FrameLines:
    """
    Transforms a FrameLines object into the current frame's coordinate system.

    This function must be implemented.

    Args:
        orig_frame (FrameLines): The original frame with lines in its local coordinates.

    Returns:
        FrameLines: Transformed frame in current coordinate system.
    """
    new_lines = []
    for line in orig_frame.lines:
        #matrix mult to turn into shared coordinates
        pts = np.hstack([line.points, np.ones((len(line.points), 1))])
        new_pts = (orig_frame.RT @ pts.T).T[:, :3] 

        err = np.hstack([line.errest, np.zeros((len(line.errest), 1))])
        new_err = (orig_frame.RT @ err.T).T[:, :3]

        #gather all in a list
        new_lines.append(Line(new_pts, line.confidence, new_err))
    return FrameLines(new_lines, orig_frame.RT)
    

def cluster_lines(frames: list[FrameLines]) -> list[Line]:
    """
    Clusters and merges lines across frames into a unified set of lines in the current frameâ€™s coordinate system.
    
    - Clusters similar lines together.
    - Merges overlapping lines while considering confidence and error estimation.
    - Filters out unreliable lines with low confidence.

    This function must be implemented.
    
    Args:
        frames (list[FrameLines]): A list of `FrameLines` objects containing predicted lines and RT matrices.

    Returns:
        list[Line]: A list of `Line` objects representing the best set of lines in the current frame.
    """
    all_lines = [line for frame in frames for line in frame.lines]

    #taking 20m distance reference for bias at the Y axis of the lines
    REF_Z = 20.0
    def y_at_ref(line):
        return np.interp(REF_Z, line.points[:, 2], line.points[:, 0])

    all_lines.sort(key=y_at_ref)

    #Clustering lines in 1.5m distance into the same cluster
    THRESHOLD = 1.5
    clusters = []
    current_cluster = [all_lines[0]]
    all_lines = [line for line in all_lines if y_at_ref(line) < 5] #filter the far sidelines
    for line in all_lines[1:]:
        if abs(y_at_ref(line) - y_at_ref(current_cluster[0])) < THRESHOLD:
            current_cluster.append(line)
        else:
            clusters.append(current_cluster)
            current_cluster = [line]
    clusters.append(current_cluster)

    #filtering lines with confidence lower than 50%
    CONF_THRESHOLD = 0.5
    clusters = [c for c in clusters if np.mean([l.confidence for l in c]) >= CONF_THRESHOLD]

    def merge_cluster(group: list[Line]) -> Line:
        z_min = min(line.points[0, 2] for line in group)
        z_max = max(line.points[-1, 2] for line in group)
        z_grid = np.linspace(z_min, z_max, 20)

        sum_x = np.zeros(20)
        sum_y = np.zeros(20)
        sum_ex = np.zeros(20)
        sum_ey = np.zeros(20)
        sum_w = np.zeros(20)

        for line in group:
            mask = (z_grid >= line.points[0, 2]) & (z_grid <= line.points[-1, 2])
            if not np.any(mask):
                continue
            w = line.confidence
            z_vals = z_grid[mask]
            sum_x[mask] += w * np.interp(z_vals, line.points[:, 2], line.points[:, 0])
            sum_y[mask] += w * np.interp(z_vals, line.points[:, 2], line.points[:, 1])
            sum_ex[mask] += w * np.interp(z_vals, line.points[:, 2], line.errest[:, 0])
            sum_ey[mask] += w * np.interp(z_vals, line.points[:, 2], line.errest[:, 1])
            sum_w[mask] += w

        #avreging the X and Y values weighted by confidence
        valid = sum_w > 0
        avg_x = sum_x[valid] / sum_w[valid]
        avg_y = sum_y[valid] / sum_w[valid]
        avg_ex = sum_ex[valid] / sum_w[valid]
        avg_ey = sum_ey[valid] / sum_w[valid]
        avg_z = z_grid[valid]

        points = np.column_stack([avg_x, avg_y, avg_z])
        errest = np.column_stack([avg_ex, avg_ey, np.zeros(valid.sum())])
        avg_conf = float(np.mean([l.confidence for l in group]))
        return Line(points, avg_conf, errest)

    return [merge_cluster(c) for c in clusters]


if __name__ == "__main__":
    data_reader = DataReader("lane_data.json", "image.png")
    converted_frames = []
    for frame in data_reader.frames:
        converted_frames.append(convert_to_current_frame(frame))

    data_reader.plot_frames(converted_frames)
    plt.gca().set_title("All lines in current frame")
    plt.show()

    clustered_lines = cluster_lines(converted_frames)
    data_reader.plot_lines(clustered_lines)
    plt.gca().set_title("Clustered lines in current frame")
    plt.ylim(0, 200)

    data_reader.plot_lines_on_image(clustered_lines)
    plt.gca().set_title("Clustered lines in current frame - Frame view")
    plt.show()

    lines = []
    for frame in converted_frames:
        lines = lines + frame.lines
    data_reader.plot_lines_on_image(lines)
    plt.gca().set_title("All lines in current frame - Frame view")   
    plt.show() 
